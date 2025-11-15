"""
Segmentation Preview Widget for Segmentation Loading
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider
from PyQt6.QtCore import pyqtSignal, Qt

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.seg_loading.ui.frame_selection_ui import Ui_constructRoi
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg


class FrameSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for previewing and confirming segmentation.
    
    This is the final step in the segmentation loading process where users
    can preview the loaded segmentation and confirm it before proceeding.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    frame_selected = pyqtSignal(int, int)  # Selected frame index, brightness value
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_constructRoi()
        self._image_data = image_data
        self._matplotlib_canvas: Optional[FigureCanvas] = None
        self._frame = 0
        self._all_frames = self._image_data.sc_bmode if self._image_data.sc_bmode is not None else self._image_data.bmode
        assert image_data.rf_data.ndim == 3, "Image data must be 3D for segmentation preview"
        
        # Animation and performance variables
        self._animation: Optional[anim.FuncAnimation] = None
        self._im_artist = None  # The image artist for fast updates
        self._target_frame = 0  # Target frame for smooth transitions
        self._frame_update_pending = False
        
        # Brightness control variables
        self._brightness_slider: Optional[QSlider] = None
        self._brightness_label: Optional[QLabel] = None
        self._brightness_value_label: Optional[QLabel] = None
        self._brightness_value = 50  # Default brightness (0-100)
        self._original_frames = self._all_frames.copy()  # Store original frames for brightness adjustment
        
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for segmentation preview only - use the main layout
        self.setLayout(self._ui.main_layout)
        
        # Configure stretch factors for confirmation
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.frame_preview_layout, 10)
        
        # Ensure the layout fills the entire widget
        self._ui.main_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.main_layout.setSpacing(0)
        self._ui.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.full_screen_layout.setSpacing(0)
        
        # Update UI to reflect inputted image, phantom, and frames
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)
        self._ui.frame_slider.setRange(0, self._all_frames.shape[0] - 1)
        self._ui.frame_slider.setValue(self._frame)
        self._ui.cur_frame_label.setText(str(self._frame + 1))
        self._ui.total_frames_label.setText(str(self._all_frames.shape[0]))

        # Setup matplotlib canvas for frame preview
        self._setup_matplotlib_canvas()
        
        # Add brightness control
        self._setup_brightness_control()
        
        # Display frame preview
        self._initialize_frame_preview()
        
    def _setup_brightness_control(self) -> None:
        """Setup brightness control slider and label."""
        # Replace the existing frame controls layout with a new layout
        # First, clear the existing frame controls layout
        while self._ui.frameControlsLayout.count():
            child = self._ui.frameControlsLayout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        
        # Create a new horizontal layout for all controls
        main_layout = QHBoxLayout()
        
        # Create left side controls layout (vertical)
        left_controls = QVBoxLayout()
        
        # Brightness control row
        brightness_row = QHBoxLayout()
        
        # Create brightness label
        self._brightness_label = QLabel("Brightness:")
        self._brightness_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: rgb(255, 255, 255);
                background-color: rgba(255, 255, 255, 0);
            }
        """)
        self._brightness_label.setMinimumSize(80, 41)
        self._brightness_label.setMaximumSize(80, 41)
        self._brightness_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        # Create brightness slider
        self._brightness_slider = QSlider()
        self._brightness_slider.setOrientation(Qt.Orientation.Horizontal)
        self._brightness_slider.setRange(0, 100)
        self._brightness_slider.setValue(self._brightness_value)
        self._brightness_slider.setMinimumSize(200, 41)
        self._brightness_slider.setMaximumSize(200, 41)
        self._brightness_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: 2px 0;
                border-radius: 3px;
            }
        """)
        
        # Create brightness value label
        self._brightness_value_label = QLabel(str(self._brightness_value))
        self._brightness_value_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: rgb(255, 255, 255);
                background-color: rgba(255, 255, 255, 0);
            }
        """)
        self._brightness_value_label.setMinimumSize(30, 41)
        self._brightness_value_label.setMaximumSize(30, 41)
        self._brightness_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        brightness_row.addWidget(self._brightness_label)
        brightness_row.addWidget(self._brightness_slider)
        brightness_row.addWidget(self._brightness_value_label)
        
        # Frame control row
        frame_row = QHBoxLayout()
        
        # Create frame label
        frame_label = QLabel("Frame:")
        frame_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: rgb(255, 255, 255);
                background-color: rgba(255, 255, 255, 0);
            }
        """)
        frame_label.setMinimumSize(80, 41)
        frame_label.setMaximumSize(80, 41)
        frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        frame_row.addWidget(frame_label)
        frame_row.addWidget(self._ui.frame_slider)
        frame_row.addWidget(self._ui.cur_frame_label)
        frame_row.addWidget(self._ui.of_frames_label)
        frame_row.addWidget(self._ui.total_frames_label)
        
        # Add both rows to left controls
        left_controls.addLayout(brightness_row)
        left_controls.addLayout(frame_row)
        
        # Add left controls to main layout
        main_layout.addLayout(left_controls)
        
        # Add accept button on the right
        main_layout.addWidget(self._ui.accept_frame_button)
        
        # Replace the frame controls layout
        self._ui.frameControlsLayout.addLayout(main_layout)
        
    def _setup_matplotlib_canvas(self) -> None:
        """Setup matplotlib canvas for high-performance frame display."""
        # Create matplotlib figure and canvas with optimized settings
        fig = plt.figure(figsize=(8, 6))        
        self._matplotlib_canvas = FigureCanvas(fig)
        self._matplotlib_canvas.figure.patch.set_facecolor((0, 0, 0, 0))
        self._matplotlib_canvas.draw()
        
        # Add canvas to the preview frame widget
        layout = QHBoxLayout(self._ui.im_display_frame)
        layout.addWidget(self._matplotlib_canvas)
        self._ui.im_display_frame.setLayout(layout)
    
    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.frame_slider.valueChanged.connect(self._on_frame_changed)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        self._ui.accept_frame_button.clicked.connect(self._on_frame_selected)
        
        # Connect brightness slider
        if self._brightness_slider:
            self._brightness_slider.valueChanged.connect(self._on_brightness_changed)
            
    def _on_brightness_changed(self, value: int) -> None:
        """Handle brightness slider change."""
        self._brightness_value = value
        if self._brightness_value_label:
            self._brightness_value_label.setText(str(value))
        # No need to call _apply_brightness_adjustment() - brightness is applied at display time
        self._force_frame_update()
        
    def _apply_brightness_adjustment(self) -> None:
        """Apply brightness adjustment to the displayed frames using exponential function."""
        # Don't modify _all_frames permanently - apply brightness only at display time
        # This ensures frame changes don't affect brightness
        pass
            
    def _get_brightness_adjusted_frame(self, frame_index: int) -> np.ndarray:
        """Get a brightness-adjusted frame for display without modifying the original data."""
        if frame_index >= self._original_frames.shape[0]:
            return self._original_frames[0]  # Fallback to first frame
            
        # Get the original frame
        original_frame = self._original_frames[frame_index]
        
        # Apply brightness adjustment only to this frame for display
        # Calculate exponential coefficient based on brightness slider (0-100)
        # Map 0-100 to 0.05-5.0 for more intense exponential range
        exp_coefficient = 0.05 + (self._brightness_value / 100.0) * 4.95
        
        # Apply exponential brightness adjustment
        frame = original_frame.astype(np.float32)
        
        # Normalize to 0-1 range for exponential operation
        normalized_frame = frame / 255.0
        
        # Apply exponential function: I_out = I_in^exp_coefficient
        # This will make the image brighter as exp_coefficient increases
        frame = np.power(normalized_frame, 1.0 / exp_coefficient)
        
        # Scale back to 0-255 range and clip
        frame = frame * 255.0
        frame = np.clip(frame, 0, 255)
        
        return frame.astype(np.uint8)
            
    def _initialize_frame_preview(self) -> None:
        """Initialize the frame preview with optimized matplotlib setup."""
        if not self._matplotlib_canvas:
            return
        
        # Calculate aspect ratio
        if self._image_data.sc_bmode is not None:
            width = self._all_frames.shape[2] * self._image_data.sc_lateral_res
            height = self._all_frames.shape[1] * self._image_data.sc_axial_res
        else:
            width = self._all_frames.shape[2] * self._image_data.lateral_res
            height = self._all_frames.shape[1] * self._image_data.axial_res
        self.aspect = width / height

        try:
            fig = self._matplotlib_canvas.figure
            fig.clear()
            self._ax = fig.add_subplot(111)
            self._ax.set_position([0, 0, 1, 1])
            self._ax.axis("off")

            # Create the initial image artist - this will be reused for all frames
            displayed_im = self._get_brightness_adjusted_frame(self._frame)
            self._im_artist = self._ax.imshow(displayed_im, cmap="gray", animated=True, vmin=0, vmax=255)
            
            # Set proper aspect ratio
            extent = self._im_artist.get_extent()
            self._ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/self.aspect)
            
            # Setup the animation for smooth frame updates
            self._setup_frame_animation()
            
            # Initial draw
            self._matplotlib_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error displaying image: {e}")
            
    def _setup_frame_animation(self) -> None:
        """Setup FuncAnimation for high-performance frame updates."""
        if self._animation:
            self._animation.event_source.stop()
        
        # Create animation with fast update interval for smooth transitions
        self._animation = anim.FuncAnimation(
            self._matplotlib_canvas.figure,
            self._update_frame_animated,
            interval=16,  # ~60 FPS for smooth updates
            blit=True,    # Use blitting for maximum performance
            repeat=False,
            cache_frame_data=False  # Don't cache to save memory
        )
        
    def _update_frame_animated(self, frame_num) -> list:
        """Animation update function for smooth frame transitions."""
        if not self._frame_update_pending:
            return [self._im_artist] if self._im_artist else []
        
        # Update to target frame
        if self._frame != self._target_frame:
            self._frame = self._target_frame
            self._update_frame_display(self._frame)
        
        self._frame_update_pending = False
        return [self._im_artist] if self._im_artist else []
            
    def _on_frame_changed(self, value: int) -> None:
        """Handle frame slider change with optimized performance."""
        self._target_frame = value
        self._frame_update_pending = True
        # Animation will handle the actual update efficiently
            
    def _update_frame_display(self, frame_index: int) -> None:
        """Update the frame display with consistent parameters."""
        if self._im_artist:
            # Get brightness-adjusted frame for display without modifying original data
            displayed_im = self._get_brightness_adjusted_frame(frame_index)
            self._im_artist.set_array(displayed_im)
            self._im_artist.set_clim(0, 255)  # Ensure consistent dynamic range
            self._ui.cur_frame_label.setText(str(frame_index + 1))
            
    def _force_frame_update(self) -> None:
        """Force immediate frame update without animation (for initialization)."""
        self._update_frame_display(self._frame)
        self._matplotlib_canvas.draw_idle()
        
    def _cleanup_animation(self):
        """Stop and clean up animation safely."""
        if self._animation:
            try:
                self._animation.event_source.stop()
                self._animation = None
            except:
                # Ignore errors if already destroyed
                self._animation = None

    def closeEvent(self, event) -> None:
        """Clean up animation when widget is closed."""
        self._cleanup_animation()
        super().closeEvent(event)

    def hideEvent(self, event):
        """Clean up animation when widget is hidden."""
        self._cleanup_animation()

    def showEvent(self, event):  
        """Restart animation when widget is shown."""
        if self._im_artist and not self._animation:
            self._setup_frame_animation()
            
    def __del__(self):
        """Ensure animation is cleaned up when object is destroyed."""
        try:
            self._cleanup_animation()
        except:
            pass  # Ignore errors during cleanup

    def _on_frame_selected(self) -> None:
        """Handle frame selection confirmation."""
        # Make sure we're on the correct frame before confirming
        if self._frame != self._target_frame:
            self._frame = self._target_frame
            self._force_frame_update()
        
        # Don't store brightness-adjusted frames - keep original frames for ROI drawing
        # The brightness control is only for frame selection display
        

        
        self.frame_selected.emit(self._frame, self._brightness_value)

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
