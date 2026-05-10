"""
Segmentation Preview Widget for Segmentation Loading
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSlider, QFrame, QCheckBox, QLabel
from PyQt6.QtCore import pyqtSignal, Qt

from ...mvc.base_view import BaseViewMixin
from ..ui.roi_preview_ui import Ui_confirmRoi
from engines.ceus.src.data_objs import UltrasoundImage, CeusSeg

# Philips CEUS Colormap: Grayscale -> Red -> Yellow
philips_colors = [
    (0.0, 0.0, 0.0),    # 0% - Black
    (0.4, 0.4, 0.4),    # 40% - Gray
    (0.8, 0.0, 0.0),    # 80% - Red
    (1.0, 1.0, 0.0)     # 100% - Yellow
]
philips_cmap = LinearSegmentedColormap.from_list("philips_ceus", philips_colors)


class ROIPreviewWidget(QWidget, BaseViewMixin):
    """
    Widget for previewing and confirming segmentation.
    
    This is the final step in the segmentation loading process where users
    can preview the loaded segmentation and confirm it before proceeding.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    segmentation_confirmed = pyqtSignal(object)
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    apply_preprocs_preview = pyqtSignal(list)  # List of dicts with 'name' and 'kwargs' keys

    def __init__(self, image_data: UltrasoundImage, seg_data: CeusSeg,
                 parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_confirmRoi()
        self._image_data = image_data
        self._seg_data = seg_data
        self._matplotlib_canvas: Optional[FigureCanvas] = None
        self._frame = 0
        self._all_frames = self._image_data.pixel_data

        # Animation and performance variables
        self._animation: Optional[anim.FuncAnimation] = None
        self._im_artist = None  # The image artist for fast updates
        self._roi_mask_artist = None  # The ROI artist for fast updates
        self._frame_update_pending = False
        
        # Enhancement parameters
        self._clahe_clip_limit = 1.2
        self._gamma = 1.5
        self._width_scale = 1.0
        
        # Enhancement parameters
        self._clahe_clip_limit = 1.2
        self._gamma = 1.5
        self._alpha = 125
        self._use_philips_ceus = False
        self._enhanced_cache = None # Cache for enhanced current frame
        self._enhanced_cache_idx = -1
        
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

        # Update UI to reflect inputted image and frames
        self._ui.scan_name_input.setText(self._image_data.scan_name)
        self._ui.frame_slider.setRange(0, self._all_frames.shape[0] - 1)
        self._ui.frame_slider.setValue(self._frame)
        self._ui.cur_frame_label.setText(str(np.round(self._frame*self._image_data.frame_rate, decimals=2)))
        self._ui.total_frames_label.setText(str(np.round(self._all_frames.shape[0]*self._image_data.frame_rate, decimals=2)))

        # Setup matplotlib canvas for frame preview
        self._setup_matplotlib_canvas()
        self._setup_enhancement_controls()
        
        # Display frame preview
        self._initialize_frame_preview()
        
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
        self._ui.confirm_roi_button.clicked.connect(self.segmentation_confirmed.emit)
            
    def _initialize_frame_preview(self) -> None:
        """Initialize the frame preview with optimized matplotlib setup."""
        if not self._matplotlib_canvas:
            return
        
        # Calculate aspect ratio
        width = self._all_frames.shape[2] * self._image_data.pixdim[1]
        height = self._all_frames.shape[1] * self._image_data.pixdim[0]
        self.aspect = width / height

        try:
            fig = self._matplotlib_canvas.figure
            fig.clear()
            self._ax = fig.add_subplot(111)
            self._ax.set_position([0, 0, 1, 1])
            self._ax.axis("off")

            # Create the initial image artist - this will be reused for all frames
            self._displayed_im = self._all_frames[self._frame]
            self._seg_mask = np.zeros(self._all_frames.shape[1:-1] + (4,))
            self._seg_mask[..., 3] = self._seg_data.seg_mask * self._alpha
            self._seg_mask[..., 0] = self._seg_data.seg_mask * 255
            self._seg_mask = self._seg_mask.astype(int)
            self._im_artist = self._ax.imshow(self._displayed_im, cmap="gray", animated=True, zorder=1)
            self._roi_mask_artist = self._ax.imshow(self._seg_mask, zorder=10)

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

        def init():
            # Return all artists that will be animated
            return [self._im_artist, self._roi_mask_artist]

        self._animation = anim.FuncAnimation(
            self._matplotlib_canvas.figure,
            self._update_frame_animated,
            init_func=init,
            interval=16,   # ~60 FPS
            blit=True,
            repeat=False,
            cache_frame_data=False
        )
        
    def _update_frame_animated(self, frame_num) -> list:
        """Animation update function for smooth frame transitions."""
        if not self._frame_update_pending:
            return [self._im_artist, self._roi_mask_artist]
        
        self._update_frame_display(self._frame)
        self._frame_update_pending = False
        return [self._im_artist, self._roi_mask_artist]
        
    def _on_width_changed(self, value: int) -> None:
        """Handle width scale change."""
        self._width_scale = value / 10.0
        if hasattr(self, 'width_val_lbl'):
            self.width_val_lbl.setText(f"{self._width_scale:.1f}")
        self._update_aspect_ratio()

    def _update_aspect_ratio(self) -> None:
        """Update the aspect ratio of the main axes based on width scale."""
        if not hasattr(self, '_ax') or self._ax is None:
            return
            
        # Calculate base physical aspect ratio
        width_phys = self._all_frames.shape[2] * self._image_data.pixdim[1] * self._width_scale
        height_phys = self._all_frames.shape[1] * self._image_data.pixdim[0]
        
        if height_phys != 0:
            new_aspect = width_phys / height_phys
            extent = self._im_artist.get_extent()
            self._ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/new_aspect)
            self._matplotlib_canvas.draw_idle()

    def _setup_enhancement_controls(self) -> None:
        """Add enhancement sliders beside the frame slider in a single horizontal line."""
        # Container frame for enhancement controls
        enh_group = QFrame()
        enh_group.setStyleSheet("background-color: rgba(255, 255, 255, 0); border: none;")
        
        # Main horizontal layout for the enhancement section
        container_layout = QHBoxLayout(enh_group)
        container_layout.setContentsMargins(0, 0, 15, 0)
        container_layout.setSpacing(15)

        def create_compact_control(label_text, min_val, max_val, current_val, callback):
            # Widget to hold label, slider, and value in ONE line
            ctrl_widget = QWidget()
            ctrl_layout = QHBoxLayout(ctrl_widget)
            ctrl_layout.setContentsMargins(0, 0, 0, 0)
            ctrl_layout.setSpacing(5)
            
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 10px; color: white; font-weight: bold;")
            ctrl_layout.addWidget(lbl)
            
            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(current_val)
            slider.setStyleSheet(self._ui.frame_slider.styleSheet())
            slider.setFixedWidth(70)
            slider.setFixedHeight(12)
            slider.valueChanged.connect(callback)
            ctrl_layout.addWidget(slider)

            val_lbl = QLabel(f"{current_val/10.0:.1f}")
            val_lbl.setStyleSheet("color: #3498db; font-weight: bold; font-size: 10px;")
            val_lbl.setMinimumWidth(22)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
            ctrl_layout.addWidget(val_lbl)
            
            return ctrl_widget, slider, val_lbl

        # Create controls
        clahe_w, self.clahe_slider, self.clahe_val_lbl = create_compact_control(
            "CLAHE", 1, 100, int(self._clahe_clip_limit * 10), self._on_clahe_changed
        )
        gamma_w, self.gamma_slider, self.gamma_val_lbl = create_compact_control(
            "GAMMA", 1, 40, int(self._gamma * 10), self._on_gamma_changed
        )
        width_w, self.width_slider, self.width_val_lbl = create_compact_control(
            "WIDTH", 1, 50, int(self._width_scale * 10), self._on_width_changed
        )
        alpha, self.alpha_slider, self.alpha_val_lbl = create_compact_control(
            "ALPHA", 0, 255, 0, self._on_alpha_changed
        )
        
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(self._alpha)
        
        # Add to horizontal layout
        container_layout.addWidget(clahe_w)
        container_layout.addWidget(gamma_w)
        container_layout.addWidget(width_w)
        container_layout.addWidget(alpha)

        # Pseudo coloring toggle nicely aligned
        if not (self._image_data.pixel_data.ndim == 4 and self._image_data.pixel_data.shape[3] > 1):
            # For RGB images, disable the Philips colormap option since it doesn't apply
            self.philips_check = QCheckBox("Pseudo coloring")
            self.philips_check.setStyleSheet("color: white; font-weight: bold; font-size: 11px;")
            self.philips_check.stateChanged.connect(self._on_philips_toggled)
            container_layout.addWidget(self.philips_check)

        # Add to the layout beside the frame slider (below the image)
        self._ui.frameControlsLayout.insertWidget(0, enh_group)

    def _on_clahe_changed(self, value: int) -> None:
        """Handle CLAHE clip limit change."""
        self._clahe_clip_limit = value / 10.0
        if hasattr(self, 'clahe_val_lbl'):
            self.clahe_val_lbl.setText(f"{self._clahe_clip_limit:.1f}")
        self._invalidate_enhancement_cache()

    def _on_gamma_changed(self, value: int) -> None:
        """Handle gamma change."""
        self._gamma = value / 10.0
        if hasattr(self, 'gamma_val_lbl'):
            self.gamma_val_lbl.setText(f"{self._gamma:.1f}")
        self._invalidate_enhancement_cache()
        
    def _on_alpha_changed(self, value: int) -> None:
        self._alpha = int(value)
        if hasattr(self, 'alpha_val_lbl'):
            self.alpha_val_lbl.setText(f"{self._alpha}")
        if not hasattr(self, '_seg_mask'):
            return
        self._seg_mask[..., 3] = (self._seg_data.seg_mask * self._alpha).astype(int)
        self._frame_update_pending = True

    def _invalidate_enhancement_cache(self) -> None:
        """Invalidate the enhancement cache (e.g. when parameters change)."""
        self._enhanced_cache = None
        self._enhanced_cache_idx = -1
        self._frame_update_pending = True  # Trigger update to request new enhanced frame

    def _on_philips_toggled(self, state: int) -> None:
        self._use_philips_ceus = state == Qt.CheckState.Checked.value
        if self._im_artist:
            new_cmap = philips_cmap if self._use_philips_ceus else 'gray'
            self._im_artist.set_cmap(new_cmap)
            
            # # Force a call to set_array() to dirty the artist for the blitter
            # self._update_frame_display(self._frame)
            
            # Flag the animation loop to blit the newly dirtied image on its next tick
            self._frame_update_pending = True

    def _request_enhanced_frame(self, frame_2d: np.ndarray) -> np.ndarray:
        """Enhance a 2D image frame using backend engine functions."""
        # Create a temporary UltrasoundImage for the current frame
        temp_im = UltrasoundImage(self._image_data.scan_path)
        temp_im.pixel_data = frame_2d.T[None].T.copy()  # Add back time dimension for processing
        temp_im.pixdim = self._image_data.pixdim
        temp_im.frame_rate = self._image_data.frame_rate

        clahe_preproc_dict = {
            'name': 'enhance_clahe',
            'image_data': temp_im,
            'frame_ix': self._frame,
            'kwargs': {
                'clip_limit': self._clahe_clip_limit,
                'tile_grid_size': (8, 8),
            }
        }

        gamma_preproc_dict = {
            'name': 'enhance_gamma',
            'image_data': None,  # signal to reuse the already CLAHE-enhanced image (all preprocs in the same batch share the same image input)
            'frame_ix': self._frame,
            'kwargs': {
                'gamma': self._gamma,
            }
        }

        preproc_dicts = [clahe_preproc_dict, gamma_preproc_dict]
        self.apply_preprocs_preview.emit(preproc_dicts) # synchronous call to apply the enhancements and update the cache via the connected slot

    def _on_frame_changed(self, value: int) -> None:
        """Handle frame slider change with optimized performance."""
        self._frame = value
        self._frame_update_pending = True

    def update_enhancement_cache(self, enhanced_frame: np.ndarray, frame: int) -> None:
        """Receives enhanced frame from controller and stores it for display."""
        self._enhanced_cache = enhanced_frame.T[0].T   # shape is (1, H, W) from the temp_im — take the single frame
        self._enhanced_cache_idx = frame
        self._frame_update_pending = True  # Flag to update display on next animation tick
            
    def _update_frame_display(self, frame_index: int) -> None:
        if self._im_artist:
            if self._enhanced_cache is None or self._enhanced_cache_idx != frame_index:
                # synchronously update self._enhanced_cache with the new enhanced frame 
                # for the current index
                self._request_enhanced_frame(self._all_frames[frame_index])
            self._im_artist.set_array(self._enhanced_cache)
            self._roi_mask_artist.set_array(self._seg_mask)

            self._ui.cur_frame_label.setText(
                str(np.round(frame_index * self._image_data.frame_rate, decimals=2))
            )
        
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
            
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
