"""
ROI Drawing Widget for Segmentation Loading
"""

import os
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy.interpolate as interpolate

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QFileDialog, QSlider, QLabel, QVBoxLayout

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.seg_loading.ui.roi_drawing_ui import Ui_constructRoi
from engines.qus.quantus.data_objs import UltrasoundRfImage


class RoiDrawingWidget(QWidget, BaseViewMixin):
    """
    Widget for drawing ROI (Region of Interest).
    
    This widget allows users to draw ROI on the ultrasound image
    for creating segmentation masks.
    Designed to be used within the main application widget stack.
    
    Features:
    - Freehand drawing with points (prevents overshooting with minimum distance)
    - Rectangle drawing with drag
    - Freehand drawing with continuous drag
    - Visual feedback for distance constraints
    - Keyboard shortcuts for distance adjustment (+/-/= keys)
    """
    
    # Signals for communicating with controller
    seg_file_selected = pyqtSignal(dict)  # seg_type_name
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, frame: int = 0, brightness: int = 50,
                 parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_constructRoi()
        self._image_data = image_data
        self._matplotlib_canvas: Optional[FigureCanvas] = None
        self._current_roi_coords: Optional[tuple] = None
        self._current_roi_border: Optional[tuple] = None   # Complete set of all boundary points for the ROI for all formats
        self._drawing = False  # Flag to track if drawing is in progress
        self._frame = frame  # Frame number for multi-frame images
        self._displayed_im: np.ndarray = None  # Placeholder for the image to be displayed
        
        # Brightness control variables
        self._brightness_slider: Optional[QSlider] = None
        self._brightness_label: Optional[QLabel] = None
        self._brightness_value_label: Optional[QLabel] = None
        self._brightness_value = brightness  # Brightness value from previous menu
        self._original_displayed_im = None  # Store original image for brightness adjustment
        
        # Drawing mode tracking
        self._current_drawing_mode = None  # 'points', 'rectangle', 'freehand', or None
        
        # Drawing parameters
        self._min_point_distance = 5.0  # Minimum distance between points in pixels

        self._setup_ui()
        self._connect_signals()

    def _calculate_spline(self, xpts, ypts):
        """Calculate spline interpolation between points (smooth curves)."""
        if len(xpts) < 2:
            return xpts, ypts
            
        # Convert to numpy arrays
        xpts = np.array(xpts)
        ypts = np.array(ypts)
        
        # Create parameter t based on cumulative distance for natural parameterization
        distances = np.sqrt(np.diff(xpts)**2 + np.diff(ypts)**2)
        cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
        t_param = cumulative_dist / cumulative_dist[-1]
        
        # Use cubic spline interpolation for smooth curves
        if len(xpts) >= 4:
            # For 4 or more points, use cubic spline interpolation
            t_smooth = np.linspace(0, 1, 200)
            x_spline = interpolate.CubicSpline(t_param, xpts, bc_type='natural')
            y_spline = interpolate.CubicSpline(t_param, ypts, bc_type='natural')
            x_smooth = x_spline(t_smooth)
            y_smooth = y_spline(t_smooth)
        elif len(xpts) == 3:
            # For 3 points, use quadratic spline interpolation for smooth curves
            t_smooth = np.linspace(0, 1, 200)
            x_spline = interpolate.interp1d(t_param, xpts, kind='quadratic', fill_value='extrapolate')
            y_spline = interpolate.interp1d(t_param, ypts, kind='quadratic', fill_value='extrapolate')
            x_smooth = x_spline(t_smooth)
            y_smooth = y_spline(t_smooth)
        else:
            # For 2 points, use linear interpolation as fallback
            t_smooth = np.linspace(0, 1, 200)
            x_smooth = np.interp(t_smooth, t_param, xpts)
            y_smooth = np.interp(t_smooth, t_param, ypts)
            
        x_smooth = np.clip(x_smooth, 0, self._displayed_im.shape[1] - 1)
        y_smooth = np.clip(y_smooth, 0, self._displayed_im.shape[0] - 1)

        return x_smooth, y_smooth

    def _check_point_distance(self, new_x: float, new_y: float) -> bool:
        """
        Check if a new point is far enough from the last point to prevent overshooting.
        
        Args:
            new_x: X coordinate of the new point
            new_y: Y coordinate of the new point
            
        Returns:
            True if the point should be added, False if it's too close
        """
        if len(self._current_roi_coords) == 0:
            return True
            
        last_x, last_y = self._current_roi_coords[-1]
        distance = np.sqrt((new_x - last_x)**2 + (new_y - last_y)**2)
        return distance >= self._min_point_distance

    def set_min_point_distance(self, distance: float) -> None:
        """
        Set the minimum distance required between points when drawing.
        
        Args:
            distance: Minimum distance in pixels (default is 5.0)
        """
        if distance < 0:
            raise ValueError("Minimum distance must be non-negative")
        self._min_point_distance = distance

    def get_min_point_distance(self) -> float:
        """
        Get the current minimum distance between points.
        
        Returns:
            Current minimum distance in pixels
        """
        return self._min_point_distance

    def _update_drawing_status(self) -> None:
        """Update the drawing status to show current settings."""
        if self._current_drawing_mode == 'points':
            # Update window title to show current distance setting
            current_title = self.windowTitle()
            if "Min Distance:" not in current_title:
                self.setWindowTitle(f"{current_title} - Min Distance: {self._min_point_distance}px")
        else:
            # Reset window title when not in points mode
            base_title = "Select Region of Interest"
            self.setWindowTitle(base_title)

    def adjust_min_point_distance(self, delta: float) -> None:
        """
        Adjust the minimum point distance by the given delta.
        
        Args:
            delta: Amount to add/subtract from current distance
        """
        new_distance = self._min_point_distance + delta
        if new_distance >= 0:
            self._min_point_distance = new_distance
            self._update_drawing_status()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for ROI drawing only - use the main layout
        self.setLayout(self._ui.main_layout)
        
        # Configure stretch factors for ROI drawing
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.draw_roi_layout, 10)
        
        # Ensure the layout fills the entire widget
        self._ui.main_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.main_layout.setSpacing(0)
        self._ui.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.full_screen_layout.setSpacing(0)

        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

        # Organize menu objects
        self._save_seg_menu_objects = [
            'dest_folder_label', 'save_folder_input',
            'choose_save_folder_button', 'clear_save_folder_button',
            'roi_name_label', 'save_name_input',
            'save_roi_button', 'back_from_save_button',
        ]
        self._draw_types_objects = [
            'draw_rect_drag_type_button', 'draw_freehand_drag_type_button', 'draw_pts_type_button',
        ]
        self._draw_freehand_drag_objects = [
            'back_from_drag_button', 'save_drag_button',
        ]
        self._draw_rect_drag_objects = [
            'back_from_drag_button', 'save_drag_button',
        ]
        self._draw_pts_objects = [
            'undo_last_pt_button', 'clear_roi_button',
            'close_roi_button', 'back_from_pts_button', 'save_pts_button',
        ]
        self._roi_dims_objs = [
            'physical_roi_dims_label', 'physical_roi_height_label', 'physical_roi_height_val',
            'physical_roi_width_label', 'physical_roi_width_val'
        ]

        # Setup matplotlib canvas for ROI drawing
        self._setup_matplotlib_canvas()
        
        # Add brightness control
        self._setup_brightness_control()
        
        # Display image for ROI drawing
        self._display_image_for_roi()

        self._hide_save_menu()
        self._show_draw_type_selection()
        self._hide_roi_dims()
        self._show_image_dims()
        
    def _setup_matplotlib_canvas(self) -> None:
        """Setup matplotlib canvas for image display and ROI drawing."""
        # Create matplotlib figure and canvas
        fig, ax = plt.subplots(figsize=(8, 6))
        self._matplotlib_canvas = FigureCanvas(fig)
        self._matplotlib_canvas.figure.set_facecolor((0, 0, 0, 0))
        
        # Add canvas to the image frame widget
        layout = QHBoxLayout(self._ui.im_display_frame)
        layout.addWidget(self._matplotlib_canvas)
        self._ui.im_display_frame.setLayout(layout)

    def _setup_brightness_control(self) -> None:
        """Setup brightness control slider and label."""
        # Create a horizontal layout for brightness control
        brightness_layout = QHBoxLayout()
        
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
        
        # Add widgets to layout
        brightness_layout.addWidget(self._brightness_label)
        brightness_layout.addWidget(self._brightness_slider)
        brightness_layout.addWidget(self._brightness_value_label)
        
        # Add brightness control to the draw_roi_layout
        # Insert it at the beginning of the layout (after the title)
        self._ui.draw_roi_layout.insertLayout(1, brightness_layout)

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.back_button.clicked.connect(self.back_requested.emit)
        self._ui.draw_freehand_drag_type_button.clicked.connect(self._on_draw_freehand_drag)
        self._ui.draw_rect_drag_type_button.clicked.connect(self._on_draw_rect_drag)
        self._ui.draw_pts_type_button.clicked.connect(self._on_draw_pts)
        self._ui.back_from_drag_button.clicked.connect(self._show_draw_type_selection)
        self._ui.back_from_pts_button.clicked.connect(self._show_draw_type_selection)
        self._ui.save_drag_button.clicked.connect(self._show_save_menu)
        self._ui.save_pts_button.clicked.connect(self._show_save_menu)
        self._ui.choose_save_folder_button.clicked.connect(self._select_dest_folder)
        self._ui.clear_save_folder_button.clicked.connect(self._ui.save_folder_input.clear)
        self._ui.save_roi_button.clicked.connect(self._on_export_roi)
        self._ui.back_from_save_button.clicked.connect(self._show_draw_type_selection)
        
        # Connect brightness slider
        if self._brightness_slider:
            self._brightness_slider.valueChanged.connect(self._on_brightness_changed)

    def _on_brightness_changed(self, value: int) -> None:
        """Handle brightness slider change."""
        self._brightness_value = value
        if self._brightness_value_label:
            self._brightness_value_label.setText(str(value))
        self._apply_brightness_adjustment()
        self._update_display()

    def _apply_brightness_adjustment(self) -> None:
        """Apply brightness adjustment to the displayed image using exponential function."""
        if self._original_displayed_im is None:
            return
            
        # Calculate exponential coefficient based on brightness slider (0-100)
        # Map 0-100 to 0.05-5.0 for more intense exponential range
        exp_coefficient = 0.05 + (self._brightness_value / 100.0) * 4.95
        
        # Apply exponential brightness adjustment
        adjusted_im = self._original_displayed_im.astype(np.float32)
        
        # Normalize to 0-1 range for exponential operation
        normalized_im = adjusted_im / 255.0
        
        # Apply exponential function: I_out = I_in^exp_coefficient
        # This will make the image brighter as exp_coefficient increases
        adjusted_im = np.power(normalized_im, 1.0 / exp_coefficient)
        
        # Scale back to 0-255 range and clip
        adjusted_im = adjusted_im * 255.0
        adjusted_im = np.clip(adjusted_im, 0, 255)
        self._displayed_im = adjusted_im.astype(np.uint8)

    def _update_display(self) -> None:
        """Update the image display with current brightness."""
        if hasattr(self, '_ax') and self._ax:
            # Clear the axes and replot with consistent dynamic range
            self._ax.clear()
            self._plot_im_on_ax(self._ax)
            
            # Redraw ROI if it exists
            self._redraw_roi()
            
            if self._matplotlib_canvas:
                self._matplotlib_canvas.draw()

    def _redraw_roi(self) -> None:
        """Redraw the current ROI on the axes."""
        if not hasattr(self, '_current_roi_coords') or not self._current_roi_coords:
            return
            
        # Draw points
        if len(self._current_roi_coords) > 0:
            x, y = zip(*self._current_roi_coords)
            self._ax.scatter(x, y, color='red', marker='o', s=0.5, zorder=10)
            
            # Draw ROI based on drawing mode
            if self._current_drawing_mode == 'points':
                # Draw spline for points mode
                if len(self._current_roi_coords) > 1 and hasattr(self, '_current_roi_border') and self._current_roi_border:
                    border_x, border_y = zip(*self._current_roi_border)
                    self._ax.plot(border_x, border_y, color='cyan', linewidth=0.75)
            elif self._current_drawing_mode == 'rectangle':
                # Draw rectangle
                if len(self._current_roi_coords) == 2:
                    x0, y0 = self._current_roi_coords[0]
                    x1, y1 = self._current_roi_coords[1]
                    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='red')
                    rect.set_edgecolor('cyan')
                    self._ax.add_patch(rect)
            elif self._current_drawing_mode == 'freehand':
                # Draw freehand path
                if len(self._current_roi_coords) > 1 and hasattr(self, '_current_roi_border') and self._current_roi_border:
                    border_x, border_y = zip(*self._current_roi_border)
                    self._ax.plot(border_x, border_y, color='cyan')

    def _display_image_for_roi(self) -> None:
        """Display image for ROI drawing."""
        if not self._matplotlib_canvas:
            return
        
        try:
            # Get B-mode image data for display
            fig = self._matplotlib_canvas.figure
            fig.clear()
            self._ax = fig.add_subplot(111)
            self._ax.set_position([0, 0, 1, 1])

            im = self._image_data.sc_bmode if self._image_data.sc_bmode is not None else self._image_data.bmode
            if im.ndim == 2:
                self._original_displayed_im = im.copy()
            else:
                if self._frame < 0 or self._frame >= im.shape[0]:
                    raise ValueError(f"Frame {self._frame} is out of bounds for image with {im.shape[0]} frames")
                self._original_displayed_im = im[self._frame].copy()

            # Apply initial brightness adjustment
            self._apply_brightness_adjustment()

            self._plot_im_on_ax(self._ax)
            self._matplotlib_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error displaying image: {e}")

    def _plot_im_on_ax(self, ax) -> None:
        """Plot the image on the given axes."""
        if self._image_data.sc_bmode is not None:
            width = self._displayed_im.shape[1]*self._image_data.sc_lateral_res
            height = self._displayed_im.shape[0]*self._image_data.sc_axial_res
        else:
            width = self._displayed_im.shape[1]*self._image_data.lateral_res
            height = self._displayed_im.shape[0]*self._image_data.axial_res
        aspect = width/height
        im = ax.imshow(self._displayed_im, cmap="gray", vmin=0, vmax=255)
        extent = im.get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        ax.axis("off")
        


    def _on_export_roi(self) -> None:
        """Handle export ROI button click."""
        if self._current_roi_border == []:
            self.show_error("No ROI drawn to export")
            return
        save_folder = Path(self._ui.save_folder_input.text().strip())
        save_folder.mkdir(parents=True, exist_ok=True)
        dest_path = save_folder / (self._ui.save_name_input.text().strip() + ".pkl")
    
        x_spline, y_spline = zip(*self._current_roi_border)
        
        dict_to_save = {
            "Spline X": x_spline,
            "Spline Y": y_spline,
            "Scan Name": self._image_data.scan_name,
            "Phantom Name": self._image_data.phantom_name,
            "Frame": self._frame,
        }

        with open(dest_path, 'wb') as f:
            pickle.dump(dict_to_save, f)

        self.seg_file_selected.emit({
            'seg_path': str(dest_path),
            'seg_loader_kwargs': {}
        })


    def _on_draw_pts(self) -> None:
        """Handle freehand points drawing button click."""
        self._show_draw_pts()
        self._current_roi_coords = []; self._current_roi_border = []
        self._drawing = True
        self._current_drawing_mode = 'points'
        self._update_drawing_status()

        def draw_cur_roi():
            self._ax.clear()
            self._plot_im_on_ax(self._ax)
            if len(self._current_roi_coords):
                x, y = zip(*self._current_roi_coords)

                if len(x) > 1:
                    x_interp, y_interp = self._calculate_spline(x, y)
                    self._ax.plot(x_interp, y_interp, color='cyan', linewidth=0.75)
                    self._current_roi_border = list(zip(x_interp, y_interp))
                else:
                    self._current_roi_border = []
                self._ax.scatter(x, y, color='red', marker='o', s=0.5, zorder=10)

            self._matplotlib_canvas.draw()

        def close_roi():
            if self._drawing and len(self._current_roi_coords) > 1:
                self._ax.clear()
                self._plot_im_on_ax(self._ax)
                self._current_roi_coords.append(self._current_roi_coords[0])  # Close the path
                x, y = zip(*self._current_roi_coords)
                x_interp, y_interp = self._calculate_spline(x, y)
                self._current_roi_border = list(zip(x_interp, y_interp))
                self._ax.plot(x_interp, y_interp, color='cyan', linewidth=0.75)
                self._matplotlib_canvas.draw()
                self._drawing = False
        
        def clear_roi():
            self._current_roi_coords = []
            self._drawing = True
            draw_cur_roi()

        def undo_last_pt():
            if self._drawing and len(self._current_roi_coords):
                self._current_roi_coords.pop()
            if self._drawing:
                draw_cur_roi()

        def on_press(event):
            if self._drawing and event.inaxes == self._ax:
                # Check for duplicate points to prevent overshooting
                if not self._check_point_distance(event.xdata, event.ydata):
                    return
                
                self._current_roi_coords.append((event.xdata, event.ydata))
                draw_cur_roi()

        def on_motion(event):
            """Handle mouse motion to show distance feedback."""
            if self._drawing and event.inaxes == self._ax and len(self._current_roi_coords) > 0:
                # Check if mouse is too close to last point
                if not self._check_point_distance(event.xdata, event.ydata):
                    # Change cursor or add visual feedback
                    self._ax.figure.canvas.setCursor(Qt.CursorShape.ForbiddenCursor)
                else:
                    self._ax.figure.canvas.setCursor(Qt.CursorShape.CrossCursor)

        def on_key_press(event):
            """Handle keyboard shortcuts for distance adjustment."""
            if self._drawing and self._current_drawing_mode == 'points':
                if event.key == '+':
                    self.adjust_min_point_distance(1.0)
                elif event.key == '-':
                    self.adjust_min_point_distance(-1.0)
                elif event.key == '=':  # Reset to default
                    self.set_min_point_distance(5.0)

        self._ui.undo_last_pt_button.clicked.connect(undo_last_pt)
        self._ui.clear_roi_button.clicked.connect(clear_roi)
        self._ui.close_roi_button.clicked.connect(close_roi)
        self._cid_press = self._ax.figure.canvas.mpl_connect('button_press_event', on_press)
        self._cid_motion = self._ax.figure.canvas.mpl_connect('motion_notify_event', on_motion)
        self._cid_key = self._ax.figure.canvas.mpl_connect('key_press_event', on_key_press)

    def _on_draw_rect_drag(self) -> None:
        """Handle rectangle drag drawing button click."""
        self._show_draw_rect_drag()
        self._current_roi_coords = []
        self._drawing = False
        self._current_drawing_mode = 'rectangle'

        def on_press(event):
            if event.inaxes == self._ax:
                self._drawing = True
                self._current_roi_coords = [(event.xdata, event.ydata)]
        def on_motion(event):
            if self._drawing and event.inaxes == self._ax:
                if len(self._current_roi_coords) == 2:
                    self._current_roi_coords[1] = (event.xdata, event.ydata)
                else:
                    self._current_roi_coords.append((event.xdata, event.ydata))
                # Draw the rectangle as user drags
                self._ax.clear()
                self._plot_im_on_ax(self._ax)

                if len(self._current_roi_coords) > 1:
                    x0, y0 = self._current_roi_coords[0]
                    x1, y1 = event.xdata, event.ydata
                    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='red')
                    rect.set_edgecolor('cyan')
                    self._ax.add_patch(rect)

                    x0, x1 = sorted([int(x0), int(x1)])
                    y0, y1 = sorted([int(y0), int(y1)])
                    points_plotted_x = (
                        list(range(x0, x1 + 1))
                        + list(np.ones(y1 - y0 + 1).astype(int) * (x1))
                        + list(range(x1, x0 - 1, -1))
                        + list(np.ones(y1 - y0 + 1).astype(int) * x0)
                    )
                    points_plotted_y = (
                        list(np.ones(x1 - x0 + 1).astype(int) * y0)
                        + list(range(y0, y1 + 1))
                        + list(np.ones(x1 - x0 + 1).astype(int) * (y1))
                        + list(range(y1, y0 - 1, -1))
                    )
                    self._current_roi_border = list(zip(points_plotted_x, points_plotted_y))

                self._matplotlib_canvas.draw()

        def on_release(event):
            self._drawing = False
            if len(self._current_roi_coords) == 2:
                x0, y0 = self._current_roi_coords[0]
                x1, y1 = self._current_roi_coords[1]
                self._current_roi_coords = [(x0, y0), (x1, y1)]

                self._ax.clear()
                self._plot_im_on_ax(self._ax)
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='red')
                rect.set_edgecolor('cyan')
                self._ax.add_patch(rect)
                self._matplotlib_canvas.draw()

                x0, x1 = sorted([int(x0), int(x1)])
                y0, y1 = sorted([int(y0), int(y1)])
                points_plotted_x = (
                    list(range(x0, x1 + 1))
                    + list(np.ones(y1 - y0 + 1).astype(int) * (x1))
                    + list(range(x1, x0 - 1, -1))
                    + list(np.ones(y1 - y0 + 1).astype(int) * x0)
                )
                points_plotted_y = (
                    list(np.ones(x1 - x0 + 1).astype(int) * y0)
                    + list(range(y0, y1 + 1))
                    + list(np.ones(x1 - x0 + 1).astype(int) * (y1))
                    + list(range(y1, y0 - 1, -1))
                )
                self._current_roi_border = list(zip(points_plotted_x, points_plotted_y))
        
        self._cid_press = self._ax.figure.canvas.mpl_connect('button_press_event', on_press)
        self._cid_motion = self._ax.figure.canvas.mpl_connect('motion_notify_event', on_motion)
        self._cid_release = self._ax.figure.canvas.mpl_connect('button_release_event', on_release)

    def _on_draw_freehand_drag(self) -> None:
        """Handle freehand drag drawing button click."""
        self._show_draw_freehand_drag()
        self._current_roi_coords = []
        self._current_roi_border = []
        self._drawing = False
        self._current_drawing_mode = 'freehand'

        def on_press(event):
            if event.inaxes == self._ax:
                self._drawing = True
                self._current_roi_coords = [(event.xdata, event.ydata)]

        def on_motion(event):
            if self._drawing and event.inaxes == self._ax:
                self._current_roi_coords.append((event.xdata, event.ydata))
                # Draw the path as user draws
                self._ax.clear()
                self._plot_im_on_ax(self._ax)

                # Draw the current path
                if len(self._current_roi_coords) > 1:
                    x, y = zip(*self._current_roi_coords)
                    self._current_roi_border = list(zip(x, y))
                    self._ax.plot(x, y, color='cyan')

                self._matplotlib_canvas.draw()

        def on_release(event):
            self._drawing = False
            self._current_roi_coords.append(self._current_roi_coords[0])  # Close the path
            self._ax.clear()
            self._plot_im_on_ax(self._ax)
            x, y = zip(*self._current_roi_coords)
            self._current_roi_coords = list(zip(x, y))
            plot = self._ax.plot(x, y, color='cyan')
            line = plot[0]
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            self._current_roi_border = list(zip(xdata, ydata))
            self._matplotlib_canvas.draw()

        self._cid_press = self._ax.figure.canvas.mpl_connect('button_press_event', on_press)
        self._cid_motion = self._ax.figure.canvas.mpl_connect('motion_notify_event', on_motion)
        self._cid_release = self._ax.figure.canvas.mpl_connect('button_release_event', on_release)
    
    def _select_dest_folder(self) -> None:
        """
        Helper method for folder selection dialogs.

        Args:
            path_input: QLineEdit widget to update with selected folder path
        """
        # Check if folder path is manually typed and exists
        if os.path.isdir(self._ui.save_folder_input.text()):
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder"
        )

        if folder:
            self._ui.save_folder_input.setText(folder)

    def _hide_save_menu(self) -> None:
        """Hide the save menu."""
        for obj_name in self._save_seg_menu_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.hide()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")
    
    def _show_save_menu(self) -> None:
        """Show the save menu."""
        self._hide_draw_freehand_drag()
        self._hide_draw_rect_drag()
        self._hide_draw_pts()
        try:
            self._ax.figure.canvas.mpl_disconnect(self._cid_press)
        except Exception:
            pass
        try:
            self._ax.figure.canvas.mpl_disconnect(self._cid_motion)
        except Exception:
            pass
        try:
            self._ax.figure.canvas.mpl_disconnect(self._cid_release)
        except Exception:
            pass
        try:
            self._ax.figure.canvas.mpl_disconnect(self._cid_key)
        except Exception:
            pass
        try:
            self._ax.figure.canvas.mpl_disconnect(self._cid_key)
        except Exception:
            pass


        for obj_name in self._save_seg_menu_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.show()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _hide_draw_type_selection(self) -> None:
        """Hide the draw type selection layout."""
        for obj_name in self._draw_types_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.hide()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _show_draw_type_selection(self) -> None:
        """Show the draw type selection layout."""
        # Remove the current ROI
        self._current_roi_coords = []
        self._current_drawing_mode = None
        self._update_drawing_status()
        self._ax.clear()
        self._plot_im_on_ax(self._ax)
        try:
            self._ax.figure.canvas.mpl_disconnect(self._cid_press)
        except Exception:
            pass
        try:
            self._ax.figure.canvas.mpl_disconnect(self._cid_motion)
        except Exception:
            pass
        try:
            self._ax.figure.canvas.mpl_disconnect(self._cid_release)
        except Exception:
            pass
        try:
            while True:
                self._ui.undo_last_pt_button.clicked.disconnect()
        except Exception:
            pass
        try:
            while True:
                self._ui.clear_roi_button.clicked.disconnect()
        except Exception:
            pass
        try:
            while True:
                self._ui.close_roi_button.clicked.disconnect()
        except Exception:
            pass
        self._matplotlib_canvas.draw()

        self._hide_save_menu()
        self._hide_draw_freehand_drag()
        self._hide_draw_rect_drag()
        self._hide_draw_pts()

        for obj_name in self._draw_types_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.show()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _hide_draw_freehand_drag(self) -> None:
        """Hide the freehand drag drawing layout."""
        for obj_name in self._draw_freehand_drag_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.hide()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _show_draw_freehand_drag(self) -> None:
        """Show the freehand drag drawing layout."""
        self._hide_draw_type_selection()
        self._hide_draw_pts()
        self._hide_draw_rect_drag()

        for obj_name in self._draw_freehand_drag_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.show()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _hide_draw_rect_drag(self) -> None:
        """Hide the rectangle drag drawing layout."""
        for obj_name in self._draw_rect_drag_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.hide()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _show_draw_rect_drag(self) -> None:
        """Show the rectangle drag drawing layout."""
        self._hide_draw_type_selection()
        self._hide_draw_freehand_drag()
        self._hide_draw_pts()

        for obj_name in self._draw_rect_drag_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.show()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")
        
    def _hide_draw_pts(self) -> None:
        """Hide the point selection drawing layout."""
        for obj_name in self._draw_pts_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.hide()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _show_draw_pts(self) -> None:
        """Show the point selection drawing layout."""
        self._hide_draw_type_selection()
        self._hide_draw_freehand_drag()
        self._hide_draw_rect_drag()

        for obj_name in self._draw_pts_objects:
            widget = getattr(self._ui, obj_name, None)
            if widget:  
                widget.show()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _hide_roi_dims(self) -> None:
        """Hide the ROI dimensions display."""
        for obj_name in self._roi_dims_objs:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.hide()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")

    def _show_roi_dims(self, width: float, height: float) -> None:
        """Show the ROI dimensions display."""
        for obj_name in self._roi_dims_objs:
            widget = getattr(self._ui, obj_name, None)
            if widget:
                widget.show()
            else:
                print(f"Warning: Widget '{obj_name}' not found in UI")
        
        # Update the dimension values
        self._ui.physical_roi_height_val.setText(f"{height:.2f} mm")
        self._ui.physical_roi_width_val.setText(f"{width:.2f} mm")

    def _show_image_dims(self) -> None:
        """Set the dimensions of the image for display.
        """
        if self._image_data.sc_bmode is None:
            im = self._image_data.bmode
            axial_res = self._image_data.axial_res          # mm/px
            lateral_res = self._image_data.lateral_res      # mm/px
        else:
            im = self._image_data.sc_bmode
            axial_res = self._image_data.sc_axial_res       # mm/px
            lateral_res = self._image_data.sc_lateral_res   # mm/px
        
        width_px = im.shape[1] if im.ndim == 2 else im.shape[2]
        height_px = im.shape[0] if im.ndim == 2 else im.shape[1]
        width_cm = lateral_res * width_px / 10
        height_cm = axial_res * height_px / 10

        self._ui.physical_width_val.setText(f"{width_cm:.2f} cm")
        self._ui.physical_depth_val.setText(f"{height_cm:.2f} cm")
        self._ui.pixel_width_val.setText(f"{width_px} px")
        self._ui.pixel_depth_val.setText(f"{height_px} px")
