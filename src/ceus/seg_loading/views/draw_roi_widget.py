"""
Segmentation Preview Widget for Segmentation Loading
"""

import os
from typing import Optional
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy import interpolate
from PIL import Image, ImageDraw
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QFileDialog
from PyQt6.QtCore import pyqtSignal, Qt

from ...mvc.base_view import BaseViewMixin
from ..ui.draw_roi_ui import Ui_constructRoi
from engines.ceus.src.data_objs import UltrasoundImage


class DrawROIWidget(QWidget, BaseViewMixin):
    """
    Widget for previewing and confirming segmentation.
    
    This is the final step in the segmentation loading process where users
    can preview the loaded segmentation and confirm it before proceeding.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    segmentation_saved = pyqtSignal(str)  # emit with saved file path
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, image_data: UltrasoundImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_constructRoi()
        self._image_data = image_data
        self._matplotlib_canvas: Optional[FigureCanvas] = None
        self._frame = 0
        self._all_frames = self._image_data.pixel_data

        # ROI drawing variables
        self._dragging_drawing = False  # Flag to track if dragging drawing is in progress
        self._roi_scattering_active = False  # Flag to track if ROI scattering is active
        self._drawing = False  # Flag to track if drawing is in progress
        self._min_point_distance = 5.0  # Minimum distance between points in pixels
        self._roi_plot_coords = [[], []]  # Cached ROI plot coordinates for blitting
        self._roi_scatter_coords = np.empty((2, 0))  # Cached ROI scatter coordinates for blitting

        # Animation and performance variables
        self._animation: Optional[anim.FuncAnimation] = None
        self._im_artist = None  # The image artist for fast updates
        self._roi_plot_artist = None  # The ROI artist for fast updates
        self._roi_scatter_artist = None  # The ROI scatter artist for fast updates
        self._target_frame = 0  # Target frame for smooth transitions
        self._frame_update_pending = False
        
        self._setup_ui()
        self._connect_signals()
        self._show_draw_type_selection()

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

        # Setup matplotlib canvas for frame preview
        self._setup_matplotlib_canvas()
        
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
        self._ui.draw_freehand_drag_type_button.clicked.connect(self._on_draw_freehand_drag)
        self._ui.draw_rect_drag_type_button.clicked.connect(self._on_draw_rect_drag)
        self._ui.draw_pts_type_button.clicked.connect(self._on_draw_pts)
        self._ui.back_from_drag_button.clicked.connect(self._show_draw_type_selection)
        self._ui.back_from_pts_button.clicked.connect(self._show_draw_type_selection)
        self._ui.save_drag_button.clicked.connect(self._show_save_menu)
        self._ui.save_pts_button.clicked.connect(self._show_save_menu)
        self._ui.choose_save_folder_button.clicked.connect(self._select_dest_folder)
        self._ui.clear_save_folder_button.clicked.connect(self._ui.save_folder_input.clear)
        self._ui.back_from_save_button.clicked.connect(self._show_draw_type_selection)
        self._ui.save_roi_button.clicked.connect(self._on_save_roi)
            
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
            self._im_artist = self._ax.imshow(self._displayed_im, cmap="gray", animated=True, zorder=1)
            self._roi_plot_artist = self._ax.plot([], [], color='cyan', linewidth=1, zorder=9, animated=True)
            self._roi_scatter_artist = self._ax.scatter([], [], color='red', marker='o', s=5, zorder=10, animated=True)

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
            return [self._im_artist, self._roi_plot_artist[0], self._roi_scatter_artist]

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
            return [self._im_artist, self._roi_plot_artist[0], self._roi_scatter_artist]
        
        # Update to target frame
        if self._frame != self._target_frame:
            self._frame = self._target_frame
            self._update_frame_display(self._frame)
        self._update_roi_plot()
        self._update_roi_scatter()

        self._frame_update_pending = False
        return [self._im_artist, self._roi_plot_artist[0], self._roi_scatter_artist]
        
    def _update_roi_plot(self) -> None:
        """Create/update the ROI plot artist for blitting."""
        if self._roi_plot_artist is None:
            return

        self._roi_plot_artist[0].set_data(self._roi_plot_coords[0], self._roi_plot_coords[1])

    def _update_roi_scatter(self) -> None:
        """Create/update the ROI scatter artist for blitting."""
        if self._roi_scatter_artist is None:
            return

        if self._roi_scattering_active:
            self._roi_scatter_artist.set_offsets(np.array(self._roi_scatter_coords).T)
        else:
            self._roi_scatter_artist.set_offsets(np.empty((0, 2)))

    def _on_frame_changed(self, value: int) -> None:
        """Handle frame slider change with optimized performance."""
        self._target_frame = value
        self._frame_update_pending = True
        # Animation will handle the actual update efficiently
            
    def _update_frame_display(self, frame_index: int) -> None:
        """Update the frame display with consistent parameters."""
        if self._im_artist:
            self._displayed_im = self._all_frames[frame_index]
            self._im_artist.set_array(self._displayed_im)
            self._ui.cur_frame_label.setText(str(np.round(frame_index*self._image_data.frame_rate, decimals=2)))

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
            
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()

    ### ROI Drawing Methods ###

    def _calculate_spline(self, xpts, ypts):
        """Calculate spline interpolation between points (smooth curves)."""
        if len(xpts) < 2:
            return xpts, ypts
            
        # Convert to numpy arrays
        xpts = np.array(xpts)
        ypts = np.array(ypts)
        
        # Remove duplicate points to prevent overshooting
        mask = np.concatenate(([True], (np.diff(xpts) != 0) | (np.diff(ypts) != 0)))
        xpts = xpts[mask]
        ypts = ypts[mask]
        
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
        
        return x_smooth, y_smooth
    
    def _update_drawing_status(self) -> None:
        """Update the drawing status to show current settings."""
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

    def _on_draw_pts(self) -> None:
        """Handle freehand points drawing button click."""
        self._show_draw_pts()
        self._roi_scatter_coords = np.empty((2, 0)); self._roi_plot_coords = [(), ()]
        self._drawing = True
        self._update_drawing_status()

        def draw_cur_roi():
            if len(self._roi_scatter_coords[0]):
                x = self._roi_scatter_coords[0]; y = self._roi_scatter_coords[1]
                self._roi_scattering_active = True
                if len(x) > 1:
                    x_interp, y_interp = self._calculate_spline(x, y)
                    self._roi_plot_coords = [x_interp, y_interp]
                else:
                    self._roi_plot_coords = [(), ()]
            else:
                self._roi_plot_coords = [(), ()]
            self._frame_update_pending = True # Trigger redraw
            
        def close_roi():
            if self._drawing and len(self._roi_scatter_coords[0]) > 1:
                self._roi_scatter_coords[0].append(self._roi_scatter_coords[0][0])  # Close the path
                self._roi_scatter_coords[1].append(self._roi_scatter_coords[1][0])

                x_interp, y_interp = self._calculate_spline(self._roi_scatter_coords[0], self._roi_scatter_coords[1])
                self._roi_plot_coords = [x_interp, y_interp]
                
                self._drawing = False
                self._roi_scattering_active = False
                self._frame_update_pending = True # Trigger redraw
                
        def clear_roi():
            self._roi_scatter_coords = np.empty((2, 0)); self._roi_plot_coords = [(), ()]
            self._drawing = True
            draw_cur_roi()

        def undo_last_pt():
            if self._drawing and len(self._roi_scatter_coords[0]):
                self._roi_scatter_coords[0].pop()
                self._roi_scatter_coords[1].pop()
            if self._drawing:
                draw_cur_roi()

        def check_point_distance(new_x: float, new_y: float) -> bool:
            """
            Check if a new point is far enough from the last point to prevent overshooting.
            
            Args:
                new_x: X coordinate of the new point
                new_y: Y coordinate of the new point
                
            Returns:
                True if the point should be added, False if it's too close
            """
            if len(self._roi_scatter_coords[0]) == 0:
                return True

            last_x = self._roi_scatter_coords[0][-1]
            last_y = self._roi_scatter_coords[1][-1]
            distance = np.sqrt((new_x - last_x)**2 + (new_y - last_y)**2)
            return distance >= self._min_point_distance

        def on_press(event):
            if self._drawing and event.inaxes == self._ax:
                # Check for duplicate points to prevent overshooting
                if not check_point_distance(event.xdata, event.ydata):
                    return

                if not len(self._roi_scatter_coords[0]):
                    self._roi_scatter_coords = ([], [])
                self._roi_scatter_coords[0].append(event.xdata)
                self._roi_scatter_coords[1].append(event.ydata)
                draw_cur_roi()

        def on_motion(event):
            """Handle mouse motion to show distance feedback."""
            if self._drawing and event.inaxes == self._ax and len(self._roi_scatter_coords[0]):
                # Check if mouse is too close to last point
                if not check_point_distance(event.xdata, event.ydata):
                    # Change cursor or add visual feedback
                    self._ax.figure.canvas.setCursor(Qt.CursorShape.ForbiddenCursor)
                else:
                    self._ax.figure.canvas.setCursor(Qt.CursorShape.CrossCursor)

        self._ui.undo_last_pt_button.clicked.connect(undo_last_pt)
        self._ui.clear_roi_button.clicked.connect(clear_roi)
        self._ui.close_roi_button.clicked.connect(close_roi)
        self._cid_press = self._ax.figure.canvas.mpl_connect('button_press_event', on_press)
        self._cid_motion = self._ax.figure.canvas.mpl_connect('motion_notify_event', on_motion)

    def _on_draw_rect_drag(self) -> None:
        """Handle rectangle drag drawing button click."""
        self._show_draw_rect_drag()
        self._roi_scatter_coords = np.empty((2, 0)); self._roi_plot_coords = [(), ()]
        self._drawing = False

        def on_press(event):
            if event.inaxes == self._ax:
                self._drawing = True
                self._roi_scatter_coords = [[event.xdata], [event.ydata]]

        def on_motion(event):
            if self._drawing and event.inaxes == self._ax:
                if len(self._roi_scatter_coords[0]) == 1:
                    self._roi_scatter_coords[0].append(event.xdata)
                    self._roi_scatter_coords[1].append(event.ydata)
                elif len(self._roi_scatter_coords[0]) == 0:
                    self._roi_scatter_coords = [[event.xdata], [event.ydata]]
                elif len(self._roi_scatter_coords[0]) == 2:
                    self._roi_scatter_coords[0][1] = event.xdata
                    self._roi_scatter_coords[1][1] = event.ydata
                else:
                    raise ValueError("Unexpected number of ROI scatter coordinates.")
                
                # Draw the rectangle as user drags
                if len(self._roi_scatter_coords[0]):
                    x0 = self._roi_scatter_coords[0][0]
                    y0 = self._roi_scatter_coords[1][0]
                    x1 = event.xdata
                    y1 = event.ydata

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
                    self._roi_plot_coords = [points_plotted_x, points_plotted_y]

                self._frame_update_pending = True # Trigger redraw

        def on_release(event):
            self._drawing = False
            if len(self._roi_plot_coords[0]) == 2:
                x0, y0 = self._roi_plot_coords[0]
                x1, y1 = self._roi_plot_coords[1]

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
                self._roi_plot_coords = [points_plotted_x, points_plotted_y]
                self._frame_update_pending = True # Trigger redraw
        
        self._cid_press = self._ax.figure.canvas.mpl_connect('button_press_event', on_press)
        self._cid_motion = self._ax.figure.canvas.mpl_connect('motion_notify_event', on_motion)
        self._cid_release = self._ax.figure.canvas.mpl_connect('button_release_event', on_release)

    def _on_draw_freehand_drag(self) -> None:
        """Handle freehand drag drawing button click."""
        self._show_draw_freehand_drag()
        self._roi_plot_coords = [[], []]; self._roi_scatter_coords = np.empty((2, 0))
        self._drawing = False

        def on_press(event):
            if event.inaxes == self._ax:
                self._drawing = True
                self._roi_scatter_coords = [[event.xdata], [event.ydata]]

        def on_motion(event):
            if self._drawing and event.inaxes == self._ax:
                self._roi_scatter_coords[0].append(event.xdata)
                self._roi_scatter_coords[1].append(event.ydata)

                # Draw the current path
                if len(self._roi_scatter_coords[0]) > 1:
                    x, y = self._roi_scatter_coords
                    self._roi_plot_coords = [x, y]

                self._frame_update_pending = True # Trigger redraw

        def on_release(event):
            self._drawing = False
            self._roi_scatter_coords[0].append(self._roi_scatter_coords[0][0])
            self._roi_scatter_coords[1].append(self._roi_scatter_coords[1][0])

            x, y = self._roi_scatter_coords
            x, y = self._calculate_spline(x, y)
            self._roi_plot_coords = [x, y]
            self._frame_update_pending = True # Trigger redraw

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
        self._roi_plot_coords = [[], []]; self._roi_scatter_coords = np.empty((2, 0))
        self._update_drawing_status()

        self._roi_scattering_active = False; self._drawing = False
        self._frame_update_pending = True # Trigger redraw
        
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

    def _on_save_roi(self) -> None:
        """Handle saving the drawn ROI as a CeusSeg object."""
        # Ensure a valid save folder is selected
        dest_folder = self._ui.save_folder_input.text().strip()
        if not os.path.isdir(dest_folder):
            self.show_error("Please select a valid destination folder to save the segmentation.")
            return

        # Ensure a valid ROI name is provided
        roi_name = self._ui.save_name_input.text().strip()
        if not roi_name:
            self.show_error("Please enter a valid name for the segmentation.")
            return

        # Ensure there is a drawn ROI to save
        if len(self._roi_plot_coords[0]) < 3:
            self.show_error("Please draw a valid region of interest before saving.")
            return

        # Create binary mask from drawn ROI
        spline = [(self._roi_plot_coords[0][i], self._roi_plot_coords[1][i]) for i in range(len(self._roi_plot_coords[0]))]
        mask = Image.new("L", (self._all_frames[self._frame].shape[1], self._all_frames[self._frame].shape[0]), 0)
        ImageDraw.Draw(mask).polygon(spline, outline=1, fill=1)
        mask = np.array(mask, dtype=np.uint8)

        # Save mask as NIfTI file
        nii_path = os.path.join(dest_folder, f"{roi_name}.nii.gz")
        nii_img = nib.Nifti1Image(mask, affine=np.eye(4))
        nib.save(nii_img, nii_path)

        self.segmentation_saved.emit(nii_path)
        print(f"Segmentation saved to: {nii_path}")
