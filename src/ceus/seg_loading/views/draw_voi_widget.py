"""
Segmentation File Selection Widget for Segmentation Loading
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes, binary_erosion
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.path import Path as Mpl_Path
import scipy.interpolate as interpolate
from scipy.spatial import ConvexHull
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QSizePolicy, QFileDialog
from PyQt6.QtCore import QEvent, pyqtSignal, Qt, QThread

from ...mvc.base_view import BaseViewMixin
from ..ui.draw_voi_ui import Ui_voi_drawer
from engines.ceus.src.data_objs import UltrasoundImage
from .spline import calculateSpline3D, calculateSpline

def _smooth_3d_mask(mask: np.ndarray) -> np.ndarray:
    """Apply 3D smoothing to the binary mask."""
    mask = binary_fill_holes(mask)
    for i in range(mask.shape[2]):
        border = np.where(mask[:, :, i] == 1)
        if (
            (not len(border[0]))
            or (max(border[0]) == min(border[0]))
            or (max(border[1]) == min(border[1]))
        ):
            continue
        border = np.array(border).T
        hull = ConvexHull(border)
        vertices = border[hull.vertices]
        shape = vertices.shape
        vertices = np.reshape(
            np.append(vertices, vertices[0]), (shape[0] + 1, shape[1])
        )

        # Linear interpolation of 2d convex hull
        tck, _ = interpolate.splprep(vertices.T, s=0.0, k=1)
        splineX, splineY = np.array(
            interpolate.splev(np.linspace(0, 1, 1000), tck)
        )

        mask[:, :, i] = np.zeros((mask.shape[0], mask.shape[1]))
        for j in range(len(splineX)):
            mask[int(splineX[j]), int(splineY[j]), i] = 1
        mask[:, :, i] = binary_fill_holes(mask[:, :, i])

    return mask

class VoiInterpolationWorker(QThread):
    """Worker thread for time-consuming VOI interpolation operations."""
    finished = pyqtSignal(np.ndarray)
    error_msg = pyqtSignal(str)

    def __init__(self, coords: np.ndarray, x_len: int, y_len: int, z_len: int):
        super().__init__()
        self.coords = coords
        self.x_len = x_len; self.y_len = y_len; self.z_len = z_len

    def run(self):
        """Execute the VOI interpolation in background thread."""
        try:
            interp_pts = calculateSpline3D(self.coords)

            # Create the 3D mask from the interpolated surface
            voi_mask = np.zeros((self.x_len, self.y_len, self.z_len), dtype=bool)

            # For simplicity, we'll mark the voxels the spline passes through.
            # A more robust solution would involve filling the volume enclosed by the spline surface.
            interp_points = np.round(np.array(list(interp_pts))).astype(int)

            # Clamp points to be within bounds
            interp_points[:, 0] = np.clip(interp_points[:, 0], 0, self.x_len - 1)
            interp_points[:, 1] = np.clip(interp_points[:, 1], 0, self.y_len - 1)
            interp_points[:, 2] = np.clip(interp_points[:, 2], 0, self.z_len - 1)

            voi_mask[interp_points[:, 0], interp_points[:, 1], interp_points[:, 2]] = True
            
            # Fill holes in the resulting mask to create a solid volume
            voi_mask = _smooth_3d_mask(voi_mask)
            
            self.finished.emit(voi_mask)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_msg.emit(f"Error interpolating VOI: {e}")

class SaveVoiWorker(QThread):
    """Worker thread for saving VOI mask to file in the background."""
    finished = pyqtSignal(str)
    error_msg = pyqtSignal(str)

    def __init__(self, parent_widget):
        super().__init__()
        self.parent_widget = parent_widget

    def run(self):
        try:
            self.parent_widget._save_voi()
            self.finished.emit("VOI saved successfully.")
        except Exception as e:
            self.error_msg.emit(str(e))


class DrawVOIWidget(QWidget, BaseViewMixin):
    """
    Widget for drawing volume of interest (VOI). VOI is drawn and then saved externally before proceeding.
    Can be thought of as file selection for the newly generated VOI.

    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    file_selected = pyqtSignal(dict)  # {'seg_path': str, 'seg_type': str}
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, image_data: UltrasoundImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_voi_drawer()
        self._image_data = image_data
        self._pix_data = image_data.pixel_data

        # State collections
        self._drawing_widgets = []
        self._voi_decision_widgets = []
        self._save_voi_widgets = []
        self._voi_alpha_widgets = []

        # Crosshair / navigation state
        self._crosshair_active = False
        self._crosshair_visible = True
        self._crosshair_xyzt = [0, 0, 0, 0]  # x,y,z,t indices

        # Dimension cache
        self._x_len, self._y_len, self._z_len, self._num_slices = self._pix_data.shape
        self._crosshair_xyzt = [self._x_len // 2, self._y_len // 2, self._z_len // 2, 0]
        
        # Segmentation drawing state
        self._plotted_pts = []
        self._drawing_mode_on = False
        self._current_drawing_plane = None
        self._drawn_rois: List[Tuple[int, List[float], np.ndarray]] = []  # (plane_index, [roi_coords_xyz], roi_mask)
        self._roi_masks_overlap = np.zeros((self._x_len, self._y_len, self._z_len, 4), dtype=np.uint8)

        # Per-plane resources (axial, sagittal, coronal)
        self._ax_sag_cor_matplotlib_canvases = [None, None, None]
        self._ax_sag_cor_planes = (None, None, None)
        self._ax_sag_cor_index_maps = ((0, 1), (2, 1), (2, 0))  # dims that vary per plane
        self._ax_sag_cor_animations = [None, None, None]
        self._ax_sag_cor_plane_artists = [None, None, None]
        self._ax_sag_cor_crosshair_lines = [(None, None), (None, None), (None, None)]
        self._ax_sag_cor_pending = [False, False, False]
        self._ax_sag_cor_roi_plots = [None, None, None]       # dynamic ROI plots
        self._ax_sag_cor_seg_masks = [None, None, None]       # segmentation masks
        self._ax_sag_cor_point_scatters = [None, None, None]  # dynamic point scatters

        self._voi_interpolation_worker: Optional[VoiInterpolationWorker] = None

        # UI & visualization setup sequence
        self._setup_ui()
        self._setup_matplotlib_canvases()
        self._initialize_plane_displays()
        self._setup_all_plane_animations()
        self._connect_signals()
        self._connect_matplotlib_events()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ======================= Matplotlib Mouse Interaction ===================
    def _connect_matplotlib_events(self):
        """Connect motion and click events on each plane's matplotlib canvas.
        Replaces any prior MouseTracker helper by using native mpl events.
        """
        for plane_ix, canvas in enumerate(self._ax_sag_cor_matplotlib_canvases):
            if not canvas:
                continue
            # Use partial-like lambdas capturing plane_ix
            canvas.mpl_connect('motion_notify_event', lambda e, p=plane_ix: self._on_canvas_motion(e, p))
            canvas.mpl_connect('button_press_event', lambda e, p=plane_ix: self._on_canvas_click(e, p))

    def _on_canvas_click(self, event, plane_ix: int):  # type: ignore
        """Handle mouse button press to (re)activate crosshair updates."""
        if event.inaxes is None:
            return
        if not self._drawing_mode_on:
            # Toggle active state even when clicking inside the image frame
            self._crosshair_active = not self._crosshair_active
            if self._crosshair_active:
                self._ui.navigating_label.show()
                self._ui.observing_label.hide()
            else:
                self._ui.navigating_label.hide()
                self._ui.observing_label.show()
        else:
            # Drawing mode: record a point at current crosshair and force plane refresh
            if self._current_drawing_plane is None:
                self._current_drawing_plane = plane_ix + 1
                self._ui.undo_last_roi_button.hide()
                self._ui.close_roi_button.show()
            if self._current_drawing_plane == plane_ix + 1:
                self._crosshair_active = True
                self._on_canvas_motion(event, plane_ix) # refresh crosshair coords before plotting
                self._plotted_pts.append(self._crosshair_xyzt[:])
                self._ax_sag_cor_pending[plane_ix] = True
        self._on_canvas_motion(event, plane_ix)

    def _on_canvas_motion(self, event, plane_ix: int):  # type: ignore
        """Handle mouse movement over a plane and update crosshair indices.

        event.xdata maps to the first varying dimension of that plane slice,
        event.ydata to the second. We clamp to valid ranges and call set_crosshair
        only if the index meaningfully changed.
        """
        if not self._crosshair_active:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        vary_dims = self._ax_sag_cor_index_maps[plane_ix]
        dim_x, dim_y = vary_dims[0], vary_dims[1]

        # Dimension lengths mapping
        dim_lengths = [self._x_len, self._y_len, self._z_len, self._num_slices]

        # Proposed new indices (int rounding & clamp)
        new_xval = int(round(event.xdata))
        new_yval = int(round(event.ydata))
        if new_xval < 0 or new_yval < 0:
            return
        if new_xval >= dim_lengths[dim_x] or new_yval >= dim_lengths[dim_y]:
            return

        # Build kwargs for set_crosshair only for dims that change
        params = {}
        if self._current_drawing_plane == None or self._current_drawing_plane == plane_ix+1:
            if self._crosshair_xyzt[dim_x] != new_xval:
                if dim_x == 0: params['x'] = new_xval
                elif dim_x == 1: params['y'] = new_xval
                elif dim_x == 2: params['z'] = new_xval
                elif dim_x == 3: params['t'] = new_xval
            if self._crosshair_xyzt[dim_y] != new_yval:
                if dim_y == 0: params['x'] = new_yval
                elif dim_y == 1: params['y'] = new_yval
                elif dim_y == 2: params['z'] = new_yval
                elif dim_y == 3: params['t'] = new_yval

            if params:
                self.set_crosshair(**params)
        
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)

        # Store QLabels to show images in each plane
        self._ax_sag_cor_planes = (self._ui.ax_plane, self._ui.sag_plane, self._ui.cor_plane)

        # Configure layout for file selection only
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors for file selection
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.voi_layout, 10)
        
        # Store widgets that should be displayed during different states
        self._drawing_widgets = [
            self._ui.draw_roi_button,
            self._ui.interpolate_voi_button,
            self._ui.undo_last_pt_button,
            self._ui.close_roi_button,
            self._ui.undo_last_roi_button,
            self._ui.construct_voi_label,
        ]
        self._voi_decision_widgets = [
            self._ui.restart_voi_button,
            self._ui.save_voi_button,
        ]
        self._save_voi_widgets = [
            self._ui.back_from_save_button,
            self._ui.dest_folder_label,
            self._ui.voi_name_label,
            self._ui.save_folder_input,
            self._ui.save_name_input,
            self._ui.choose_save_folder_button,
            self._ui.clear_save_folder_button,
            self._ui.export_voi_button,
        ]
        self._voi_alpha_widgets = [
            self._ui.alpha_label,
            self._ui.alpha_of_label,
            self._ui.alpha_spin_box,
            self._ui.alpha_status,
            self._ui.alpha_total
        ]

        self._ui.scan_name_input.setText(self._image_data.scan_name)
        self._ui.toggle_crosshair_visibility_button.setText('Hide Crosshair')

        self._ui.interp_loading_label.hide(); self._ui.saving_voi_label.hide()
        self._ui.navigating_label.hide(); self._ui.undo_last_roi_button.hide()
        self._hide_widget_lists([self._voi_decision_widgets, 
                                 self._save_voi_widgets, self._voi_alpha_widgets])

    def _setup_matplotlib_canvases(self):
        """Setup matplotlib canvases for high-performance plane display."""
        for i in range(3):
            fig = plt.figure()
            fig.patch.set_facecolor((0, 0, 0, 0))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            canvas = FigureCanvas(fig)
            canvas.figure.patch.set_facecolor((0, 0, 0, 0))
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._ax_sag_cor_matplotlib_canvases[i] = canvas
            layout = QHBoxLayout(self._ax_sag_cor_planes[i])
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(canvas, stretch=1)
            self._ax_sag_cor_planes[i].setLayout(layout)
            # Make canvas expand to fill its QLabel container
            # Install event filter on parent label for resize handling
            self._ax_sag_cor_planes[i].installEventFilter(self)
        # Initial sizing pass
        self._resize_all_canvases()

    def _initialize_plane_displays(self) -> None:
        """Initialize all 2D plane displays with optimized matplotlib setup."""
        for plane_ix, canvas in enumerate(self._ax_sag_cor_matplotlib_canvases):
            if not canvas:
                continue
            try:
                fig = canvas.figure
                if plane_ix == 0:  # Axial: y vs x
                    aspect = (self._image_data.pixdim[0]) / (self._image_data.pixdim[1]) if self._image_data.pixdim[0] != 0 else 1
                elif plane_ix == 1:  # Sagittal: y vs z
                    aspect = (self._image_data.pixdim[2]) / (self._image_data.pixdim[1]) if self._image_data.pixdim[2] != 0 else 1
                elif plane_ix == 2:  # Coronal: x vs z
                    aspect = (self._image_data.pixdim[2]) / (self._image_data.pixdim[0]) if self._image_data.pixdim[2] != 0 else 1
                else:
                    self.show_error(f"Invalid plane index: {plane_ix}")
                
                fig.clear()
                ax = fig.add_subplot(111)
                ax.axis('off')
                # Get initial slice
                slice_arr = self._get_plane_slice(plane_ix)
                mask_arr = self._get_mask_slice(plane_ix)

                artist = ax.imshow(slice_arr, cmap='gray', aspect=float(aspect), zorder=1, animated=True)
                v_line = ax.axvline(x=0, color='yellow', lw=0.8, animated=True, zorder=11)
                h_line = ax.axhline(y=0, color='yellow', lw=0.8, animated=True, zorder=11)
                seg_mask = ax.imshow(mask_arr, zorder=8, aspect=float(aspect), animated=True)
                roi_plot = ax.plot([], [], c='cyan', lw=1, zorder=9, animated=True)
                point_scatter = ax.scatter([], [], c='red', s=5, marker='o', zorder=10, animated=True)
                
                self._ax_sag_cor_plane_artists[plane_ix] = artist
                self._ax_sag_cor_crosshair_lines[plane_ix] = (v_line, h_line)
                self._ax_sag_cor_point_scatters[plane_ix] = point_scatter
                self._ax_sag_cor_roi_plots[plane_ix] = roi_plot
                self._ax_sag_cor_seg_masks[plane_ix] = seg_mask

                canvas.draw()
                self._update_crosshair_lines(plane_ix)  # position correctly
            except Exception as e:
                self.show_error(f"Error initializing plane display {plane_ix}: {e}")

    def _get_plane_slice(self, plane_ix: int):
        """Return 2D numpy slice for given plane index based on current crosshair."""
        idx = self._get_plane_indices(plane_ix)
        arr = self._pix_data[idx]
        if arr.ndim != 2:
            arr = arr.squeeze()
        # Axial plane (index 0) needs transpose for correct orientation
        if plane_ix == 0:
            arr = arr.T
        return arr

    def _get_mask_slice(self, plane_ix: int):
        """Return RGBA numpy slice for the mask of the given plane index."""
        idx = self._get_plane_indices(plane_ix)[:-1] # no time dimension
        arr = self._roi_masks_overlap[idx]
        # Mask needs transpose for correct orientation to match the image slice
        if plane_ix == 0:
            arr = np.transpose(arr, (1, 0, 2))  # Transpose for axial plane
        return arr

    def _get_plane_indices(self, plane_ix: int) -> Tuple[int]:
        """Return a list of indices for the given plane."""
        idx = self._crosshair_xyzt[:]
        for d in self._ax_sag_cor_index_maps[plane_ix]:
            idx[d] = slice(None)
        return tuple(idx)

    def _setup_plane_animation(self, plane_ix: int) -> None:
        """Setup FuncAnimation for a specific plane."""
        if self._ax_sag_cor_animations[plane_ix]:
            try:
                self._ax_sag_cor_animations[plane_ix].event_source.stop()
            except Exception:
                pass

        canvas = self._ax_sag_cor_matplotlib_canvases[plane_ix]
        if not canvas:
            return

        def _update(_frame):
            if not self._ax_sag_cor_plane_artists[plane_ix]:
                return []
            # Always refresh slice when pending
            if self._ax_sag_cor_pending[plane_ix]:
                try:
                    slice_arr = self._get_plane_slice(plane_ix)
                    self._ax_sag_cor_plane_artists[plane_ix].set_array(slice_arr)
                    self._update_crosshair_lines(plane_ix)
                except Exception as e:
                    self.show_error(f"Plane {plane_ix} update error: {e}")
                finally:
                    self._ax_sag_cor_pending[plane_ix] = False
            # Update point scatter every frame (cheap; typically few points)
            self._update_roi_plot(plane_ix)
            self._update_point_scatter(plane_ix)
            self._update_seg_masks(plane_ix)
            
            v_line, h_line = self._ax_sag_cor_crosshair_lines[plane_ix]
            roi_plot = self._ax_sag_cor_roi_plots[plane_ix]
            scatter = self._ax_sag_cor_point_scatters[plane_ix]
            mask = self._ax_sag_cor_seg_masks[plane_ix]
            artists = [self._ax_sag_cor_plane_artists[plane_ix]]
            if v_line: artists.append(v_line)
            if h_line: artists.append(h_line)
            if roi_plot: artists.append(roi_plot[0])
            if scatter: artists.append(scatter)
            if mask: artists.append(mask)

            # Only update frame counters occasionally or when pending refreshed
            if self._ax_sag_cor_pending[plane_ix]:
                self._update_scan_display()
            return artists

        self._ax_sag_cor_animations[plane_ix] = anim.FuncAnimation(
            canvas.figure,
            _update,
            interval=33,  # ~30 FPS
            blit=True,
            repeat=False,
            cache_frame_data=False
        )

    def _setup_all_plane_animations(self):
        for i in range(3):
            self._setup_plane_animation(i)

    def _update_crosshair_lines(self, plane_ix: int):
        """Update crosshair line positions for given plane."""
        v_line, h_line = self._ax_sag_cor_crosshair_lines[plane_ix]
        if not (v_line and h_line):
            return
        vary_dims = self._ax_sag_cor_index_maps[plane_ix]
        x_dim, y_dim = vary_dims[0], vary_dims[1]
        x_idx = self._crosshair_xyzt[x_dim]
        y_idx = self._crosshair_xyzt[y_dim]
        v_line.set_xdata([x_idx, x_idx])
        h_line.set_ydata([y_idx, y_idx])

        if not self._crosshair_visible:
            v_line.set_visible(False); h_line.set_visible(False)
        else:
            # Ensure visible when expected (avoids lingering hidden state)
            v_line.set_visible(True); h_line.set_visible(True)

    # ------------------------ Public API ------------------------------------
    def set_crosshair(self, x: Optional[int] = None, y: Optional[int] = None,
                      z: Optional[int] = None, t: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Set (and clamp) crosshair indices then flag planes for redraw.

        Parameters are optional; only provided axes are updated. Values are
        clamped into valid bounds. All three orthogonal plane views are marked
        pending so the animation loop refreshes them on the next frame.
        Returns the updated (x,y,z,t) tuple.
        """
        # Current values
        cx, cy, cz, ct = self._crosshair_xyzt
        if x is not None:
            cx = max(0, min(self._x_len - 1, int(x)))
        if y is not None:
            cy = max(0, min(self._y_len - 1, int(y)))
        if z is not None:
            cz = max(0, min(self._z_len - 1, int(z)))
        if t is not None:
            ct = max(0, min(self._num_slices - 1, int(t)))
        # Only proceed if changed
        if [cx, cy, cz, ct] != self._crosshair_xyzt:
            self._crosshair_xyzt = [cx, cy, cz, ct]
            self._refresh_frames()
        return cx, cy, cz, ct
    
    def _update_seg_masks(self, plane_ix):
        """Create/update the segmentation masks for frames on a given plane for blitting."""
        mask_2d = self._get_mask_slice(plane_ix)
        self._ax_sag_cor_seg_masks[plane_ix].set_array(mask_2d)

    def _update_roi_plot(self, plane_ix):
        """Create/update the ROI plot artist for points on a given plane for blitting."""
        # Determine which dimensions vary on this plane (plane coordinate axes)
        if self._current_drawing_plane is None or self._current_drawing_plane != plane_ix + 1:
            return
        vary_x_dim, vary_y_dim = self._ax_sag_cor_index_maps[plane_ix]
        cur_dim = 3 - vary_x_dim - vary_y_dim

        plane_points = [(pt[vary_x_dim], pt[vary_y_dim]) for pt in self._plotted_pts
                        if pt[cur_dim] == self._crosshair_xyzt[cur_dim]]
        
        if not plane_points or len(plane_points) == 1:
            # Hide existing ROI plot if present
            self._ax_sag_cor_roi_plots[plane_ix][0].set_data([], [])
            return
        
        x, y = zip(*plane_points)
        x_interp, y_interp = calculateSpline(x, y)
        self._ax_sag_cor_roi_plots[plane_ix][0].set_data(x_interp, y_interp)

    def _update_point_scatter(self, plane_ix: int):
        """Create/update the scatter artist for points on a given plane for blitting."""
        # Determine which dimensions vary on this plane (plane coordinate axes)
        vary_x_dim, vary_y_dim = self._ax_sag_cor_index_maps[plane_ix]
        cur_dim = 3 - vary_x_dim - vary_y_dim

        plane_points = [(pt[vary_x_dim], pt[vary_y_dim]) for pt in self._plotted_pts
                        if pt[cur_dim] == self._crosshair_xyzt[cur_dim]]

        scatter = self._ax_sag_cor_point_scatters[plane_ix]
        if not plane_points:
            # Hide existing scatter if present
            scatter.set_offsets(np.empty((0, 2)))
            return

        offsets = np.array(plane_points)
        scatter.set_offsets(offsets)

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        self._ui.draw_roi_button.clicked.connect(self._on_draw_roi_clicked)
        self._ui.undo_last_pt_button.clicked.connect(self._on_undo_last_pt)
        self._ui.close_roi_button.clicked.connect(self._on_roi_close)
        self._ui.undo_last_roi_button.clicked.connect(self._on_undo_last_roi)
        self._ui.interpolate_voi_button.clicked.connect(self._on_interpolate_voi)
        self._ui.restart_voi_button.clicked.connect(self._on_restart_voi)
        self._ui.export_voi_button.clicked.connect(self._on_export_voi_clicked)
        self._ui.choose_save_folder_button.clicked.connect(self._on_choose_folder)
        self._ui.clear_save_folder_button.clicked.connect(self._ui.save_folder_input.clear)
        self._ui.back_from_save_button.clicked.connect(self._on_back_from_save)
        self._ui.toggle_crosshair_visibility_button.clicked.connect(self._on_toggle_crosshair_visibility)
        self._ui.save_voi_button.clicked.connect(self._on_save_voi_clicked)
        
        self._ui.cur_slice_slider.setMinimum(0)
        self._ui.cur_slice_slider.setMaximum(max(0, self._num_slices - 1))
        self._ui.cur_slice_slider.setValue(self._crosshair_xyzt[3])
        self._ui.cur_slice_slider.valueChanged.connect(self._on_time_slider_changed)

    def _on_time_slider_changed(self, value: int):  # type: ignore
        """Handle user sliding through time dimension (t)."""
        # Clamp safety (though slider should enforce)
        if value < 0:
            value = 0
        if value >= self._num_slices:
            value = self._num_slices - 1
        prev_t = self._crosshair_xyzt[3]
        if value == prev_t:
            return
        self.set_crosshair(t=value)
        self._refresh_frames()
        # Keep slider in sync if set_crosshair clamped
        if self._ui.cur_slice_slider.value() != self._crosshair_xyzt[3]:
            self._ui.cur_slice_slider.blockSignals(True)
            self._ui.cur_slice_slider.setValue(self._crosshair_xyzt[3])
            self._ui.cur_slice_slider.blockSignals(False)

    def _on_draw_roi_clicked(self):
        """Handle draw ROI button click."""
        self._drawing_mode_on = not self._drawing_mode_on
        if self._drawing_mode_on:
            self._ui.draw_roi_button.setText('Disable Draw')
        else:
            self._ui.draw_roi_button.setText('Draw ROI')

    def _on_undo_last_pt(self):
        """Undo the last drawn point."""
        if self._plotted_pts:
            self._plotted_pts.pop()
            self._refresh_frames()
        if not self._plotted_pts:
            self._current_drawing_plane = None

    def _on_roi_close(self):
        """Handle ROI close event by creating a 2D mask on the correct plane."""
        if len(self._plotted_pts) < 3 or self._current_drawing_plane is None:
            return

        if self._drawing_mode_on:
            self._on_draw_roi_clicked()

        # Local copy of points and close the loop
        current_roi_pts = self._plotted_pts[:]
        current_roi_pts.append(current_roi_pts[0])

        plane_ix = self._current_drawing_plane - 1
        vary_x_dim, vary_y_dim = self._ax_sag_cor_index_maps[plane_ix]
        fixed_dim = 3 - vary_x_dim - vary_y_dim
        fixed_val = self._crosshair_xyzt[fixed_dim]

        # Get 2D points projected onto the current plane
        plane_points_2d_raw = [(p[vary_x_dim], p[vary_y_dim]) for p in current_roi_pts]
        
        # Get interpolated points for a smoother mask
        x_raw, y_raw = zip(*plane_points_2d_raw)
        x_interp, y_interp = calculateSpline(x_raw, y_raw)
        plane_points_2d = np.vstack((x_interp, y_interp)).T

        # Define the grid for the plane
        dims = self._pix_data.shape
        plane_dim_x_len = dims[vary_x_dim]
        plane_dim_y_len = dims[vary_y_dim]
        
        x_grid, y_grid = np.meshgrid(np.arange(plane_dim_x_len), np.arange(plane_dim_y_len))
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

        # Create a 2D mask from the path of the interpolated spline
        path = Mpl_Path(plane_points_2d)
        mask_2d = path.contains_points(grid_points).reshape(plane_dim_y_len, plane_dim_x_len)

        # Create a 4D RGBA mask for this single ROI
        current_roi_mask_rgba = np.zeros((self._x_len, self._y_len, self._z_len, 4), dtype=np.uint8)
        
        # Get a boolean mask for the correct 3D slice
        target_slice_mask = np.zeros((self._x_len, self._y_len, self._z_len), dtype=bool)
        
        # Place the 2D mask into the correct 3D slice, handling orientation
        if plane_ix == 0:  # Axial
            target_slice_mask[:, :, fixed_val] = mask_2d.T
        elif plane_ix == 1:  # Sagittal
            target_slice_mask[fixed_val, :, :] = mask_2d
        elif plane_ix == 2:  # Coronal
            target_slice_mask[:, fixed_val, :] = mask_2d

        # Apply colors to the RGBA mask where the 3D mask is true
        current_roi_mask_rgba[target_slice_mask, 0] = 255  # Red
        current_roi_mask_rgba[target_slice_mask, 3] = 128  # Alpha

        # Store the original points and the generated mask
        self._drawn_rois.append((self._current_drawing_plane, current_roi_pts, current_roi_mask_rgba))

        # Update the master overlap mask by blending all stored ROIs
        self._roi_masks_overlap.fill(0)
        for _, _, roi_mask in self._drawn_rois:
            # Add color channels, clipping at 255
            self._roi_masks_overlap[:,:,:,:3] = np.clip(self._roi_masks_overlap[:,:,:,:3].astype(np.uint16) + roi_mask[:,:,:,:3].astype(np.uint16), 0, 255).astype(np.uint8)
            # Add alpha, clipping at a reasonable max to avoid full opacity
            self._roi_masks_overlap[:,:,:,3] = np.clip(self._roi_masks_overlap[:,:,:,3].astype(np.uint16) + roi_mask[:,:,:,3].astype(np.uint16), 0, 128).astype(np.uint8)

        # Clear points and hide the ROI plot for the next ROI
        self._plotted_pts.clear()
        self._ax_sag_cor_roi_plots[plane_ix][0].set_data([], [])
        self._current_drawing_plane = None
        
        # Update button states
        self._ui.draw_roi_button.setChecked(False)
        self._ui.undo_last_roi_button.show()
        self._ui.close_roi_button.hide()

        self._refresh_frames()

    def _on_undo_last_roi(self):
        """Handle undoing the last completed ROI."""
        if not self._drawn_rois:
            return

        # Remove the last ROI
        self._drawn_rois.pop()

        # Recalculate the overlap mask from the remaining ROIs
        self._roi_masks_overlap.fill(0)
        if self._drawn_rois:
            for _, _, roi_mask in self._drawn_rois:
                # Add color channels, clipping at 255
                self._roi_masks_overlap[:,:,:,:3] = np.clip(self._roi_masks_overlap[:,:,:,:3].astype(np.uint16) + roi_mask[:,:,:,:3].astype(np.uint16), 0, 255).astype(np.uint8)
                # Add alpha, clipping at a reasonable max to avoid full opacity
                self._roi_masks_overlap[:,:,:,3] = np.clip(self._roi_masks_overlap[:,:,:,3].astype(np.uint16) + roi_mask[:,:,:,3].astype(np.uint16), 0, 128).astype(np.uint8)

        # Hide the button if no ROIs are left to undo
        if not self._drawn_rois:
            self._ui.undo_last_roi_button.hide()
            self._ui.close_roi_button.show()

        self._refresh_frames()

    def _on_toggle_crosshair_visibility(self):
        # Toggle visibility state but keep indices updating
        self._crosshair_visible = not self._crosshair_visible
        self._refresh_frames()
        self._ui.toggle_crosshair_visibility_button.setText(
            'Show Crosshair' if not self._crosshair_visible else 'Hide Crosshair'
        )

    def _on_restart_voi(self):
        """Handle restarting the VOI creation process."""
        # Reset the drawing state
        self._drawn_rois.clear()
        self._roi_masks_overlap.fill(0)
        self._plotted_pts.clear()
        self._current_drawing_plane = None
        
        # Update UI
        self._hide_widget_lists([self._voi_decision_widgets])
        self._show_widget_lists([self._drawing_widgets])
        self._ui.undo_last_roi_button.hide()
        self._refresh_frames()

    def _on_save_voi_clicked(self):
        self._hide_widget_lists([self._voi_decision_widgets])
        self._show_widget_lists([self._save_voi_widgets, self._voi_alpha_widgets])
        self._refresh_frames()

    def _on_export_voi_clicked(self):
        # Show saving label, hide save widgets
        self._ui.saving_voi_label.show()
        self._hide_widget_lists([self._save_voi_widgets])
        
        self._save_worker = SaveVoiWorker(self)
        self._save_worker.finished.connect(self._on_save_voi_finished)
        self._save_worker.error_msg.connect(self._on_save_voi_error)
        self._save_worker.start()

    def _on_save_voi_finished(self, msg):
        self._ui.saving_voi_label.hide()
        self._show_widget_lists([self._save_voi_widgets])
        print(msg)

    def _on_save_voi_error(self, err):
        self._ui.saving_voi_label.hide()
        self._show_widget_lists([self._save_voi_widgets])
        self.show_error(f"Error saving VOI: {err}")

    def _on_back_from_save(self):
        """Handle back button click from save VOI."""
        self._hide_widget_lists([self._save_voi_widgets])
        self._show_widget_lists([self._voi_decision_widgets])
        self._refresh_frames()

    def _on_choose_folder(self):
        """Select folder to save VOI to."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._ui.save_folder_input.setText(folder)

    def _refresh_frames(self) -> None:
        """Refresh the displayed frames."""
        for i in range(3):
            self._ax_sag_cor_pending[i] = True

    def _update_scan_display(self):
        """Update the scan display with the current frames and frame numbers"""
        # Update frame numbers
        self._ui.ax_frame_num.setText(str(self._crosshair_xyzt[2] + 1))
        self._ui.sag_frame_num.setText(str(self._crosshair_xyzt[0] + 1))
        self._ui.cor_frame_num.setText(str(self._crosshair_xyzt[1] + 1))

    def mousePressEvent(self, a0):
        super().mousePressEvent(a0)
        self._crosshair_active = not self._crosshair_active
        if self._crosshair_active:
            self._ui.navigating_label.show(); self._ui.observing_label.hide()
        else:
            self._ui.navigating_label.hide(); self._ui.observing_label.show()

    def keyPressEvent(self, event):  # type: ignore
        """Handle key presses for quick actions (e.g., 'd' to toggle draw ROI)."""
        if event.key() == Qt.Key.Key_D:
            self._on_draw_roi_clicked()
            self._ui.draw_roi_button.setChecked(self._drawing_mode_on)
            event.accept()
            return
        if event.key() == Qt.Key.Key_U:
            self._on_undo_last_pt()
            event.accept()
            return
        if event.key() == Qt.Key.Key_H:
            self._on_toggle_crosshair_visibility()
            return
        if event.key() == Qt.Key.Key_C:
            self._on_roi_close()
            return
        if event.key() == Qt.Key.Key_R:
            self._on_undo_last_roi()
            return
        super().keyPressEvent(event)
        
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.clear_error()
        self.back_requested.emit()

    def _hide_widget_lists(self, widgets: List[List[QWidget]]) -> None:
        """
        Hide all relevant widgets in the lists.
        """
        for widget_list in widgets:
            for widget in widget_list:
                widget.hide()

    def _show_widget_lists(self, widgets: List[List[QWidget]]) -> None:
        """
        Show all relevant widgets in the lists.
        """
        for widget_list in widgets:
            for widget in widget_list:
                widget.show()

    # ======================= Lifecycle / Cleanup ==============================
    def _cleanup_animations(self):
        for i, anim_obj in enumerate(self._ax_sag_cor_animations):
            if anim_obj:
                try:
                    anim_obj.event_source.stop()
                except Exception:
                    pass
                self._ax_sag_cor_animations[i] = None

    def closeEvent(self, event):  # type: ignore
        self._cleanup_animations()
        return super().closeEvent(event)

    def hideEvent(self, event):  # type: ignore
        self._cleanup_animations()
        return super().hideEvent(event)

    def showEvent(self, event):  # type: ignore
        # Recreate animations when shown again
        if not any(self._ax_sag_cor_animations):
            self._setup_all_plane_animations()
        # Ensure canvases sized properly when shown
        self._resize_all_canvases()
        return super().showEvent(event)

    # ======================= Resize Handling =================================
    def eventFilter(self, obj, event):  # type: ignore
        if event.type() == QEvent.Type.Resize and obj in self._ax_sag_cor_planes:
            self._resize_canvas_for(obj)
        return super().eventFilter(obj, event)

    def _resize_canvas_for(self, label_widget: QLabel):
        try:
            idx = self._ax_sag_cor_planes.index(label_widget)
        except ValueError:
            return
        canvas = self._ax_sag_cor_matplotlib_canvases[idx]
        if not canvas:
            return

        canvas.figure.tight_layout(pad=0)
        canvas.draw_idle()

    def _resize_all_canvases(self):
        for lbl in self._ax_sag_cor_planes:
            self._resize_canvas_for(lbl)

    def _remove_duplicates(self, points: List[List[float]]) -> List[List[float]]:
        """Remove duplicate points from a list of points."""
        seen = set()
        unique_points = []
        for p in points:
            p_tuple = tuple(p)
            if p_tuple not in seen:
                unique_points.append(p)
                seen.add(p_tuple)
        return unique_points

    def _on_interpolate_voi(self):
        """Handle VOI interpolation from the drawn 2D ROIs."""
        if len(self._drawn_rois) == 2 or not len(self._drawn_rois):
            print("At least 3 ROIs on different planes or 1 ROI is required for 3D interpolation.")
            return

        # Combine all points from all drawn ROIs
        all_points = []
        for _, pts, _ in self._drawn_rois:
            xyz_pts = np.array(pts)[:, :3].T
            x_interp, y_interp, z_interp = calculateSpline(*xyz_pts)
            all_points.extend(zip(x_interp, y_interp, z_interp))

        # Ensure no duplicate points are used for interpolation
        unique_points = self._remove_duplicates(all_points)
        if len(unique_points) < 4:
            self.show_error("Interpolation Error", "Not enough unique points for 3D spline interpolation.")
            return

        # Perform 3D spline interpolation
        x_coords, y_coords, z_coords = zip(*unique_points)
        coords = np.transpose([x_coords, y_coords, z_coords])
        
        if len(self._drawn_rois) > 2:
            # Stop any existing worker
            if self._voi_interpolation_worker and self._voi_interpolation_worker.isRunning():
                self._voi_interpolation_worker.quit()
                self._voi_interpolation_worker.wait()

            # Create and start worker
            self._voi_interpolation_worker = VoiInterpolationWorker(
                coords, self._x_len, self._y_len, self._z_len
            )

            # Connect worker signals
            self._voi_interpolation_worker.finished.connect(self._on_interpolation_finished)
            self._voi_interpolation_worker.error_msg.connect(self.show_error)

            # Start interpolatoin loading
            self._set_interp_loading(True)
            self._voi_interpolation_worker.start()
        else:
            voi_mask = np.zeros((self._x_len, self._y_len, self._z_len), dtype=bool)

            # For simplicity, we'll mark the voxels the spline passes through.
            # A more robust solution would involve filling the volume enclosed by the spline surface.
            interp_points = np.round(np.array(list(coords))).astype(int)

            # Clamp points to be within bounds
            interp_points[:, 0] = np.clip(interp_points[:, 0], 0, self._x_len - 1)
            interp_points[:, 1] = np.clip(interp_points[:, 1], 0, self._y_len - 1)
            interp_points[:, 2] = np.clip(interp_points[:, 2], 0, self._z_len - 1)

            voi_mask[interp_points[:, 0], interp_points[:, 1], interp_points[:, 2]] = True
            
            # Fill holes in the resulting mask to create a solid volume
            voi_mask = _smooth_3d_mask(voi_mask)
            self._hide_widget_lists([self._drawing_widgets])
            self._on_interpolation_finished(voi_mask)

    def _save_voi(self):
        """Save the current VOI mask to a file."""
        if not Path(self._ui.save_folder_input.text()).is_dir():
            print("Invalid Folder", "Please select a valid folder to save the VOI.")
            return
        
        out_name = self._ui.save_name_input.text()
        if not out_name:
            print("Invalid Name", "Please enter a valid name for the VOI.")
            return
        out_name = out_name + '.nii.gz' if not out_name.endswith('.nii.gz') else out_name

        out_path = Path(self._ui.save_folder_input.text()) / out_name

        affine = np.eye(4)
        for i, res in enumerate(self._image_data.pixdim[:3]):
            affine[i, i] = res
        voi_mask = np.array(self._roi_masks_overlap[:, :, :, 0] / 255.0).astype(np.uint8)
        niiarray = nib.Nifti1Image(voi_mask, affine)
        niiarray.header["descrip"] = self._image_data.scan_name
        nib.save(niiarray, out_path)

    def _set_interp_loading(self, loading_state: bool) -> None:
        """Set the interpolation loading state."""
        if loading_state:
            self._hide_widget_lists([self._drawing_widgets])
            self._ui.interp_loading_label.show()
            self._ui.back_button.setEnabled(False)
        else:
            self._ui.interp_loading_label.hide()
            self._show_widget_lists([self._voi_decision_widgets])
            self._ui.back_button.setEnabled(True)

    def _on_interpolation_finished(self, voi_mask: np.ndarray):
        # Update the master overlap mask with the new 3D VOI
        self._roi_masks_overlap.fill(0)
        self._roi_masks_overlap[voi_mask, 0] = 255  # Red
        self._roi_masks_overlap[voi_mask, 3] = 128  # Alpha
        
        self._set_interp_loading(False)
        self._refresh_frames()
