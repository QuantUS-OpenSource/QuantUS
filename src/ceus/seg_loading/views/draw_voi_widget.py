"""
Segmentation File Selection Widget for Segmentation Loading
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.path import Path as Mpl_Path
from matplotlib.colors import LinearSegmentedColormap
import scipy.interpolate as interpolate
from scipy.spatial import ConvexHull
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QSizePolicy, QFileDialog, QSlider, QVBoxLayout, QFrame, QCheckBox
from PyQt6.QtCore import QEvent, pyqtSignal, Qt, QThread

from ...mvc.base_view import BaseViewMixin
from ..ui.draw_voi_ui import Ui_voi_drawer
from engines.ceus.src.data_objs import UltrasoundImage
from .spline import calculateSpline3D, calculateSpline

# Philips CEUS Colormap: Grayscale -> Red -> Yellow
philips_colors = [
    (0.0, 0.0, 0.0),    # 0% - Black
    (0.4, 0.4, 0.4),    # 40% - Gray
    (0.8, 0.0, 0.0),    # 80% - Red
    (1.0, 1.0, 0.0)     # 100% - Yellow
]
philips_cmap = LinearSegmentedColormap.from_list("philips_ceus", philips_colors)

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
    segmentation_saved = pyqtSignal(str)  # emit with saved file path
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    apply_preprocs_preview = pyqtSignal(list)  # List of dicts with 'name' and 'kwargs' keys

    def __init__(self, image_data: UltrasoundImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_voi_drawer()
        self._image_data = image_data
        self._pix_data = image_data.pixel_data
        
        # Enhancement parameters
        self._clahe_clip_limit = 1.2
        self._gamma = 1.5
        self._width_scale_axial = 1.0
        self._width_scale_sagittal = 1.0
        self._width_scale_coronal = 1.0
        self._use_philips_ceus = False
        
        # Cache for enhanced volume
        self._enhanced_cache = None
        self._enhanced_cache_frame = -1

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
        self._update_scan_display()  # Initial UI update
        self._refresh_frames()       # Mark all planes for first update

    def update_enhancement_cache(self, enhanced_frame: np.ndarray, frame: int) -> None:
        """Update the displayed image data, e.g. after preprocessing."""
        assert enhanced_frame.shape[:-1] == self._pix_data.shape[:-1], "Enhanced pixel data must have the same shape as original"
        self._enhanced_cache = enhanced_frame[:, :, :, 0]  # Store only the current time frame in cache
        self._enhanced_cache_frame = frame
        self._refresh_frames()

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
        self._ui.cur_slice_label.setText("Current Frame:")

        self._ui.interp_loading_label.hide(); self._ui.saving_voi_label.hide()
        self._ui.navigating_label.hide(); self._ui.undo_last_roi_button.hide()
        self._hide_widget_lists([self._voi_decision_widgets, 
                                 self._save_voi_widgets, self._voi_alpha_widgets])
                                 
        # Setup enhancement controls in sidebar
        self._setup_enhancement_controls()

    def _setup_enhancement_controls(self) -> None:
        """Add enhancement sliders to the sidebar, styled like the existing slice slider."""
        # Container frame for enhancement controls
        enh_group = QFrame()
        enh_group.setStyleSheet("background-color: rgba(255, 255, 255, 0); border: none;")
        
        # Main vertical layout to stack rows
        container_layout = QVBoxLayout(enh_group)
        container_layout.setContentsMargins(0, 10, 0, 10)
        container_layout.setSpacing(15)

        # Rows for horizontal grouping
        row1_layout = QHBoxLayout()
        row2_layout = QHBoxLayout()
        row1_layout.setSpacing(20)
        row2_layout.setSpacing(20)

        # Helper to create a styled slider column
        def create_enh_column(label_text, min_val, max_val, current_val, callback):
            col_widget = QWidget()
            col_layout = QVBoxLayout(col_widget)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(5)
            
            # Label
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 14px; color: white; font-weight: bold;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col_layout.addWidget(lbl)
            
            # Slider + Value Row
            row_layout = QHBoxLayout()
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(current_val)
            # Copy style and size constraints from slice slider if possible
            slider.setStyleSheet(self._ui.cur_slice_slider.styleSheet())
            slider.setMinimumWidth(100)
            slider.setMaximumWidth(120)
            slider.valueChanged.connect(callback)
            
            val_lbl = QLabel(f"{current_val/10.0:.1f}")
            val_lbl.setMinimumWidth(40)
            val_lbl.setStyleSheet("color: #3498db; font-weight: bold; font-size: 14px;")
            
            row_layout.addWidget(slider)
            row_layout.addWidget(val_lbl)
            col_layout.addLayout(row_layout)
            
            return col_widget, slider, val_lbl

        # Create the columns
        clahe_col, self.clahe_slider, self.clahe_val_lbl = create_enh_column(
            "CLAHE", 1, 100, int(self._clahe_clip_limit * 10), self._on_clahe_changed
        )
        gamma_col, self.gamma_slider, self.gamma_val_lbl = create_enh_column(
            "GAMMA", 1, 40, int(self._gamma * 10), self._on_gamma_changed
        )
        width_ax_col, self.width_ax_slider, self.width_ax_val_lbl = create_enh_column(
            "WIDTH (AX)", 1, 50, int(self._width_scale_axial * 10), self._on_width_axial_changed
        )
        width_sag_col, self.width_sag_slider, self.width_sag_val_lbl = create_enh_column(
            "WIDTH (SAG)", 1, 50, int(self._width_scale_sagittal * 10), self._on_width_sagittal_changed
        )
        width_cor_col, self.width_cor_slider, self.width_cor_val_lbl = create_enh_column(
            "WIDTH (COR)", 1, 50, int(self._width_scale_coronal * 10), self._on_width_coronal_changed
        )
        
        row1_layout.addWidget(clahe_col)
        row1_layout.addWidget(gamma_col)
        
        # Philips CEUS Toggle (Pseudocoloring) - now in row 1
        self.philips_check = QCheckBox("Pseudocoloring")
        self.philips_check.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        self.philips_check.setToolTip("Philips CEUS Style:\n"
                                     "Black → Low Intensity\n"
                                     "Gray → Tissue Background\n"
                                     "Red → Medium Enhancement\n"
                                     "Yellow → Peak Enhancement")
        self.philips_check.stateChanged.connect(self._on_philips_toggled)
        row1_layout.addWidget(self.philips_check)

        row2_layout.addWidget(width_ax_col)
        row2_layout.addWidget(width_sag_col)
        row2_layout.addWidget(width_cor_col)
        
        container_layout.addLayout(row1_layout)
        container_layout.addLayout(row2_layout)

        # Add to the layout below the current slice slider
        self._ui.verticalLayout_2.addWidget(enh_group)

    def _on_philips_toggled(self, state: int) -> None:
        """Handle Philips CEUS pseudocolor toggle."""
        self._use_philips_ceus = state == Qt.CheckState.Checked.value
        # Update colormap on all artists
        new_cmap = philips_cmap if self._use_philips_ceus else 'gray'
        for artist in self._ax_sag_cor_plane_artists:
            if artist:
                artist.set_cmap(new_cmap)
        self._refresh_frames()

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

    def _on_width_axial_changed(self, value: int) -> None:
        """Handle axial width scale change."""
        self._width_scale_axial = value / 10.0
        if hasattr(self, 'width_ax_val_lbl'):
            self.width_ax_val_lbl.setText(f"{self._width_scale_axial:.1f}")
        self._update_aspect_ratios()
        self._refresh_frames()

    def _on_width_sagittal_changed(self, value: int) -> None:
        """Handle sagittal width scale change."""
        self._width_scale_sagittal = value / 10.0
        if hasattr(self, 'width_sag_val_lbl'):
            self.width_sag_val_lbl.setText(f"{self._width_scale_sagittal:.1f}")
        self._update_aspect_ratios()
        self._refresh_frames()

    def _on_width_coronal_changed(self, value: int) -> None:
        """Handle coronal width scale change."""
        self._width_scale_coronal = value / 10.0
        if hasattr(self, 'width_cor_val_lbl'):
            self.width_cor_val_lbl.setText(f"{self._width_scale_coronal:.1f}")
        self._update_aspect_ratios()
        self._refresh_frames()

    def _update_aspect_ratios(self) -> None:
        """Update the aspect ratios of the axes based on the plane-specific width scales."""
        if not hasattr(self, '_image_data') or self._image_data is None:
            return
            
        try:
            pix = self._image_data.pixdim
            # Index 0: Axial (Plane 0)
            if self._ax_sag_cor_matplotlib_canvases[0]:
                dx, dy = pix[0], pix[1]
                aspect = (dy / dx if dx != 0 else 1.0) * self._width_scale_axial
                fig0 = self._ax_sag_cor_matplotlib_canvases[0].figure
                if fig0.axes:
                    fig0.axes[0].set_aspect(aspect)
            
            # Index 1: Sagittal (Plane 1) 
            if self._ax_sag_cor_matplotlib_canvases[1]:
                dy, dz = pix[1], pix[2]
                aspect = (dy / dz if dz != 0 else 1.0) * self._width_scale_sagittal
                fig1 = self._ax_sag_cor_matplotlib_canvases[1].figure
                if fig1.axes:
                    fig1.axes[0].set_aspect(aspect)
                
            # Index 2: Coronal (Plane 2)
            if self._ax_sag_cor_matplotlib_canvases[2]:
                dx, dz = pix[0], pix[2]
                aspect = (dx / dz if dz != 0 else 1.0) * self._width_scale_coronal
                fig2 = self._ax_sag_cor_matplotlib_canvases[2].figure
                if fig2.axes:
                    fig2.axes[0].set_aspect(aspect)
                
            for canvas in self._ax_sag_cor_matplotlib_canvases:
                if canvas:
                    canvas.draw_idle()
        except Exception as e:
            print(f"Error updating aspect ratios: {e}")
        
    def _reset_enhancement(self, _=None) -> None:
        """Reset enhancement parameters to defaults."""
        self.clahe_slider.setValue(12)  # 1.2
        self.gamma_slider.setValue(15)  # 1.5
        self.width_ax_slider.setValue(10)   # 1.0
        self.width_sag_slider.setValue(10)  # 1.0
        self.width_cor_slider.setValue(10)  # 1.0
        
    def _invalidate_enhancement_cache(self) -> None:
        """Invalidate the cache and trigger a refresh of all planes."""
        self._enhanced_cache = None
        self._refresh_frames()

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
                if plane_ix == 0:  # Axial
                    dy = self._image_data.pixdim[1]
                    dx = self._image_data.pixdim[0]
                    base_aspect = dy / dx if dx != 0 else 1
                    aspect = base_aspect * self._width_scale_axial
                elif plane_ix == 1:  # Sagittal
                    dy = self._image_data.pixdim[1]
                    dz = self._image_data.pixdim[2]
                    base_aspect = dy / dz if dz != 0 else 1
                    aspect = base_aspect * self._width_scale_sagittal
                elif plane_ix == 2:  # Coronal
                    dx = self._image_data.pixdim[0]
                    dz = self._image_data.pixdim[2]
                    base_aspect = dx / dz if dz != 0 else 1
                    aspect = base_aspect * self._width_scale_coronal
                else:
                    self.show_error(f"Invalid plane index: {plane_ix}")
                fig.clear()
                ax = fig.add_subplot(111)
                ax.axis('off')
                # Get initial slice
                slice_arr = self._get_plane_slice(plane_ix, initializing=True)
                mask_arr = self._get_mask_slice(plane_ix)

                current_cmap = philips_cmap if self._use_philips_ceus else 'gray'
                artist = ax.imshow(slice_arr, cmap=current_cmap, aspect=float(aspect), zorder=1, animated=True) # add vmin and vmax for the 0 - 255 show
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

    def _get_plane_slice(self, plane_ix: int, initializing=False):
        """Return 2D numpy slice for given plane index based on current crosshair."""
        idx = self._get_plane_indices(plane_ix)
        current_t = self._crosshair_xyzt[3]
        
        # Check if we need to enhance a new frame
        if not initializing and (self._enhanced_cache is None or self._enhanced_cache_frame != current_t):
            # Get the 3D volume for current time frame
            current_frame_3d = self._pix_data[:, :, :, current_t]
            
            # Enhance the entire 3D volume ONCE per frame
            self._enhance_volume(current_frame_3d) # performs enhancement SYNCHRONOUSLY
            self._enhanced_cache_frame = current_t
        elif initializing:
            self._enhanced_cache = self._image_data.pixel_data[:, :, :, current_t]  # Cache the initial frame without enhancement for faster startup
        
        # Extract the 2D slice from cached enhanced volume
        slice_idx = list(idx[:3])  # Remove time dimension
        arr = self._enhanced_cache[tuple(slice_idx)]
        
        if arr.ndim != 2:
            arr = arr.squeeze()
        # Axial plane (index 0) needs transpose for correct orientation
        if plane_ix == 0:
            arr = arr.T
        return arr
    
    def _enhance_volume(self, volume_3d: np.ndarray) -> None:
        """Enhance a 3D image volume using predefined enhancement methods in the backend engine."""
        # Create a temporary UltrasoundImage for the current frame
        temp_im = UltrasoundImage(self._image_data.scan_path)
        temp_im.pixel_data = volume_3d.T[None].T.copy()  # Add back time dimension for processing
        temp_im.pixdim = self._image_data.pixdim
        temp_im.frame_rate = self._image_data.frame_rate

        clahe_preproc_dict = {
            'name': 'enhance_clahe',
            'image_data': temp_im,
            'frame_ix': self._crosshair_xyzt[3],
            'kwargs': {
                'clip_limit': self._clahe_clip_limit,
                'tile_grid_size': (8, 8),
            }
        }

        gamma_preproc_dict = {
            'name': 'enhance_gamma',
            'image_data': None,  # signal to reuse the already CLAHE-enhanced image (all preprocs in the same batch share the same image input)
            'frame_ix': self._crosshair_xyzt[3],
            'kwargs': {
                'gamma': self._gamma,
            }
        }

        preproc_dicts = [clahe_preproc_dict, gamma_preproc_dict]
        self.apply_preprocs_preview.emit(preproc_dicts) # synchronous call to apply the enhancements and update the cache via the connected slot

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
            self._update_scan_display()
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
        
        # Configure slice/time controls
        self._ui.cur_slice_slider.setMinimum(0)
        self._ui.cur_slice_slider.setMaximum(max(0, self._num_slices - 1))
        self._ui.cur_slice_slider.setValue(self._crosshair_xyzt[3])
        self._ui.cur_slice_slider.valueChanged.connect(self._on_time_slider_changed)

        # Configure spin box for frame-based navigation
        self._ui.cur_slice_spin_box.setRange(1, self._num_slices)
        self._ui.cur_slice_spin_box.setSingleStep(1)
        self._ui.cur_slice_spin_box.setDecimals(0)
        self._ui.cur_slice_spin_box.setValue(self._crosshair_xyzt[3] + 1)
        self._ui.cur_slice_spin_box.valueChanged.connect(self._on_time_spin_box_changed)

        # Set initial total frames for all planes
        self._ui.ax_total_frames.setText(str(self._z_len))
        self._ui.sag_total_frames.setText(str(self._x_len))
        self._ui.cor_total_frames.setText(str(self._y_len))
        self._ui.cur_slice_total.setText(str(self._num_slices))

    def _on_time_spin_box_changed(self, value: float):
        """Handle user changing the time spin box."""
        frame_idx = int(value) - 1
        
        # Clamp to valid range
        frame_idx = max(0, min(self._num_slices - 1, frame_idx))
        
        if self._ui.cur_slice_slider.value() != frame_idx:
            self._ui.cur_slice_slider.setValue(frame_idx)

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
        if hasattr(self, '_last_saved_path'):
            self.segmentation_saved.emit(str(self._last_saved_path))

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

        # Update total frame labels
        self._ui.ax_total_frames.setText(str(self._z_len))
        self._ui.sag_total_frames.setText(str(self._x_len))
        self._ui.cor_total_frames.setText(str(self._y_len))

        # Update current slice/frame display
        current_t = self._crosshair_xyzt[3]
        
        # Block signals to avoid feedback loop
        self._ui.cur_slice_spin_box.blockSignals(True)
        self._ui.cur_slice_spin_box.setValue(current_t + 1)
        self._ui.cur_slice_spin_box.blockSignals(False)
        
        self._ui.cur_slice_total.setText(str(self._num_slices))

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
        self._last_saved_path = out_path

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
