"""
Segmentation Preview Widget for Segmentation Loading
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.path import Path as Mpl_Path

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSizePolicy, QLabel
from PyQt6.QtCore import pyqtSignal, Qt, QEvent

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.seg_loading.ui.voi_preview_ui import Ui_confirm_voi
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg


class VoiPreviewWidget(QWidget, BaseViewMixin):
    """
    Widget for previewing and confirming segmentation.
    
    This is the final step in the segmentation loading process where users
    can preview the loaded segmentation and confirm it before proceeding.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    segmentation_confirmed = pyqtSignal(object)  # BmodeSeg data
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    
    def __init__(self, seg_data: BmodeSeg, image_data: UltrasoundRfImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_confirm_voi()
        self._seg_data = seg_data
        self._image_data = image_data
        self._matplotlib_canvas: Optional[FigureCanvas] = None
        self._frame = seg_data.frame
        
        if image_data.sc_bmode.ndim == 4:
            self._pix_data = image_data.sc_bmode[:, :, :, self._frame]
        else:
            self._pix_data = image_data.sc_bmode
        
        # Crosshair / navigation state
        self._crosshair_active = False
        self._crosshair_visible = True
        self._crosshair_xyz = [0, 0, 0]  # x,y,z indices
        
        # Dimension cache
        self._x_len, self._y_len, self._z_len = self._pix_data.shape
        self._crosshair_xyz = [self._x_len // 2, self._y_len // 2, self._z_len // 2]
        
        # Per-plane resources (axial, sagittal, coronal)
        self._ax_sag_cor_matplotlib_canvases = [None, None, None]
        self._ax_sag_cor_planes = (None, None, None)
        self._ax_sag_cor_index_maps = ((1, 0), (2, 0), (2, 1))  # dims that vary per plane
        self._ax_sag_cor_animations = [None, None, None]
        self._ax_sag_cor_plane_artists = [None, None, None]
        self._ax_sag_cor_crosshair_lines = [(None, None), (None, None), (None, None)]
        self._ax_sag_cor_pending = [False, False, False]
        self._ax_sag_cor_seg_masks = [None, None, None]       # segmentation masks
        
        self._seg_mask_arr = seg_data.sc_seg_mask if hasattr(seg_data, 'sc_seg_mask') and seg_data.sc_seg_mask is not None else seg_data.seg_mask
        self._roi_masks_overlap = np.zeros((self._x_len, self._y_len, self._z_len, 4), dtype=np.uint8)
        self._roi_masks_overlap[self._seg_mask_arr > 0] = [255, 0, 0, 125]
        
        self._setup_ui()
        self._initialize_plane_displays()
        self._setup_all_plane_animations()
        self._connect_signals()
        self._connect_matplotlib_events()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Store QLabels to show images in each plane
        self._ax_sag_cor_planes = (self._ui.ax_plane, self._ui.sag_plane, self._ui.cor_plane)
        
        # Configure layout for segmentation preview only - use the main layout
        self.setLayout(self._ui.full_screen_layout)

        # Configure stretch factors for confirmation
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.voi_layout, 10)
        
        # Ensure the layout fills the entire widget
        self._ui.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.full_screen_layout.setSpacing(0)
        self._ui.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.full_screen_layout.setSpacing(0)
        
        # Update labels to reflect inputted image, phantom, and frame
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

        # Update alpha slider to default value
        self._ui.alpha_spin_box.setMaximum(255); self._ui.alpha_spin_box.setMinimum(0)
        self._ui.alpha_status.setMaximum(255); self._ui.alpha_status.setMinimum(0)
        self._ui.alpha_spin_box.setValue(125); self._ui.alpha_status.setValue(125)

        if self._image_data.rf_data.ndim == 4:
            self._ui.confirmation_frame_label.hide()
        else:
            self._ui.confirmation_frame_label.setText(f"Frame: {self._frame}")

        # Setup matplotlib canvas for segmentation preview
        self._ui.navigating_label.hide()
        self._setup_matplotlib_canvases()
        
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
    
    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.confirm_seg_button.clicked.connect(self._on_confirm_segmentation)
        self._ui.back_from_confirm_button.clicked.connect(self._on_back_clicked)
        self._ui.alpha_spin_box.valueChanged.connect(self._on_alpha_changed)

    def _initialize_plane_displays(self) -> None:
        """Initialize all 2D plane displays with optimized matplotlib setup."""
        for plane_ix, canvas in enumerate(self._ax_sag_cor_matplotlib_canvases):
            if not canvas:
                continue
            try:
                fig = canvas.figure
                if plane_ix == 0:  # Axial: y vs x
                    aspect = (self._image_data.sc_coronal_res) / (self._image_data.sc_lateral_res) if self._image_data.sc_coronal_res != 0 else 1
                elif plane_ix == 1:  # Sagittal: y vs z
                    aspect = (self._image_data.sc_axial_res) / (self._image_data.sc_lateral_res) if self._image_data.sc_axial_res != 0 else 1
                elif plane_ix == 2:  # Coronal: x vs z
                    aspect = (self._image_data.sc_axial_res) / (self._image_data.sc_coronal_res) if self._image_data.sc_axial_res != 0 else 1
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
                
                self._ax_sag_cor_plane_artists[plane_ix] = artist
                self._ax_sag_cor_crosshair_lines[plane_ix] = (v_line, h_line)
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
        return arr

    def _get_mask_slice(self, plane_ix: int):
        """Return RGBA numpy slice for the mask of the given plane index."""
        idx = self._get_plane_indices(plane_ix) # no time dimension
        arr = self._roi_masks_overlap[idx]
        return arr

    def _get_plane_indices(self, plane_ix: int) -> Tuple[int]:
        """Return a list of indices for the given plane."""
        idx = self._crosshair_xyz[:]
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
            self._update_seg_masks(plane_ix)
            
            v_line, h_line = self._ax_sag_cor_crosshair_lines[plane_ix]
            mask = self._ax_sag_cor_seg_masks[plane_ix]
            artists = [self._ax_sag_cor_plane_artists[plane_ix]]
            if v_line: artists.append(v_line)
            if h_line: artists.append(h_line)
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
        x_idx = self._crosshair_xyz[x_dim]
        y_idx = self._crosshair_xyz[y_dim]
        v_line.set_xdata([x_idx, x_idx])
        h_line.set_ydata([y_idx, y_idx])

        if not self._crosshair_visible:
            v_line.set_visible(False); h_line.set_visible(False)
        else:
            # Ensure visible when expected (avoids lingering hidden state)
            v_line.set_visible(True); h_line.set_visible(True)
            
    # ------------------------ Public API ------------------------------------
    def set_crosshair(self, x: Optional[int] = None, y: Optional[int] = None,
                      z: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Set (and clamp) crosshair indices then flag planes for redraw.

        Parameters are optional; only provided axes are updated. Values are
        clamped into valid bounds. All three orthogonal plane views are marked
        pending so the animation loop refreshes them on the next frame.
        Returns the updated (x,y,z) tuple.
        """
        # Current values
        cx, cy, cz = self._crosshair_xyz
        if x is not None:
            cx = max(0, min(self._x_len - 1, int(x)))
        if y is not None:
            cy = max(0, min(self._y_len - 1, int(y)))
        if z is not None:
            cz = max(0, min(self._z_len - 1, int(z)))
        # Only proceed if changed
        if [cx, cy, cz] != self._crosshair_xyz:
            self._crosshair_xyz = [cx, cy, cz]
            self._refresh_frames()
        return cx, cy, cz

    def _update_seg_masks(self, plane_ix):
        """Create/update the segmentation masks for frames on a given plane for blitting."""
        mask_2d = self._get_mask_slice(plane_ix)
        self._ax_sag_cor_seg_masks[plane_ix].set_array(mask_2d)
        
    def _on_toggle_crosshair_visibility(self):
        # Toggle visibility state but keep indices updating
        self._crosshair_visible = not self._crosshair_visible
        self._refresh_frames()
        self._ui.toggle_crosshair_visibility_button.setText(
            'Show Crosshair' if not self._crosshair_visible else 'Hide Crosshair'
        )

    def _on_alpha_changed(self):
        # Update alpha for segmentation overlay
        alpha = self._ui.alpha_spin_box.value()
        self._ui.alpha_status.setValue(alpha)
        self._roi_masks_overlap[self._seg_mask_arr > 0] = [255, 0, 0, int(alpha)]
        self._refresh_frames()

    def _refresh_frames(self) -> None:
        """Refresh the displayed frames."""
        for i in range(3):
            self._ax_sag_cor_pending[i] = True
            
    def mousePressEvent(self, a0):
        super().mousePressEvent(a0)
        self._crosshair_active = not self._crosshair_active
        if self._crosshair_active:
            self._ui.navigating_label.show(); self._ui.observing_label.hide()
        else:
            self._ui.navigating_label.hide(); self._ui.observing_label.show()
    
    def keyPressEvent(self, event):  # type: ignore
        """Handle key presses for quick actions (e.g., 'd' to toggle draw ROI)."""
        if event.key() == Qt.Key.Key_H:
            self._on_toggle_crosshair_visibility()
            return
        super().keyPressEvent(event)
        
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
        self._crosshair_active = not self._crosshair_active
        if self._crosshair_active:
            self._ui.navigating_label.show()
            self._ui.observing_label.hide()
        else:
            self._ui.navigating_label.hide()
            self._ui.observing_label.show()
            
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
        dim_lengths = [self._x_len, self._y_len, self._z_len]

        # Proposed new indices (int rounding & clamp)
        new_xval = int(round(event.xdata))
        new_yval = int(round(event.ydata))
        if new_xval < 0 or new_yval < 0:
            return
        if new_xval >= dim_lengths[dim_x] or new_yval >= dim_lengths[dim_y]:
            return

        # Build kwargs for set_crosshair only for dims that change
        params = {}
        if self._crosshair_xyz[dim_x] != new_xval:
            if dim_x == 0: params['x'] = new_xval
            elif dim_x == 1: params['y'] = new_xval
            elif dim_x == 2: params['z'] = new_xval
            elif dim_x == 3: params['t'] = new_xval
        if self._crosshair_xyz[dim_y] != new_yval:
            if dim_y == 0: params['x'] = new_yval
            elif dim_y == 1: params['y'] = new_yval
            elif dim_y == 2: params['z'] = new_yval
            elif dim_y == 3: params['t'] = new_yval

        if params:
            self.set_crosshair(**params)
    
    def _update_scan_display(self):
        """Update the scan display with the current frames and frame numbers"""
        # Update frame numbers
        self._ui.ax_frame_num.setText(str(self._crosshair_xyz[2] + 1))
        self._ui.sag_frame_num.setText(str(self._crosshair_xyz[0] + 1))
        self._ui.cor_frame_num.setText(str(self._crosshair_xyz[1] + 1))

    def _on_confirm_segmentation(self) -> None:
        """Handle segmentation confirmation."""
        if not self._seg_data:
            self.show_error("No segmentation data to confirm")
            return
            
        self.segmentation_confirmed.emit(self._seg_data)
        
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
        
    def get_segmentation_data(self) -> BmodeSeg:
        """Get the current segmentation data."""
        return self._seg_data


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
