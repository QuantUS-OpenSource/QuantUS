"""
Segmentation Preview Widget for CEUS
"""

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QSizePolicy, QSlider, QVBoxLayout, QFrame, QCheckBox, QPushButton, QFileDialog
from PyQt6.QtCore import QEvent, pyqtSignal, Qt

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from ...mvc.base_view import BaseViewMixin
from ..ui.draw_voi_ui import Ui_voi_drawer
from engines.ceus.src.data_objs import UltrasoundImage, CeusSeg
from engines.ceus.src.image_preprocessing.functions import enhance_clahe, enhance_gamma

# Philips CEUS Colormap: Grayscale -> Red -> Yellow
philips_colors = [
    (0.0, 0.0, 0.0),    # 0% - Black
    (0.4, 0.4, 0.4),    # 40% - Gray
    (0.8, 0.0, 0.0),    # 80% - Red
    (1.0, 1.0, 0.0)     # 100% - Yellow
]
philips_cmap = LinearSegmentedColormap.from_list("philips_ceus", philips_colors)

class SegPreviewWidget(QWidget, BaseViewMixin):
    """
    Widget for previewing and confirming segmentation for CEUS.
    Reuses UI components from VOI drawer but in a read-only preview mode.
    """
    
    # Signals for communicating with controller
    segmentation_confirmed = pyqtSignal()
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, image_data: UltrasoundImage, seg_data: CeusSeg, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_voi_drawer()
        self._image_data = image_data
        self._seg_data = seg_data
        self._pix_data = image_data.pixel_data
        
        # Enhancement parameters (Inherited from seg_data)
        self._clahe_clip_limit = getattr(seg_data, 'clahe_clip_limit', 1.2)
        self._gamma = getattr(seg_data, 'gamma', 1.5)
        self._width_scale_axial = getattr(seg_data, 'width_scale_axial', 1.0)
        self._width_scale_sagittal = getattr(seg_data, 'width_scale_sagittal', 1.0)
        self._width_scale_coronal = getattr(seg_data, 'width_scale_coronal', 1.0)
        self._use_philips_ceus = getattr(seg_data, 'use_philips_ceus', False)
        self._mask_alpha = 125 # Default alpha for mask overlay (0-255)
        
        # Cache for enhanced volume
        self._enhanced_cache = None
        self._enhanced_cache_frame = -1

        # Crosshair / navigation state
        self._crosshair_active = False
        self._crosshair_visible = True
        
        # Dimensions: x, y, z, t
        if self._pix_data.ndim == 4:
            self._x_len, self._y_len, self._z_len, self._num_slices = self._pix_data.shape
        else:
            # Fallback if 3D
            self._x_len, self._y_len, self._z_len = self._pix_data.shape
            self._num_slices = 1
            self._pix_data = self._pix_data.reshape((self._x_len, self._y_len, self._z_len, 1))

        self._crosshair_xyzt = [self._x_len // 2, self._y_len // 2, self._z_len // 2, 0]
        
        # Segmentation mask overlay
        self._roi_masks_overlap = np.zeros((self._x_len, self._y_len, self._z_len, 4), dtype=np.uint8)
        self._seg_mask_indices = None # Store binary mask indices for alpha updates
        if hasattr(seg_data, 'seg_mask') and seg_data.seg_mask is not None:
            # seg_mask should be same spatial shape (x, y, z)
            mask = seg_data.seg_mask
            
            # Ensure spatial alignment with image data if dimensions are flipped or permuted
            # If the mask was saved from DrawVOIWidget, it should match the pixel_data shape
            if mask.shape == (self._x_len, self._y_len, self._z_len):
                self._seg_mask_indices = np.where(mask > 0)
            elif mask.shape == (self._y_len, self._x_len, self._z_len):
                # Handle common XY transpose if detected
                self._seg_mask_indices = np.where(mask.transpose(1, 0, 2) > 0)
            else:
                # Log or handle shape mismatch more gracefully if needed
                print(f"Warning: Mask shape {mask.shape} does not match image shape {(self._x_len, self._y_len, self._z_len)}")
                # Try to fit the mask as much as possible if shapes match in 3D volume
                try:
                    self._seg_mask_indices = np.where(mask[:self._x_len, :self._y_len, :self._z_len] > 0)
                except Exception:
                    pass
            
            if self._seg_mask_indices is not None and len(self._seg_mask_indices[0]) > 0:
                self._roi_masks_overlap[self._seg_mask_indices[0], self._seg_mask_indices[1], self._seg_mask_indices[2]] = [255, 0, 0, 125]

        # Jump crosshair to a point within the mask to show it immediately
        mask_indices = np.where(self._roi_masks_overlap[..., 3] > 0)
        if len(mask_indices[0]) > 0:
            mid_idx = len(mask_indices[0]) // 2
            self._crosshair_xyzt = [
                mask_indices[0][mid_idx],
                mask_indices[1][mid_idx],
                mask_indices[2][mid_idx],
                0
            ]
        else:
            self._crosshair_xyzt = [self._x_len // 2, self._y_len // 2, self._z_len // 2, 0]

        # Per-plane resources (axial, sagittal, coronal)
        self._ax_sag_cor_matplotlib_canvases = [None, None, None]
        self._ax_sag_cor_planes = (None, None, None)
        self._ax_sag_cor_index_maps = ((0, 1), (2, 1), (0, 2))  # (horiz_dim, vert_dim)
        self._ax_sag_cor_animations = [None, None, None]
        self._ax_sag_cor_plane_artists = [None, None, None]
        self._ax_sag_cor_crosshair_lines = [(None, None), (None, None), (None, None)]
        self._ax_sag_cor_pending = [False, False, False]
        self._ax_sag_cor_seg_masks = [None, None, None]

        # UI & visualization setup
        self._setup_ui()
        self._setup_matplotlib_canvases()
        self._initialize_plane_displays()
        self._setup_all_plane_animations()
        self._connect_signals()
        self._connect_matplotlib_events()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_scan_display()
        self._refresh_frames()

    def _setup_ui(self) -> None:
        """Setup the user interface to match the segmentation menu style."""
        self._ui.setupUi(self)

        # Store QLabels as tags for layout mapping
        self._ax_sag_cor_planes = (self._ui.ax_plane, self._ui.sag_plane, self._ui.cor_plane)

        # Configure layout
        self.setLayout(self._ui.full_screen_layout)
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.voi_layout, 10)
        
        # Widget groups matching DrawVOIWidget for consistency
        self._drawing_widgets = [
            self._ui.draw_roi_button,
            self._ui.interpolate_voi_button,
            self._ui.undo_last_pt_button,
            self._ui.close_roi_button,
            self._ui.undo_last_roi_button,
            self._ui.construct_voi_label,
        ]
        
        # Add a "Confirm & Analysis" button programmatically
        self.confirm_review_button = QPushButton("Confirm && Analysis")
        self.confirm_review_button.setMinimumSize(self._ui.save_voi_button.minimumSize())
        self.confirm_review_button.setMaximumSize(self._ui.save_voi_button.maximumSize())
        self.confirm_review_button.setStyleSheet(self._ui.save_voi_button.styleSheet())
        
        # Insert it into the layout that has Restart and Save
        self._ui.horizontalLayout_2.addWidget(self.confirm_review_button)
        
        # Update existing button texts for clarity in preview mode
        self._ui.restart_voi_button.setText("Review / Redraw")
        self._ui.save_voi_button.setText("Save Setup")
        
        self._voi_decision_widgets = [
            self._ui.restart_voi_button,
            self._ui.save_voi_button,
            self.confirm_review_button
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

        # Initial visibility
        self._ui.scan_name_input.setText(self._image_data.scan_name)
        self._ui.segSidebarLabel_2.setText("Segmentation Selection")
        self._ui.toggle_crosshair_visibility_button.setText('Hide Crosshair')
        self._ui.cur_slice_label.setText("Current Frame:")
        
        self._hide_widget_lists([self._drawing_widgets, self._save_voi_widgets, self._voi_alpha_widgets])
        self._show_widget_lists([self._voi_decision_widgets])
        
        # Hide original plane labels (replaced by canvases)
        # Note: We do NOT hide ax_plane, sag_plane, cor_plane here because they are needed as containers
        for widget in [self._ui.interp_loading_label, self._ui.saving_voi_label]:
            widget.hide()

        self._ui.navigating_label.hide()
        self._ui.observing_label.show()

        # Setup enhancement controls
        self._setup_enhancement_controls()
        
        # Update slider for frames
        self._ui.cur_slice_slider.setMinimum(0)
        self._ui.cur_slice_slider.setMaximum(self._num_slices - 1)
        self._ui.cur_slice_slider.setValue(0)
        self._ui.cur_slice_total.setText(str(self._num_slices))
        self._ui.cur_slice_spin_box.setRange(1, self._num_slices)
        self._ui.cur_slice_spin_box.setValue(1)
        
        self._ui.ax_total_frames.setText(str(self._z_len))
        self._ui.sag_total_frames.setText(str(self._x_len))
        self._ui.cor_total_frames.setText(str(self._y_len))

        # Install event filters
        for label in self._ax_sag_cor_planes:
            if label:
                label.installEventFilter(self)

    def _cleanup_animations(self):
        """Internal helper to stop animations safely."""
        for i in range(3):
            if i < len(self._ax_sag_cor_animations) and self._ax_sag_cor_animations[i]:
                try:
                    self._ax_sag_cor_animations[i].event_source.stop()
                except Exception:
                    pass
                self._ax_sag_cor_animations[i] = None

    # ============================================================================
    # UI SETUP & HELPERS
    # ============================================================================

    def _show_widget_lists(self, widget_lists: List[List[QWidget]]) -> None:
        """Helper to show groups of widgets."""
        for widget_list in widget_lists:
            for widget in widget_list:
                widget.show()

    def _hide_widget_lists(self, widget_lists: List[List[QWidget]]) -> None:
        """Helper to hide groups of widgets."""
        for widget_list in widget_lists:
            for widget in widget_list:
                widget.hide()

    def _setup_enhancement_controls(self) -> None:
        """Add enhancement sliders to the sidebar, mirroring DrawVOIWidget style."""
        enh_group = QFrame()
        enh_group.setStyleSheet("background-color: rgba(255, 255, 255, 0); border: none;")
        
        container_layout = QVBoxLayout(enh_group)
        container_layout.setContentsMargins(0, 10, 0, 10)
        container_layout.setSpacing(15)

        row1_layout = QHBoxLayout()
        row2_layout = QHBoxLayout()
        row1_layout.setSpacing(20)
        row2_layout.setSpacing(20)

        def create_enh_column(label_text, min_val, max_val, current_val, callback):
            col_widget = QWidget()
            col_layout = QVBoxLayout(col_widget)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(5)
            
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 14px; color: white; font-weight: bold;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col_layout.addWidget(lbl)
            
            row_layout = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(current_val)
            slider.setStyleSheet(self._ui.cur_slice_slider.styleSheet())
            slider.setMinimumWidth(80)
            slider.setMaximumWidth(120)
            slider.valueChanged.connect(callback)
            
            val_lbl = QLabel(f"{current_val/10.0:.1f}")
            val_lbl.setMinimumWidth(40)
            val_lbl.setStyleSheet("color: #3498db; font-weight: bold; font-size: 14px;")
            
            row_layout.addWidget(slider)
            row_layout.addWidget(val_lbl)
            col_layout.addLayout(row_layout)
            return col_widget, slider, val_lbl

        # Sliders
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
        
        # Add VOI Alpha slider
        alpha_col, self.alpha_slider, self.alpha_val_lbl = create_enh_column(
            "VOI ALPHA", 0, 2550, int(self._mask_alpha * 10), lambda v: self._on_alpha_changed(v // 10)
        )
        # Fix label to show integer for alpha
        self.alpha_val_lbl.setText(str(self._mask_alpha))
        self.alpha_slider.valueChanged.disconnect()
        self.alpha_slider.valueChanged.connect(lambda v: (self._on_alpha_changed(v // 10), self.alpha_val_lbl.setText(str(v // 10))))

        row1_layout.addWidget(clahe_col)
        row1_layout.addWidget(gamma_col)
        
        self.philips_check = QCheckBox("Pseudocoloring")
        self.philips_check.setChecked(self._use_philips_ceus)
        self.philips_check.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        self.philips_check.stateChanged.connect(self._on_philips_toggled)
        row1_layout.addWidget(self.philips_check)
        row1_layout.addWidget(alpha_col)

        row2_layout.addWidget(width_ax_col)
        row2_layout.addWidget(width_sag_col)
        row2_layout.addWidget(width_cor_col)
        
        container_layout.addLayout(row1_layout)
        container_layout.addLayout(row2_layout)
        self._ui.verticalLayout_2.addWidget(enh_group)

    def _invalidate_enhancement_cache(self) -> None:
        """Clear cache when processing parameters change."""
        self._enhanced_cache = None
        self._enhanced_cache_frame = -1
        self._refresh_frames()

    def _on_clahe_changed(self, value: int) -> None:
        """Handle CLAHE change."""
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
        """Handle axial aspect ratio."""
        self._width_scale_axial = value / 10.0
        if hasattr(self, 'width_ax_val_lbl'):
            self.width_ax_val_lbl.setText(f"{self._width_scale_axial:.1f}")
        self._update_aspect_ratios()

    def _on_width_sagittal_changed(self, value: int) -> None:
        """Handle sagittal aspect ratio."""
        self._width_scale_sagittal = value / 10.0
        if hasattr(self, 'width_sag_val_lbl'):
            self.width_sag_val_lbl.setText(f"{self._width_scale_sagittal:.1f}")
        self._update_aspect_ratios()

    def _on_width_coronal_changed(self, value: int) -> None:
        """Handle coronal aspect ratio."""
        self._width_scale_coronal = value / 10.0
        if hasattr(self, 'width_cor_val_lbl'):
            self.width_cor_val_lbl.setText(f"{self._width_scale_coronal:.1f}")
        self._update_aspect_ratios()

    def _on_alpha_changed(self, value: int) -> None:
        """Handle alpha transparency change for the VOI mask."""
        self._mask_alpha = value
        # Update the rgba mask transparency
        # Use stored mask indices to ensure we can recover from alpha=0
        if self._seg_mask_indices is not None and len(self._seg_mask_indices[0]) > 0:
            # Re-apply color (Red) and new alpha to the relevant indices
            self._roi_masks_overlap[self._seg_mask_indices[0], self._seg_mask_indices[1], self._seg_mask_indices[2]] = [255, 0, 0, self._mask_alpha]
        self._refresh_frames()

    def _update_aspect_ratios(self) -> None:
        """Update artist aspect ratios based on physics (pixdim) and sliders."""
        if not hasattr(self, '_image_data') or self._image_data is None:
            return

        try:
            pix = self._image_data.pixdim
            
            # Plane 0: Axial (XY) -> show (Y, X) -> Rows=Y, Cols=X -> dy / dx
            if self._ax_sag_cor_matplotlib_canvases[0]:
                dx, dy = pix[0], pix[1]
                aspect_ax = (dy / dx if dx != 0 else 1.0) * self._width_scale_axial
                self._ax_sag_cor_matplotlib_canvases[0].figure.gca().set_aspect(aspect_ax)
            
            # Plane 1: Sagittal (YZ) -> 90 CW Rotation -> show (Y, Z) -> Rows=Y, Cols=Z -> dy / dz
            if self._ax_sag_cor_matplotlib_canvases[1]:
                dy, dz = pix[1], pix[2]
                aspect_sag = (dy / dz if dz != 0 else 1.0) * self._width_scale_sagittal
                self._ax_sag_cor_matplotlib_canvases[1].figure.gca().set_aspect(aspect_sag)
                
            # Plane 2: Coronal (XZ) -> show (Z, X) -> Rows=Z, Cols=X -> dz / dx
            if self._ax_sag_cor_matplotlib_canvases[2]:
                dx, dz = pix[0], pix[2]
                aspect_cor = (dz / dx if dx != 0 else 1.0) * self._width_scale_coronal
                self._ax_sag_cor_matplotlib_canvases[2].figure.gca().set_aspect(aspect_cor)

            for canvas in self._ax_sag_cor_matplotlib_canvases:
                if canvas:
                    canvas.draw_idle()
            
            self._refresh_frames()
        except Exception as e:
            print(f"Error updating aspect ratios: {e}")

    def _on_philips_toggled(self, state: int) -> None:
        """Handle Philips CEUS pseudocolor toggle."""
        self._use_philips_ceus = state == Qt.CheckState.Checked.value
        new_cmap = philips_cmap if self._use_philips_ceus else 'gray'
        for artist in self._ax_sag_cor_plane_artists:
            if artist:
                artist.set_cmap(new_cmap)
        self._refresh_frames()

    def _setup_matplotlib_canvases(self) -> None:
        """Initialize and embed matplotlib canvases into each plane's placeholder layout."""
        # Use the axial, sagittal, coronal labels themselves as the parent for the canvases
        # This ensures they are inside their respective QFrame boxes and aligned correctly
        for i, parent_label in enumerate(self._ax_sag_cor_planes):
            fig, ax = plt.subplots(facecolor='black')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.setParent(parent_label)
            
            # Use Expanding policy
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            # Hide the label's text but keep the background/frame
            parent_label.setText("")
            
            self._ax_sag_cor_matplotlib_canvases[i] = canvas

    def _initialize_plane_displays(self) -> None:
        """Initial render of each orthogonal plane with fixed intensity scaling."""
        for i, canvas in enumerate(self._ax_sag_cor_matplotlib_canvases):
            if not canvas: continue
            ax = canvas.figure.axes[0]
            
            # Initial data slice
            slice_data = self._get_plane_slice(i)
            # Use fixed vmin/vmax to prevent auto-scaling contrast per slice
            artist = ax.imshow(slice_data, cmap='gray', interpolation='nearest', 
                               zorder=1, vmin=0, vmax=255)
            self._ax_sag_cor_plane_artists[i] = artist

            # Mask overlay
            mask_slice = self._get_mask_slice(i)
            # Match interpolation and zorder to DrawVOIWidget
            mask_artist = ax.imshow(mask_slice, interpolation='nearest', 
                                    zorder=8)
            self._ax_sag_cor_seg_masks[i] = mask_artist

            # Crosshair lines
            # Get actual coordinate indices for current plane from maps
            idx_x, idx_y = self._ax_sag_cor_index_maps[i]
            v_line = ax.axvline(x=self._crosshair_xyzt[idx_x], color='yellow', lw=0.8, animated=True, zorder=11)
            h_line = ax.axhline(y=self._crosshair_xyzt[idx_y], color='yellow', lw=0.8, animated=True, zorder=11)
            self._ax_sag_cor_crosshair_lines[i] = (v_line, h_line)
        
        self._update_aspect_ratios()

    def _get_plane_slice(self, plane_ix: int) -> np.ndarray:
        """Extract a 2D image slice for the specified plane at current crosshair indices."""
        x, y, z, t = self._crosshair_xyzt
        vol = self._get_enhanced_volume(t)
        
        if plane_ix == 0:  # Axial (XY) at Z -> show (Y, X)
            return vol[:, :, z].T
        elif plane_ix == 1:  # Sagittal (YZ) at X -> show (Z, Y) then rotate 90 CW -> (Y, Z)
            # Match DrawVOIWidget approach: arr.T then rot90(k=-1)
            arr = vol[x, :, :]
            arr_t = arr.T
            return np.rot90(arr_t, k=-1)
        elif plane_ix == 2:  # Coronal (XZ) at Y -> show (Y, X)
            # Mirror Axial for Coronal to match DrawVOI behavior
            return vol[:, y, :].T
        return np.zeros((10, 10))

    def _get_mask_slice(self, plane_ix: int) -> np.ndarray:
        """Extract a 2D mask slice for overlay."""
        x, y, z, _ = self._crosshair_xyzt
        if plane_ix == 0:  # Axial (XY) at Z -> show (Y, X)
            # Match DrawVOIWidget approach: index mask then transpose (1, 0, 2)
            arr = self._roi_masks_overlap[:, :, z, :]
            return np.transpose(arr, (1, 0, 2))
        elif plane_ix == 1:  # Sagittal (YZ) at X -> show (Z, Y) then rotate 90 CW -> (Y, Z)
            arr = self._roi_masks_overlap[x, :, :, :]
            # Consistent with DrawVOIWidget: (1, 0, 2) then rot90(k=-1)
            arr_t = np.transpose(arr, (1, 0, 2))
            return np.rot90(arr_t, k=-1)
        elif plane_ix == 2:  # Coronal (XZ) at Y -> show (Y, X)
            # Mirror Axial for Coronal to match DrawVOI behavior
            arr = self._roi_masks_overlap[:, y, :, :]
            return np.transpose(arr, (1, 0, 2))
        return np.zeros((10, 10, 4), dtype=np.uint8)

    def _get_enhanced_volume(self, t: int) -> np.ndarray:
        """Apply image processing and return the 3D volume at frame t."""
        if self._enhanced_cache is not None and self._enhanced_cache_frame == t:
            return self._enhanced_cache
        
        # Extract the 3D volume for current frame
        vol_3d = self._pix_data[:, :, :, t]
        
        # Create a temporary UltrasoundImage for the engine preprocessors
        temp_im = UltrasoundImage(self._image_data.scan_path)
        temp_im.pixel_data = vol_3d
        temp_im.pixdim = self._image_data.pixdim
        temp_im.frame_rate = self._image_data.frame_rate
        
        # Apply backend engine functions
        temp_im = enhance_clahe(temp_im, clip_limit=self._clahe_clip_limit)
        temp_im = enhance_gamma(temp_im, gamma=self._gamma)
        
        self._enhanced_cache = temp_im.pixel_data
        self._enhanced_cache_frame = t
        return self._enhanced_cache

    def _setup_all_plane_animations(self) -> None:
        """Setup refresh animations for each matplotlib canvas."""
        for i in range(3):
            canvas = self._ax_sag_cor_matplotlib_canvases[i]
            self._ax_sag_cor_animations[i] = anim.FuncAnimation(
                canvas.figure,
                lambda frame, p_ix=i: self._update_plane(p_ix),
                interval=33, 
                blit=True,
                cache_frame_data=False
            )

    def _update_plane(self, plane_ix: int):
        """Update artist data for a single plane."""
        # Always return the list of artists for blitting
        v_line, h_line = self._ax_sag_cor_crosshair_lines[plane_ix]
        artist = self._ax_sag_cor_plane_artists[plane_ix]
        mask_artist = self._ax_sag_cor_seg_masks[plane_ix]
        
        artists = []
        if artist: artists.append(artist)
        if mask_artist: artists.append(mask_artist)
        if v_line: 
            v_line.set_visible(self._crosshair_visible)
            artists.append(v_line)
        if h_line: 
            h_line.set_visible(self._crosshair_visible)
            artists.append(h_line)

        if not self._ax_sag_cor_pending[plane_ix]:
            return artists
        
        if artist:
            artist.set_data(self._get_plane_slice(plane_ix))
        if mask_artist:
            mask_artist.set_data(self._get_mask_slice(plane_ix))
            
        if v_line and h_line:
            idx_x, idx_y = self._ax_sag_cor_index_maps[plane_ix]
            # When refreshing (e.g. slice changed), snap back to stored indices
            v_line.set_xdata([self._crosshair_xyzt[idx_x]])
            h_line.set_ydata([self._crosshair_xyzt[idx_y]])
            v_line.set_visible(self._crosshair_visible)
            h_line.set_visible(self._crosshair_visible)

        self._ax_sag_cor_pending[plane_ix] = False
        return artists

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers, matching DrawVOIWidget patterns."""
        # Frame/Time Navigation
        self._ui.cur_slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        self._ui.cur_slice_spin_box.valueChanged.connect(lambda v: self._ui.cur_slice_slider.setValue(int(v)-1))
        self._ui.toggle_crosshair_visibility_button.clicked.connect(self._on_toggle_crosshair)
        self._ui.back_button.clicked.connect(self._on_back_requested)

        # Decision Buttons
        self.confirm_review_button.clicked.connect(self._on_confirm_review)
        self._ui.restart_voi_button.clicked.connect(self._on_back_requested)
        self._ui.save_voi_button.clicked.connect(self._on_save_voi_clicked)
        
        # Save Form Actions
        self._ui.back_from_save_button.clicked.connect(self._on_back_from_save_clicked)
        self._ui.choose_save_folder_button.clicked.connect(self._on_choose_save_folder)
        self._ui.clear_save_folder_button.clicked.connect(lambda: self._ui.save_folder_input.clear())
        self._ui.export_voi_button.clicked.connect(self._on_export_voi_clicked)

    def _on_back_requested(self):
        """Handle back request with cleanup."""
        self._cleanup_animations()
        self.back_requested.emit()

    def _on_confirm_review(self):
        """Handle confirmation with cleanup."""
        self._cleanup_animations()
        self.segmentation_confirmed.emit()

    def _on_save_voi_clicked(self) -> None:
        """Switch to the save file configuration menu."""
        self._hide_widget_lists([self._voi_decision_widgets])
        self._show_widget_lists([self._save_voi_widgets, self._voi_alpha_widgets])
        # Default save name
        self._ui.save_name_input.setText(f"{self._image_data.scan_name}_mask")

    def _on_back_from_save_clicked(self) -> None:
        """Switch back from save menu to decision menu."""
        self._hide_widget_lists([self._save_voi_widgets, self._voi_alpha_widgets])
        self._show_widget_lists([self._voi_decision_widgets])

    def closeEvent(self, event):
        """Clean up animations and canvases before the widget is destroyed."""
        self._cleanup_animations()
        
        for i in range(3):
            canvas = self._ax_sag_cor_matplotlib_canvases[i]
            if canvas:
                try:
                    plt.close(canvas.figure)
                except Exception:
                    pass
                self._ax_sag_cor_matplotlib_canvases[i] = None
        
        super().closeEvent(event)

    def _on_choose_save_folder(self) -> None:
        """Open directory dialog for saving."""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if folder:
            self._ui.save_folder_input.setText(folder)

    def _on_export_voi_clicked(self) -> None:
        """Export the current 3D mask to NIfTI."""
        folder_path = self._ui.save_folder_input.text()
        file_name = self._ui.save_name_input.text()
        
        if not folder_path or not Path(folder_path).is_dir():
            self.show_error("Please select a valid folder.")
            return
        if not file_name:
            self.show_error("Please enter a file name.")
            return

        if not file_name.endswith('.nii.gz'):
            file_name += '.nii.gz'
        
        out_path = Path(folder_path) / file_name
        
        try:
            self.show_loading()
            # Construct affine
            affine = np.eye(4)
            for i, res in enumerate(self._image_data.pixdim[:3]):
                affine[i, i] = res
            
            # The mask is stored in self._seg_data.seg_mask
            mask = self._seg_data.seg_mask
            nii_img = nib.Nifti1Image(mask, affine)
            nii_img.header["descrip"] = self._image_data.scan_name
            nib.save(nii_img, out_path)
            
            self.hide_loading()
            # After export, show decision again
            self._on_back_from_save_clicked()
        except Exception as e:
            self.hide_loading()
            self.show_error(f"Export failed: {str(e)}")

    def _on_slice_slider_changed(self, value: int) -> None:
        """Handle time-series frame change."""
        self._crosshair_xyzt[3] = value
        self._ui.cur_slice_spin_box.blockSignals(True)
        self._ui.cur_slice_spin_box.setValue(value + 1)
        self._ui.cur_slice_spin_box.blockSignals(False)
        self._refresh_frames()

    def _on_toggle_crosshair(self) -> None:
        """Toggle crosshair visibility."""
        self._crosshair_visible = not self._crosshair_visible
        self._ui.toggle_crosshair_visibility_button.setText(
            'Show Crosshair' if not self._crosshair_visible else 'Hide Crosshair'
        )
        self._refresh_frames()

    def _refresh_frames(self) -> None:
        """Mark all planes for refresh."""
        self._ax_sag_cor_pending = [True, True, True]

    def _update_scan_display(self) -> None:
        """Sync UI labels with current crosshair indices. Using 1-based indexing for display."""
        self._ui.ax_frame_num.setText(str(self._crosshair_xyzt[2] + 1))
        self._ui.sag_frame_num.setText(str(self._crosshair_xyzt[0] + 1))
        self._ui.cor_frame_num.setText(str(self._crosshair_xyzt[1] + 1))
        
        # Update spinbox for t
        self._ui.cur_slice_spin_box.blockSignals(True)
        self._ui.cur_slice_spin_box.setValue(self._crosshair_xyzt[3] + 1)
        self._ui.cur_slice_spin_box.blockSignals(False)

    def set_crosshair(self, x=None, y=None, z=None, t=None):
        """Update crosshair position and trigger refresh."""
        changed = False
        if x is not None and 0 <= x < self._x_len:
            self._crosshair_xyzt[0] = x; changed = True
        if y is not None and 0 <= y < self._y_len:
            self._crosshair_xyzt[1] = y; changed = True
        if z is not None and 0 <= z < self._z_len:
            self._crosshair_xyzt[2] = z; changed = True
        if t is not None and 0 <= t < self._num_slices:
            self._crosshair_xyzt[3] = t; changed = True
        
        if changed:
            self._update_scan_display()
            self._refresh_frames()

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
        
        # Match canvas size to the QLabel/placeholder size
        # We ensure it fills the parent label precisely
        canvas_width = label_widget.width()
        canvas_height = label_widget.height()
        canvas.setFixedSize(canvas_width, canvas_height)
        canvas.move(0, 0)
        canvas.draw_idle()

    def _resize_all_canvases(self):
        """Force a resize of all embedded matplotlib canvases."""
        for label in self._ax_sag_cor_planes:
            if label:
                self._resize_canvas_for(label)

    def showEvent(self, event):
        # Ensure canvases sized properly when shown
        self._resize_all_canvases()
        return super().showEvent(event)

    def _connect_matplotlib_events(self):
        """Connect motion and click events on each plane's matplotlib canvas."""
        for plane_ix, canvas in enumerate(self._ax_sag_cor_matplotlib_canvases):
            if not canvas: continue
            canvas.mpl_connect('motion_notify_event', lambda e, p=plane_ix: self._on_canvas_motion(e, p))
            canvas.mpl_connect('button_press_event', lambda e, p=plane_ix: self._on_canvas_click(e, p))

    def _on_canvas_click(self, event, plane_ix: int):
        if event.inaxes is None: return
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
        if self._crosshair_xyzt[dim_x] != new_xval:
            key = ['x','y','z','t'][dim_x]
            params[key] = new_xval
        if self._crosshair_xyzt[dim_y] != new_yval:
            key = ['x','y','z','t'][dim_y]
            params[key] = new_yval

        if params:
            self.set_crosshair(**params)
        
    def _update_hover_crosshair(self, x, y, plane_ix):
        """Update crosshair lines to follow mouse hover."""
        v_line, h_line = self._ax_sag_cor_crosshair_lines[plane_ix]
        if v_line and h_line:
            v_line.set_xdata([x, x])
            h_line.set_ydata([y, y])
            v_line.set_visible(self._crosshair_visible)
            h_line.set_visible(self._crosshair_visible)
            # Only update the background for this ONE canvas
            self._ax_sag_cor_matplotlib_canvases[plane_ix].draw_idle()
