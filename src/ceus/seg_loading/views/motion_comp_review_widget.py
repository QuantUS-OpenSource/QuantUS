"""
Motion Compensation Review Widget

Shown after motion compensation completes. Lets the user:
  - Browse each frame with the MC-shifted VOI overlay + tracked bounding box
  - Inspect the per-frame correlation bar chart
  - Adjust the search margin and re-run MC on the full video
  - Re-anchor: pick a frame where tracking went wrong, redraw VOI there,
    and MC is re-run from that frame forward (frames before are kept from
    the original run)
  - Accept and proceed to analysis
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import shift as ndimage_shift
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSlider, QDoubleSpinBox, QFrame, QSizePolicy,
)

from ...mvc.base_view import BaseViewMixin
from engines.ceus.src.data_objs import UltrasoundImage, CeusSeg


def _build_button(text: str, color: str = "rgb(90, 37, 255)") -> QPushButton:
    btn = QPushButton(text)
    btn.setStyleSheet(
        f"QPushButton {{ color: white; font-size: 14px; font-weight: bold; "
        f"background: {color}; border-radius: 10px; padding: 6px 14px; }}"
        f"QPushButton:disabled {{ background: rgb(80,80,80); color: #888; }}"
    )
    btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return btn


class MotionCompReviewWidget(QWidget, BaseViewMixin):
    """
    Review panel for motion compensation results.

    Displays the MC-shifted VOI and tracked bounding box per frame so the user
    can verify quality.  Bad frames can be re-anchored: the user is sent back to
    DrawVOIWidget to redraw the VOI at that frame; MC then re-runs from that
    frame forward while keeping earlier results intact.

    Signals
    -------
    accepted
        User is satisfied and wants to proceed to analysis.
    reanchor_requested(int)
        User wants to redraw the VOI at the given frame index and re-run MC
        from that frame onward.
    rerun_mc_requested(dict)
        User wants to redo the *entire* MC with updated kwargs
        (e.g. different search margin).
    back_requested
        Return to seg type selection.
    close_requested
        Close the application.
    """

    accepted = pyqtSignal()
    reanchor_requested = pyqtSignal(int)   # frame index to re-anchor from
    rerun_mc_requested = pyqtSignal(dict)  # full re-run with new kwargs
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        seg_data: CeusSeg,
        image_data: UltrasoundImage,
        bmode_image_data: Optional[UltrasoundImage],
        parent: Optional[QWidget] = None,
    ):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self.setStyleSheet("QWidget { background: rgb(42, 42, 42); }")

        self._seg_data = seg_data
        self._image_data = image_data
        self._bmode_image_data = bmode_image_data

        mc = getattr(seg_data, 'motion_compensation', None)
        self._mc = mc
        self._voi_mask: Optional[np.ndarray] = getattr(seg_data, 'seg_mask', None)
        self._correlations: np.ndarray = (
            mc.correlations if mc is not None else np.array([])
        )
        self._tracked_bboxes = mc.tracked_bboxes if mc is not None else []
        self._reference_frame: int = mc.reference_frame if mc is not None else 0

        self._num_frames = image_data.pixel_data.shape[3]
        self._current_frame = self._reference_frame

        # Cache for shifted-mask slices (cleared on MC update)
        self._shifted_slice_cache: dict = {}

        self._setup_ui()
        self._connect_signals()
        self._refresh_frame(self._current_frame)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ---- Left column: controls ----
        left = QVBoxLayout()
        left.setSpacing(10)
        left.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Motion Compensation Review")
        title.setStyleSheet("color: white; font-size: 17px; font-weight: bold;")
        left.addWidget(title)

        ref_lbl = QLabel(f"Reference frame: {self._reference_frame}")
        ref_lbl.setStyleSheet("color: #aaa; font-size: 13px;")
        left.addWidget(ref_lbl)
        self._ref_lbl = ref_lbl

        if len(self._correlations):
            mean_corr = float(np.mean(self._correlations))
            self._mean_corr_lbl = QLabel(f"Mean correlation: {mean_corr:.3f}")
            self._mean_corr_lbl.setStyleSheet("color: #aaa; font-size: 13px;")
            left.addWidget(self._mean_corr_lbl)

        # Frame slider
        slider_lbl = QLabel("Browse frame:")
        slider_lbl.setStyleSheet("color: #ccc; font-size: 13px;")
        left.addWidget(slider_lbl)

        slider_row = QHBoxLayout()
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, self._num_frames - 1)
        self._frame_slider.setValue(self._current_frame)
        self._frame_slider.setStyleSheet(
            "QSlider::handle:horizontal { background: #3af; border-radius: 5px; }"
        )
        self._frame_num_lbl = QLabel(str(self._current_frame))
        self._frame_num_lbl.setStyleSheet("color: #3af; font-size: 13px; min-width: 30px;")
        slider_row.addWidget(self._frame_slider)
        slider_row.addWidget(self._frame_num_lbl)
        left.addLayout(slider_row)

        # Per-frame correlation info
        self._frame_corr_lbl = QLabel("")
        self._frame_corr_lbl.setStyleSheet("color: #fa0; font-size: 13px;")
        left.addWidget(self._frame_corr_lbl)
        self._update_frame_corr_label(self._current_frame)

        left.addWidget(self._make_separator())

        # ---- Re-anchor section ----
        reanchor_title = QLabel("Fix a bad frame")
        reanchor_title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        left.addWidget(reanchor_title)

        reanchor_info = QLabel(
            "Navigate to a frame with bad tracking.\n"
            "Click Re-anchor to redraw the VOI there.\n"
            "MC will re-run from that frame forward,\n"
            "keeping earlier results intact."
        )
        reanchor_info.setStyleSheet("color: #aaa; font-size: 12px;")
        reanchor_info.setWordWrap(True)
        left.addWidget(reanchor_info)

        self._reanchor_btn = _build_button(
            "Re-anchor at current frame", "rgb(130, 0, 160)"
        )
        left.addWidget(self._reanchor_btn)

        left.addWidget(self._make_separator())

        # ---- Full re-run section ----
        rerun_title = QLabel("Full re-run")
        rerun_title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        left.addWidget(rerun_title)

        margin_row = QHBoxLayout()
        margin_lbl = QLabel("Search margin:")
        margin_lbl.setStyleSheet("color: #ccc; font-size: 13px;")
        self._margin_spinbox = QDoubleSpinBox()
        self._margin_spinbox.setRange(0.005, 0.5)
        self._margin_spinbox.setSingleStep(0.005)
        self._margin_spinbox.setDecimals(3)
        self._margin_spinbox.setValue(0.02)
        self._margin_spinbox.setStyleSheet(
            "QDoubleSpinBox { background: #333; color: white; "
            "border-radius: 4px; font-size: 13px; }"
        )
        self._margin_spinbox.setMaximumWidth(95)
        margin_row.addWidget(margin_lbl)
        margin_row.addStretch()
        margin_row.addWidget(self._margin_spinbox)
        left.addLayout(margin_row)

        self._rerun_btn = _build_button("Re-run MC (full)", "rgb(160, 90, 0)")
        left.addWidget(self._rerun_btn)

        left.addStretch()

        # Bottom navigation
        btn_row = QHBoxLayout()
        self._back_btn = _build_button("Back", "rgb(80, 80, 80)")
        self._accept_btn = _build_button("Accept & Proceed", "rgb(0, 140, 60)")
        btn_row.addWidget(self._back_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._accept_btn)
        left.addLayout(btn_row)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(320)
        root.addWidget(left_widget)

        # ---- Right column: correlation chart + frame view ----
        right = QVBoxLayout()
        right.setSpacing(8)

        self._corr_fig = Figure(figsize=(6, 2), facecolor='#2a2a2a')
        self._corr_canvas = FigureCanvas(self._corr_fig)
        self._corr_canvas.setMinimumHeight(130)
        self._corr_canvas.setMaximumHeight(180)
        right.addWidget(self._corr_canvas)
        self._build_correlation_chart()

        self._frame_fig = Figure(figsize=(6, 5), facecolor='#1a1a1a')
        self._frame_canvas = FigureCanvas(self._frame_fig)
        right.addWidget(self._frame_canvas)

        self._frame_ax = self._frame_fig.add_subplot(111)
        self._frame_ax.set_facecolor('#1a1a1a')
        self._frame_ax.tick_params(colors='white')
        for spine in self._frame_ax.spines.values():
            spine.set_edgecolor('#555')
        self._frame_img_artist = None
        self._frame_voi_artist = None
        self._frame_bbox_rect = None

        root.addLayout(right)

    def _make_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555; max-height: 1px;")
        return sep

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        self._reanchor_btn.clicked.connect(self._on_reanchor_clicked)
        self._rerun_btn.clicked.connect(self._on_rerun_clicked)
        self._accept_btn.clicked.connect(self.accepted.emit)
        self._back_btn.clicked.connect(self.back_requested.emit)

    # ------------------------------------------------------------------
    # Correlation chart
    # ------------------------------------------------------------------

    def _build_correlation_chart(self) -> None:
        self._corr_fig.clear()
        ax = self._corr_fig.add_subplot(111)
        ax.set_facecolor('#2a2a2a')
        self._corr_fig.patch.set_facecolor('#2a2a2a')

        if len(self._correlations) == 0:
            ax.text(0.5, 0.5, "No MC data", color='white', ha='center',
                    va='center', transform=ax.transAxes)
            self._corr_canvas.draw()
            return

        n = len(self._correlations)
        colors = ['#fa4' if c < 0.5 else '#3af' for c in self._correlations]
        colors[self._reference_frame % n] = '#0f0'
        ax.bar(np.arange(n), self._correlations, color=colors, width=1.0)

        self._corr_vline = ax.axvline(self._current_frame, color='white', lw=1.5, ls='--')

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frame", color='white', fontsize=9)
        ax.set_ylabel("Corr", color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#555')

        legend_els = [
            mpatches.Patch(facecolor='#3af', label='Good (≥0.5)'),
            mpatches.Patch(facecolor='#fa4', label='Poor (<0.5)'),
            mpatches.Patch(facecolor='#0f0', label='Reference'),
        ]
        ax.legend(handles=legend_els, loc='lower right', fontsize=7,
                  facecolor='#333', labelcolor='white', framealpha=0.7)

        self._corr_ax = ax
        self._corr_fig.tight_layout(pad=0.5)
        self._corr_canvas.draw()

    def _update_corr_vline(self, frame: int) -> None:
        if hasattr(self, '_corr_vline') and self._corr_vline is not None:
            self._corr_vline.set_xdata([frame, frame])
            self._corr_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _get_shifted_mask_slice(self, frame: int) -> Optional[np.ndarray]:
        """Return the axial (X-Y) slice of the VOI mask shifted to frame T."""
        if self._mc is None or self._voi_mask is None:
            return None
        if frame in self._shifted_slice_cache:
            return self._shifted_slice_cache[frame]

        tx, ty, tz = self._mc.get_translation(frame)
        z_mid = self._voi_mask.shape[2] // 2

        # Shift the full 3D mask and take axial slice
        shifted = ndimage_shift(
            self._voi_mask.astype(float), shift=[tx, ty, tz], order=0, cval=0
        )
        mask_slice = shifted[:, :, z_mid]  # (X, Y)
        self._shifted_slice_cache[frame] = mask_slice
        return mask_slice

    def _refresh_frame(self, frame: int) -> None:
        pix = self._image_data.pixel_data       # (X, Y, Z, T)
        z_mid = pix.shape[2] // 2
        raw = pix[:, :, z_mid, frame].astype(float)

        p_lo = np.percentile(raw, 2)
        p_hi = np.percentile(raw, 98)
        denom = p_hi - p_lo if p_hi > p_lo else 1.0
        img = np.clip((raw - p_lo) / denom, 0, 1).T   # (Y, X) for imshow

        ax = self._frame_ax
        if self._frame_img_artist is None:
            self._frame_img_artist = ax.imshow(
                img, cmap='gray', origin='upper', aspect='auto', vmin=0, vmax=1
            )
        else:
            self._frame_img_artist.set_data(img)
            self._frame_img_artist.set_extent(
                [-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5]
            )

        # Shifted VOI mask overlay (semi-transparent red)
        mask_slice = self._get_shifted_mask_slice(frame)
        if mask_slice is not None:
            mask_rgba = np.zeros((*mask_slice.T.shape, 4), dtype=float)
            mask_rgba[..., 0] = 1.0   # red
            mask_rgba[..., 3] = np.clip(mask_slice.T * 0.45, 0, 0.45)  # alpha

            if self._frame_voi_artist is None:
                self._frame_voi_artist = ax.imshow(
                    mask_rgba, origin='upper', aspect='auto',
                    extent=[-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5]
                )
            else:
                self._frame_voi_artist.set_data(mask_rgba)
                self._frame_voi_artist.set_extent(
                    [-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5]
                )

        # Tracked bounding box (cyan outline)
        bbox = self._tracked_bboxes[frame] if self._tracked_bboxes else None
        if bbox is not None:
            if self._frame_bbox_rect is not None:
                self._frame_bbox_rect.remove()
            self._frame_bbox_rect = mpatches.Rectangle(
                (bbox.x_min, bbox.y_min),
                bbox.x_max - bbox.x_min,
                bbox.y_max - bbox.y_min,
                linewidth=2, edgecolor='#3af', facecolor='none',
            )
            ax.add_patch(self._frame_bbox_rect)

        # Title with quality indicator
        corr_val = (self._correlations[frame]
                    if len(self._correlations) > frame else float('nan'))
        corr_str = f"{corr_val:.3f}" if not np.isnan(corr_val) else "N/A"
        quality = "Good" if corr_val >= 0.5 else "⚠ Poor"
        color = "#3af" if corr_val >= 0.5 else "#fa4"
        ref_marker = "  ← ref" if frame == self._reference_frame else ""
        ax.set_title(
            f"Frame {frame}{ref_marker}  |  Corr: {corr_str}  ({quality})",
            color=color, fontsize=11, pad=4,
        )
        self._frame_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _update_frame_corr_label(self, frame: int) -> None:
        if len(self._correlations) > frame:
            corr = self._correlations[frame]
            quality = "Good" if corr >= 0.5 else "⚠ Poor"
            self._frame_corr_lbl.setText(f"Correlation: {corr:.3f}  ({quality})")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_frame_changed(self, frame: int) -> None:
        self._current_frame = frame
        self._frame_num_lbl.setText(str(frame))
        self._update_frame_corr_label(frame)
        self._update_corr_vline(frame)
        self._refresh_frame(frame)

    def _on_reanchor_clicked(self) -> None:
        """Send user back to draw a new VOI at the current bad frame."""
        self.reanchor_requested.emit(self._current_frame)

    def _on_rerun_clicked(self) -> None:
        """Re-run MC on the full video with the current kwargs."""
        kwargs = {
            'reference_frame': self._reference_frame,
            'search_margin_ratio': self._margin_spinbox.value(),
            'padding': 5,
        }
        self.rerun_mc_requested.emit(kwargs)

    # ------------------------------------------------------------------
    # Public update API (called after re-run / re-anchor MC completes)
    # ------------------------------------------------------------------

    def update_mc_result(self, seg_data: CeusSeg) -> None:
        """Refresh all displays with a new MC result (re-run or re-anchor)."""
        self._seg_data = seg_data
        self._voi_mask = getattr(seg_data, 'seg_mask', None)
        mc = getattr(seg_data, 'motion_compensation', None)
        self._mc = mc
        if mc is not None:
            self._correlations = mc.correlations
            self._tracked_bboxes = mc.tracked_bboxes
            self._reference_frame = mc.reference_frame
            if hasattr(self, '_ref_lbl'):
                self._ref_lbl.setText(f"Reference frame: {self._reference_frame}")
            if hasattr(self, '_mean_corr_lbl'):
                self._mean_corr_lbl.setText(
                    f"Mean correlation: {float(np.mean(mc.correlations)):.3f}"
                )

        # Clear slice cache so displays recompute
        self._shifted_slice_cache.clear()
        self._frame_img_artist = None
        self._frame_voi_artist = None
        self._frame_bbox_rect = None
        self._frame_ax.cla()
        self._frame_ax.set_facecolor('#1a1a1a')
        self._frame_ax.tick_params(colors='white')
        for spine in self._frame_ax.spines.values():
            spine.set_edgecolor('#555')

        self._build_correlation_chart()
        self._refresh_frame(self._current_frame)

    # ------------------------------------------------------------------
    # BaseViewMixin overrides
    # ------------------------------------------------------------------

    def show_loading(self) -> None:
        self._accept_btn.setEnabled(False)
        self._rerun_btn.setEnabled(False)
        self._reanchor_btn.setEnabled(False)

    def hide_loading(self) -> None:
        self._accept_btn.setEnabled(True)
        self._rerun_btn.setEnabled(True)
        self._reanchor_btn.setEnabled(True)
