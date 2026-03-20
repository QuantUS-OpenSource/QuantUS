"""
Motion Compensation Review Widget

Shown after motion compensation completes. Lets the user:
  - Inspect per-frame tracking quality (correlation bar chart)
  - Browse frames with the tracked VOI overlay
  - Adjust the search margin and re-run MC
  - Re-anchor: go back and draw a corrected VOI on a specific bad frame
  - Accept the result and proceed
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSlider, QDoubleSpinBox, QFrame, QSizePolicy,
)

from ...mvc.base_view import BaseViewMixin
from engines.ceus.src.data_objs import UltrasoundImage, CeusSeg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_button(text: str, color: str = "rgb(90, 37, 255)") -> QPushButton:
    btn = QPushButton(text)
    btn.setStyleSheet(
        f"QPushButton {{ color: white; font-size: 14px; font-weight: bold; "
        f"background: {color}; border-radius: 10px; padding: 6px 14px; }}"
        f"QPushButton:disabled {{ background: rgb(100,100,100); }}"
    )
    btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return btn


class MotionCompReviewWidget(QWidget, BaseViewMixin):
    """
    Review panel for motion compensation results.

    Signals:
        accepted: user accepts MC and proceeds to analysis
        reanchor_requested(int): user wants to redraw VOI on the given frame index
        rerun_mc_requested(dict): user wants to re-run MC with updated kwargs
        back_requested: go back to seg type selection
        close_requested: close the application
    """

    accepted = pyqtSignal()
    reanchor_requested = pyqtSignal(int)
    rerun_mc_requested = pyqtSignal(dict)
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

        # Pull MC results out of seg_data (set by motion_compensation_function.py)
        mc = getattr(seg_data, 'motion_compensation', None)
        self._mc = mc
        self._correlations: np.ndarray = mc.correlations if mc is not None else np.array([])
        self._tracked_bboxes = mc.tracked_bboxes if mc is not None else []
        self._reference_frame: int = mc.reference_frame if mc is not None else 0
        self._search_margin: float = 0.02  # default shown in spinbox

        self._num_frames = image_data.pixel_data.shape[3]
        self._current_frame = self._reference_frame

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
        left.setSpacing(14)
        left.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Motion Compensation Review")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        left.addWidget(title)

        # Reference frame info
        ref_lbl = QLabel(f"Reference frame: {self._reference_frame}")
        ref_lbl.setStyleSheet("color: #aaa; font-size: 13px;")
        left.addWidget(ref_lbl)

        # Mean correlation
        if len(self._correlations):
            mean_corr = float(np.mean(self._correlations))
            corr_lbl = QLabel(f"Mean correlation: {mean_corr:.3f}")
            corr_lbl.setStyleSheet("color: #aaa; font-size: 13px;")
            left.addWidget(corr_lbl)

        # Frame slider
        slider_lbl = QLabel("Current frame:")
        slider_lbl.setStyleSheet("color: #ccc; font-size: 13px;")
        left.addWidget(slider_lbl)

        slider_row = QHBoxLayout()
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, self._num_frames - 1)
        self._frame_slider.setValue(self._current_frame)
        self._frame_slider.setStyleSheet("QSlider::handle:horizontal { background: #3af; }")
        self._frame_num_lbl = QLabel(str(self._current_frame))
        self._frame_num_lbl.setStyleSheet("color: #3af; font-size: 13px; min-width: 30px;")
        slider_row.addWidget(self._frame_slider)
        slider_row.addWidget(self._frame_num_lbl)
        left.addLayout(slider_row)

        # Per-frame correlation on current frame
        self._frame_corr_lbl = QLabel("")
        self._frame_corr_lbl.setStyleSheet("color: #fa0; font-size: 13px;")
        left.addWidget(self._frame_corr_lbl)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555;")
        left.addWidget(sep)

        # Search margin re-run
        margin_lbl = QLabel("Search margin (re-run MC):")
        margin_lbl.setStyleSheet("color: #ccc; font-size: 13px;")
        left.addWidget(margin_lbl)

        margin_row = QHBoxLayout()
        self._margin_spinbox = QDoubleSpinBox()
        self._margin_spinbox.setRange(0.005, 0.5)
        self._margin_spinbox.setSingleStep(0.005)
        self._margin_spinbox.setDecimals(3)
        self._margin_spinbox.setValue(self._search_margin)
        self._margin_spinbox.setStyleSheet(
            "QDoubleSpinBox { background: #333; color: white; border-radius: 4px; font-size: 13px; }"
        )
        self._margin_spinbox.setMaximumWidth(100)
        margin_row.addWidget(self._margin_spinbox)
        margin_row.addStretch()
        left.addLayout(margin_row)

        self._rerun_btn = _build_button("Re-run MC", "rgb(160, 90, 0)")
        left.addWidget(self._rerun_btn)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #555;")
        left.addWidget(sep2)

        # Re-anchor: draw VOI on the selected frame
        reanchor_info = QLabel(
            "If tracking is bad on the current frame,\n"
            "click Re-anchor to redraw the VOI there."
        )
        reanchor_info.setStyleSheet("color: #aaa; font-size: 12px;")
        reanchor_info.setWordWrap(True)
        left.addWidget(reanchor_info)

        self._reanchor_btn = _build_button("Re-anchor at current frame", "rgb(130, 0, 160)")
        left.addWidget(self._reanchor_btn)

        left.addStretch()

        # Bottom action buttons
        btn_row = QHBoxLayout()
        self._back_btn = _build_button("Back", "rgb(80, 80, 80)")
        self._accept_btn = _build_button("Accept & Continue", "rgb(0, 140, 60)")
        btn_row.addWidget(self._back_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._accept_btn)
        left.addLayout(btn_row)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(320)
        root.addWidget(left_widget)

        # ---- Right column: visualizations ----
        right = QVBoxLayout()
        right.setSpacing(8)

        # Correlation bar chart (top)
        self._corr_fig = Figure(figsize=(6, 2), facecolor='#2a2a2a')
        self._corr_canvas = FigureCanvas(self._corr_fig)
        self._corr_canvas.setMinimumHeight(130)
        self._corr_canvas.setMaximumHeight(180)
        right.addWidget(self._corr_canvas)
        self._build_correlation_chart()

        # Frame display with bounding box overlay (bottom)
        self._frame_fig = Figure(figsize=(6, 5), facecolor='#1a1a1a')
        self._frame_canvas = FigureCanvas(self._frame_fig)
        right.addWidget(self._frame_canvas)

        # The frame ax will be created lazily
        self._frame_ax = self._frame_fig.add_subplot(111)
        self._frame_ax.set_facecolor('#1a1a1a')
        self._frame_ax.tick_params(colors='white')
        for spine in self._frame_ax.spines.values():
            spine.set_edgecolor('#555')
        self._frame_img_artist = None
        self._frame_bbox_rect = None
        self._frame_title_artist = None

        root.addLayout(right)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        self._rerun_btn.clicked.connect(self._on_rerun_clicked)
        self._reanchor_btn.clicked.connect(self._on_reanchor_clicked)
        self._accept_btn.clicked.connect(self.accepted.emit)
        self._back_btn.clicked.connect(self.back_requested.emit)

    # ------------------------------------------------------------------
    # Correlation chart
    # ------------------------------------------------------------------

    def _build_correlation_chart(self) -> None:
        """Draw a bar chart of per-frame correlations."""
        self._corr_fig.clear()
        ax = self._corr_fig.add_subplot(111)
        ax.set_facecolor('#2a2a2a')
        self._corr_fig.patch.set_facecolor('#2a2a2a')

        if len(self._correlations) == 0:
            ax.text(0.5, 0.5, "No MC data", color='white', ha='center', va='center',
                    transform=ax.transAxes)
            self._corr_canvas.draw()
            return

        n = len(self._correlations)
        x = np.arange(n)
        colors = ['#fa4' if c < 0.5 else '#3af' for c in self._correlations]
        colors[self._reference_frame] = '#0f0'  # green for reference
        ax.bar(x, self._correlations, color=colors, width=1.0)

        # Vertical line for current frame
        self._corr_vline = ax.axvline(self._current_frame, color='white', lw=1.5, ls='--')

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frame", color='white', fontsize=9)
        ax.set_ylabel("Corr", color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#555')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3af', label='Good (≥0.5)'),
            Patch(facecolor='#fa4', label='Poor (<0.5)'),
            Patch(facecolor='#0f0', label='Reference'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
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

    def _refresh_frame(self, frame: int) -> None:
        """Update the frame display with the tracked bounding box overlay."""
        pix = self._image_data.pixel_data  # (X, Y, Z, T)
        # Take axial slice (fixed z = middle of volume)
        z_mid = pix.shape[2] // 2
        raw_slice = pix[:, :, z_mid, frame].astype(float)

        # Simple percentile normalisation
        p_lo = np.percentile(raw_slice, 2)
        p_hi = np.percentile(raw_slice, 98)
        denom = p_hi - p_lo if p_hi > p_lo else 1.0
        img_norm = np.clip((raw_slice - p_lo) / denom, 0, 1)
        img_display = img_norm.T  # (Y, X) for imshow

        ax = self._frame_ax
        if self._frame_img_artist is None:
            self._frame_img_artist = ax.imshow(
                img_display, cmap='gray', origin='upper', aspect='auto',
                vmin=0, vmax=1
            )
        else:
            self._frame_img_artist.set_data(img_display)
            self._frame_img_artist.set_extent(
                [-0.5, img_display.shape[1] - 0.5, img_display.shape[0] - 0.5, -0.5]
            )

        # Draw tracked bounding box
        bbox = self._tracked_bboxes[frame] if self._tracked_bboxes else None
        if bbox is not None:
            from matplotlib.patches import Rectangle
            x_min, x_max = bbox.x_min, bbox.x_max
            y_min, y_max = bbox.y_min, bbox.y_max
            rect_x = x_min
            rect_y = y_min
            rect_w = x_max - x_min
            rect_h = y_max - y_min

            if self._frame_bbox_rect is not None:
                self._frame_bbox_rect.remove()
            self._frame_bbox_rect = Rectangle(
                (rect_x, rect_y), rect_w, rect_h,
                linewidth=2, edgecolor='#3af', facecolor='none'
            )
            ax.add_patch(self._frame_bbox_rect)

        # Title
        corr_val = self._correlations[frame] if len(self._correlations) > frame else float('nan')
        corr_str = f"{corr_val:.3f}" if not np.isnan(corr_val) else "N/A"
        quality = "Good" if corr_val >= 0.5 else "Poor"
        quality_color = "#3af" if corr_val >= 0.5 else "#fa4"
        ax.set_title(
            f"Frame {frame} | Corr: {corr_str} ({quality})",
            color=quality_color, fontsize=11, pad=4
        )

        self._frame_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_frame_changed(self, frame: int) -> None:
        self._current_frame = frame
        self._frame_num_lbl.setText(str(frame))

        if len(self._correlations) > frame:
            corr = self._correlations[frame]
            quality = "Good" if corr >= 0.5 else "⚠ Poor"
            self._frame_corr_lbl.setText(f"Correlation: {corr:.3f}  ({quality})")

        self._update_corr_vline(frame)
        self._refresh_frame(frame)

    def _on_rerun_clicked(self) -> None:
        new_margin = self._margin_spinbox.value()
        self._search_margin = new_margin
        kwargs = {
            'reference_frame': self._reference_frame,
            'search_margin_ratio': new_margin,
            'padding': 5,
        }
        self.rerun_mc_requested.emit(kwargs)

    def _on_reanchor_clicked(self) -> None:
        self.reanchor_requested.emit(self._current_frame)

    # ------------------------------------------------------------------
    # BaseViewMixin compatibility
    # ------------------------------------------------------------------

    def show_loading(self) -> None:
        self._accept_btn.setEnabled(False)
        self._rerun_btn.setEnabled(False)
        self._reanchor_btn.setEnabled(False)

    def hide_loading(self) -> None:
        self._accept_btn.setEnabled(True)
        self._rerun_btn.setEnabled(True)
        self._reanchor_btn.setEnabled(True)

    def update_mc_result(self, seg_data: CeusSeg) -> None:
        """Refresh the review widget with a new MC result (after re-run)."""
        self._seg_data = seg_data
        mc = getattr(seg_data, 'motion_compensation', None)
        self._mc = mc
        if mc is not None:
            self._correlations = mc.correlations
            self._tracked_bboxes = mc.tracked_bboxes
            self._reference_frame = mc.reference_frame
        self._build_correlation_chart()
        self._refresh_frame(self._current_frame)
