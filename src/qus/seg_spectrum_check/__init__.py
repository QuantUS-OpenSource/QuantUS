"""
Spectrum check seeded from the confirmed segmentation.

When the user turns on "Include spectrum check after segmentation", the spectrum check runs after
segmentation. This wraps Yuanshan's Spectrum Check (src/qus/spectrum_check, unchanged) so its
sample ROI starts as the bounding box of the segmentation (on the RF grid, at the segmentation's
frame), and the reference ROI starts at the same rows / lines (same depth in the phantom).
"Reset ROIs" returns to these instead of the central default. The user can still drag either ROI.
"""

from typing import Optional, Tuple

import numpy as np

from src.qus.mvc.base_controller import BaseController
from src.qus.spectrum_check.spectrum_check_controller import SpectrumCheckController
from src.qus.spectrum_check.views.spectrum_check_widget import SpectrumCheckWidget
from engines.qus.quantus.data_objs import BmodeSeg, UltrasoundRfImage
from engines.qus.quantus.seg_loading.power_spectrum.spectrum import default_roi

Roi = Tuple[int, int, int, int]  # r0, r1, c0, c1 (half-open), as in quantus.seg_loading.power_spectrum


def seg_bounding_roi(seg_data: Optional[BmodeSeg], frame_shape: Tuple[int, int]) -> Optional[Roi]:
    """Bounding box (r0, r1, c0, c1; half-open) of a 2D segmentation mask on the RF grid, or None
    when there is no usable 2D mask matching the RF frame."""
    mask = getattr(seg_data, "seg_mask", None)
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != tuple(frame_shape[:2]):
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size < 2 or cols.size < 1:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _clip_roi(roi: Roi, shape: Tuple[int, int]) -> Optional[Roi]:
    r0, r1, c0, c1 = roi
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, shape[0]), min(c1, shape[1])
    return (r0, r1, c0, c1) if r1 - r0 >= 2 and c1 - c0 >= 1 else None


class SegSpectrumCheckWidget(SpectrumCheckWidget):
    """SpectrumCheckWidget whose ROIs start from the segmentation instead of the central default."""

    def __init__(self, image_data: UltrasoundRfImage, seg_data: Optional[BmodeSeg], parent=None):
        super().__init__(image_data, parent)
        self._seg_frame = self._valid_frame(getattr(seg_data, "frame", 0))
        rf = image_data.rf_data
        frame_shape = rf.shape[1:] if rf.ndim == 3 else rf.shape
        self._seg_roi = seg_bounding_roi(seg_data, frame_shape)
        self._seg_reference_roi = _clip_roi(self._seg_roi, self._reference_rf.shape) if self._seg_roi else None
        if self._seg_roi is None:
            return  # no usable 2D mask: keep the widget's defaults

        self._reset_button.setText("Reset ROIs to segmentation")
        self._frame = self._seg_frame
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(self._frame)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText(f"{self._frame + 1} / {self._n_frames}")
        self._apply_seg_rois()

    def _valid_frame(self, frame) -> int:
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            frame = 0
        return frame if 0 <= frame < self._n_frames else 0

    def _apply_seg_rois(self) -> None:
        self._sample_roi = self._seg_roi
        self._reference_roi = self._seg_reference_roi or default_roi(self._reference_rf.shape)
        self._draw_images()  # redraws the current frame and recreates both selectors at these ROIs
        self._update_spectrum()

    def _on_reset_rois(self) -> None:
        if self._seg_roi is None:
            super()._on_reset_rois()
        else:
            self._apply_seg_rois()


class SegSpectrumCheckController(SpectrumCheckController):
    """SpectrumCheckController with the segmentation-seeded widget; same actions and model updates."""

    def __init__(self, model, image_data: UltrasoundRfImage, seg_data: Optional[BmodeSeg]):
        view = SegSpectrumCheckWidget(image_data, seg_data)
        BaseController.__init__(self, model, view)  # skip the parent's __init__, which builds its own widget
        view.back_requested.connect(self.back_requested.emit)
