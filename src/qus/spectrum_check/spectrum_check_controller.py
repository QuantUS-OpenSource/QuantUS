"""
Spectrum Check Controller: optional step between image loading and segmentation where the
user compares sample and reference-phantom ROI power spectra and picks the analysis band.
"""

from typing import Any

from PyQt6.QtCore import pyqtSignal

from src.qus.mvc.base_controller import BaseController
from engines.qus.quantus.data_objs import UltrasoundRfImage
from .views.spectrum_check_widget import SpectrumCheckWidget


def spectrum_check_supported(image_data: UltrasoundRfImage) -> bool:
    """The check needs 2D (optionally multi-frame) RF data and a 2D phantom frame."""
    rf = getattr(image_data, "rf_data", None)
    ref = getattr(image_data, "phantom_rf_data", None)
    return (getattr(image_data, "spatial_dims", 2) == 2 and rf is not None and rf.ndim in (2, 3)
            and ref is not None and ref.ndim in (2, 3))


class SpectrumCheckController(BaseController):
    """Hands the accepted band (or None when skipped) to the model for the config step."""

    user_action = pyqtSignal(str, object)  # 'spectrum_check_completed', dict | None
    back_requested = pyqtSignal()

    def __init__(self, model, image_data: UltrasoundRfImage):
        view = SpectrumCheckWidget(image_data)
        super().__init__(model, view)
        view.back_requested.connect(self.back_requested.emit)

    def handle_user_action(self, action_name: str, action_data: Any) -> None:
        if action_name == "spectrum_band_accepted":
            self._model.set_spectrum_band(action_data["bandwidth"], action_data["sampling_frequency"])
            self.user_action.emit("spectrum_check_completed", action_data)
        elif action_name == "spectrum_check_skipped":
            self._model.set_spectrum_band(None, None)
            self.user_action.emit("spectrum_check_completed", None)

    def get_widget(self) -> SpectrumCheckWidget:
        return self._view

    def cleanup(self) -> None:
        pass
