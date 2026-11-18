"""Power Guided ROI Selection (a segmentation type): correlation lengths on the reference phantom
(page 1) -> first-pass power check with that window + ROI drawing with a live band check (page 2,
DrawROIWithBand). The accepted ROI is saved as a QuantUS ROI pickle and loaded with pkl_roi."""

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QStackedLayout

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.window_roi import POWER_GUIDED_ROI, power_guided_roi_supported  # noqa: F401 (re-export)
from .correlation_page import CorrelationPage
from .roi_band_page import RoiBandPage


class WindowRoiWidget(QWidget, BaseViewMixin):
    roi_accepted = pyqtSignal(dict)   # {'seg_path', 'seg_loader_kwargs', 'roi', 'intersect', 'correlation'}
    back_requested = pyqtSignal()     # from page 1: back to the segmentation type choice
    close_requested = pyqtSignal()
    user_action = pyqtSignal(str, object)

    def __init__(self, image_data, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._stack = QStackedLayout(self)
        self._correlation = CorrelationPage(image_data)
        self._roi = RoiBandPage(image_data)
        self._stack.addWidget(self._correlation)
        self._stack.addWidget(self._roi)

        self._correlation.back_requested.connect(self.back_requested.emit)
        self._correlation.next_requested.connect(self._to_roi_page)
        self._roi.back_requested.connect(lambda: self._stack.setCurrentWidget(self._correlation))
        self._roi.accepted.connect(self._on_accepted)

    def _to_roi_page(self, correlation) -> None:
        w = correlation.recommended_window()
        self._stack.setCurrentWidget(self._roi)
        # saved with the ROI: the Power and Correlation Guided config re-measures inside the ROI with these
        self._roi.correlation_settings = dict(self._correlation.settings(), preset=self._correlation.preset_name())
        self._roi.set_windows(w["ax_window_mm"], w["lat_window_mm"])

    def _on_accepted(self, data: dict) -> None:
        self.roi_accepted.emit(dict(data, seg_loader_kwargs={}, correlation=self._correlation.result))

    def show_roi_page(self) -> None:
        self._stack.setCurrentWidget(self._roi)

    def shutdown(self) -> None:
        """Wait for any running computation before the widget is deleted."""
        self._correlation.shutdown()
        self._roi.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    # BaseViewMixin
    def show_error(self, error_message: str) -> None:
        self._roi.show_error(error_message)

    def hide_loading(self) -> None:
        pass

    def show_loading(self) -> None:
        pass
