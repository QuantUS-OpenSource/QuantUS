"""Power Guided ROI Selection (a segmentation type), on one screen: the probe preset is detected from
the scan, the correlation lengths on the reference phantom give the window size, and the first-pass
power check runs with it automatically, so the user just draws the ROI (DrawROIWithBand) and accepts.
'Adjust window sizes...' opens the correlation-length settings / plots and first-pass options in a
dialog; any change there reruns the first pass. The accepted ROI is saved as a QuantUS ROI pickle
and loaded with pkl_roi.

Window depth range: the correlation lengths are measured over the full image depth, but the window
is the largest recommendation only over the depths where the first pass finds usable signal. The
first pass needs a window, so: first pass with the window over all depths -> usable depths ->
window over those -> first pass again if the window changed (a few rounds at most)."""

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.window_roi import POWER_GUIDED_ROI, power_guided_roi_supported  # noqa: F401 (re-export)
from src.qus.window_roi.probe import detect_probe_preset
from src.qus.window_roi.windows import WINDOW_MAX_DEPTH_MM, usable_depth_range

MAX_WINDOW_ROUNDS = 3   # first pass -> usable depths -> window -> first pass ...
from .common import MUTED
from .correlation_page import CorrelationPage
from .roi_band_page import RoiBandPage


class WindowRoiWidget(QWidget, BaseViewMixin):
    roi_accepted = pyqtSignal(dict)   # {'seg_path', 'seg_loader_kwargs', 'roi', 'intersect', 'correlation'}
    back_requested = pyqtSignal()     # back to the segmentation type choice
    close_requested = pyqtSignal()
    user_action = pyqtSignal(str, object)

    def __init__(self, image_data, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._detected_preset = detect_probe_preset(image_data)
        self._rounds = 0

        self._roi = RoiBandPage(image_data)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._roi)

        self._dialog = self._build_dialog()
        # starts computing right away (with the detected preset, if any)
        self._correlation = CorrelationPage(image_data, preset=self._detected_preset)
        self._correlation.set_dialog_mode()
        self._dialog_layout.insertWidget(0, self._correlation, 1)

        self._correlation.computed.connect(self._apply_correlation)
        self._correlation.failed.connect(self._roi.show_window_error)
        self._roi.adjust_requested.connect(self._open_adjust)
        self._roi.back_requested.connect(self.back_requested.emit)
        self._roi.accepted.connect(self._on_accepted)
        self._roi.first_pass_done.connect(self._on_first_pass)

    def _build_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Window sizes (correlation lengths) & first pass")
        dialog.setModal(True)
        dialog.resize(1350, 800)
        self._dialog_layout = QVBoxLayout(dialog)
        bottom = QHBoxLayout()
        first_pass = QGroupBox("First pass")
        QVBoxLayout(first_pass).addWidget(self._roi.advanced_widget)
        bottom.addWidget(first_pass, 1)
        done = QPushButton("Done")
        done.clicked.connect(dialog.accept)
        bottom.addWidget(done)
        self._dialog_layout.addLayout(bottom)
        return dialog

    def _open_adjust(self) -> None:
        self._dialog.show()
        self._dialog.raise_()

    def _apply_correlation(self, correlation) -> None:
        """New correlation lengths: window over the current usable depths (all depths the first time)."""
        self._rounds = 0
        self._apply_window()

    def _on_first_pass(self, intersect) -> None:
        """Window over the depths this first pass found usable; rerun the first pass if it changed."""
        if self._correlation.result is None or self._rounds >= MAX_WINDOW_ROUNDS:
            return
        depths = usable_depth_range(intersect)
        if depths is None:
            return
        self._rounds += 1
        self._correlation.set_window_depth_range(depths)
        self._apply_window()   # set_windows reruns the first pass only if the window changed

    def _apply_window(self) -> None:
        w = self._correlation.recommended_window()
        preset = self._correlation.preset_name()
        if preset is None:
            source = "custom settings"
        elif preset == self._detected_preset:
            source = f"{preset} preset, detected from the probe"
        else:
            source = f"{preset} preset"
        lim = w.get("limited_to_mm")
        cap = WINDOW_MAX_DEPTH_MM.get(preset)
        if lim and self._correlation.window_depths is not None:
            depths = f"max over the usable depths {max(lim[0], 0):.0f}-{lim[1]:.0f} mm"
        elif lim:
            depths = f"max down to {lim[1]:.0f} mm; narrowing to the usable depths after the first pass..."
        else:
            depths = "all depths; narrowing to the usable depths after the first pass..."
        if cap is not None:
            depths += f", {preset} limit {cap:g} mm"
        self._roi.set_window_summary(
            f"<b>Window</b>: {w['ax_window_mm']:.1f} mm axial × {w['lat_window_mm']:.1f} mm lateral<br>"
            f"<span style='color:{MUTED}'>From correlation lengths on the reference phantom ({source}; "
            f"{depths})</span>")
        # saved with the ROI: the Power and Correlation Guided config re-measures inside the ROI with these
        self._roi.correlation_settings = dict(self._correlation.settings(), preset=preset)
        self._roi.set_windows(w["ax_window_mm"], w["lat_window_mm"])

    def _on_accepted(self, data: dict) -> None:
        self.roi_accepted.emit(dict(data, seg_loader_kwargs={}, correlation=self._correlation.result))

    def show_roi_page(self) -> None:
        pass  # single screen

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
