"""
Spectrum Check Widget: draw an ROI on the sample and on the reference phantom, compare their
power spectra, and pick the analysis frequency band before segmentation.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from scipy.signal import hilbert

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QDoubleSpinBox, QPushButton, QFormLayout, QFrame,
)

from src.qus.mvc.base_view import BaseViewMixin
from engines.qus.quantus.data_objs import UltrasoundRfImage
from engines.qus.quantus.power_spectrum.spectrum import compare_spectra, default_roi, DISPLAY_FLOOR_DB

BG = (42 / 255, 42 / 255, 42 / 255)
DISPLAY_DYNAMIC_RANGE_DB = 50.0
SAMPLE_COLOR = "#4fa3ff"
REFERENCE_COLOR = "#ffb347"


def _rf_to_display(rf: np.ndarray) -> np.ndarray:
    """Log-compressed envelope (dB re max, clipped to the display dynamic range) of a 2D RF frame."""
    with np.errstate(divide="ignore"):
        env = 20 * np.log10(np.abs(hilbert(np.asarray(rf, dtype=np.float64), axis=0)))
    finite = env[np.isfinite(env)]
    vmax = finite.max() if finite.size else 0.0
    return np.clip(np.nan_to_num(env, neginf=vmax - DISPLAY_DYNAMIC_RANGE_DB), vmax - DISPLAY_DYNAMIC_RANGE_DB, vmax)


def _style_axes(ax) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


class SpectrumCheckWidget(QWidget, BaseViewMixin):
    """
    Sample / reference ROI power-spectrum comparison (see quantus.power_spectrum).

    ROIs are drawn on the pre-scan-conversion B-mode, so for sector probes the rectangle is a
    beam x depth region, which is the region the spectrum is actually computed over.
    """

    user_action = pyqtSignal(str, object)
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, image_data: UltrasoundRfImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._image_data = image_data

        rf = image_data.rf_data
        self._n_frames = rf.shape[0] if rf.ndim == 3 else 1
        self._frame = 0
        ref = image_data.phantom_rf_data
        self._reference_rf = ref[0] if ref.ndim == 3 else ref

        self._sample_roi = default_roi(self._sample_frame().shape)
        self._reference_roi = default_roi(self._reference_rf.shape)
        self._result = None

        self._setup_ui()
        self._connect_signals()
        self._draw_images()
        self._update_spectrum()

    # ------------------------------------------------------------------ data helpers
    def _sample_frame(self) -> np.ndarray:
        rf = self._image_data.rf_data
        return rf[self._frame] if rf.ndim == 3 else rf

    def _aspect(self):
        ax_res = getattr(self._image_data, "axial_res", None)
        lat_res = getattr(self._image_data, "lateral_res", None)
        try:
            if ax_res and lat_res and np.isfinite(ax_res) and np.isfinite(lat_res):
                return float(ax_res) / float(lat_res)
        except TypeError:
            pass
        return "auto"

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        # Sidebar
        side = QVBoxLayout()
        side_frame = QFrame()
        side_frame.setLayout(side)
        side_frame.setFixedWidth(330)

        title = QLabel("Spectrum Check")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        side.addWidget(title)

        info = QLabel(f"Image: {self._image_data.scan_name}\nPhantom: {self._image_data.phantom_name}")
        info.setWordWrap(True)
        side.addWidget(info)

        hint = QLabel("Drag a rectangle on each image to set the sample and reference ROIs. "
                      "The analysis band is where both spectra stay above the cutoff.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgb(190, 190, 190);")
        side.addWidget(hint)

        form = QFormLayout()
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, self._n_frames - 1)
        self._frame_label = QLabel(f"1 / {self._n_frames}")
        frame_row = QHBoxLayout()
        frame_row.addWidget(self._frame_slider)
        frame_row.addWidget(self._frame_label)
        if self._n_frames > 1:
            form.addRow("Sample frame:", frame_row)

        self._fs_spin = QDoubleSpinBox()
        self._fs_spin.setRange(0.0, 5000.0)
        self._fs_spin.setDecimals(3)
        self._fs_spin.setSuffix(" MHz")
        fs = getattr(self._image_data, "sampling_frequency", None)
        self._fs_spin.setValue(fs / 1e6 if fs else 0.0)
        self._fs_spin.setKeyboardTracking(False)
        form.addRow("Sampling frequency:", self._fs_spin)

        self._cutoff_spin = QDoubleSpinBox()
        self._cutoff_spin.setRange(-60.0, -1.0)
        self._cutoff_spin.setDecimals(1)
        self._cutoff_spin.setSingleStep(1.0)
        self._cutoff_spin.setValue(-10.0)
        self._cutoff_spin.setSuffix(" dB")
        self._cutoff_spin.setKeyboardTracking(False)
        form.addRow("Bandwidth cutoff:", self._cutoff_spin)
        side.addLayout(form)

        self._reset_button = QPushButton("Reset ROIs to default")
        side.addWidget(self._reset_button)

        self._results_label = QLabel()
        self._results_label.setWordWrap(True)
        self._results_label.setTextFormat(Qt.TextFormat.RichText)
        self._results_label.setStyleSheet("font-size: 14px; padding: 6px; border: 1px solid rgb(80, 80, 80);")
        side.addWidget(self._results_label)

        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: rgb(255, 110, 110);")
        self._error_label.hide()
        side.addWidget(self._error_label)

        side.addStretch()
        self._accept_button = QPushButton("Use band && continue")
        self._accept_button.setToolTip("Use this band as the analysis band and center frequency in the config step")
        self._skip_button = QPushButton("Skip")
        self._skip_button.setToolTip("Continue without setting the band from the spectrum")
        self._back_button = QPushButton("Back")
        side.addWidget(self._accept_button)
        side.addWidget(self._skip_button)
        side.addWidget(self._back_button)
        root.addWidget(side_frame)

        # Main area: two B-mode canvases on top, spectrum below
        main = QVBoxLayout()
        images_row = QHBoxLayout()
        self._sample_fig = Figure(facecolor=BG)
        self._sample_canvas = FigureCanvas(self._sample_fig)
        self._reference_fig = Figure(facecolor=BG)
        self._reference_canvas = FigureCanvas(self._reference_fig)
        images_row.addWidget(self._sample_canvas)
        images_row.addWidget(self._reference_canvas)
        main.addLayout(images_row, 3)

        self._spectrum_fig = Figure(facecolor=BG)
        self._spectrum_canvas = FigureCanvas(self._spectrum_fig)
        main.addWidget(self._spectrum_canvas, 2)
        root.addLayout(main, 1)

        self._sample_ax = self._sample_fig.add_subplot(111)
        self._reference_ax = self._reference_fig.add_subplot(111)
        self._spectrum_ax = self._spectrum_fig.add_subplot(111)
        for fig in (self._sample_fig, self._reference_fig):
            fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.1)
        self._spectrum_fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.18)

    def _connect_signals(self) -> None:
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        self._fs_spin.valueChanged.connect(lambda _: self._update_spectrum())
        self._cutoff_spin.valueChanged.connect(lambda _: self._update_spectrum())
        self._reset_button.clicked.connect(self._on_reset_rois)
        self._accept_button.clicked.connect(self._on_accept)
        self._skip_button.clicked.connect(lambda: self.user_action.emit("spectrum_check_skipped", None))
        self._back_button.clicked.connect(self.back_requested.emit)

    # ------------------------------------------------------------------ drawing
    def _make_selector(self, ax, roi, color, on_select) -> RectangleSelector:
        selector = RectangleSelector(
            ax, on_select, useblit=True, button=[1], minspanx=2, minspany=4, spancoords="data",
            interactive=True, props=dict(edgecolor=color, facecolor="none", linewidth=2),
        )
        selector.extents = self._roi_to_extents(roi)
        return selector

    @staticmethod
    def _roi_to_extents(roi) -> Tuple[float, float, float, float]:
        r0, r1, c0, c1 = roi
        return (c0 - 0.5, c1 - 0.5, r0 - 0.5, r1 - 0.5)

    @staticmethod
    def _extents_to_roi(extents, shape) -> Tuple[int, int, int, int]:
        x0, x1, y0, y1 = extents
        r0 = int(np.clip(np.round(y0 + 0.5), 0, shape[0] - 1))
        r1 = int(np.clip(np.round(y1 + 0.5), r0 + 1, shape[0]))
        c0 = int(np.clip(np.round(x0 + 0.5), 0, shape[1] - 1))
        c1 = int(np.clip(np.round(x1 + 0.5), c0 + 1, shape[1]))
        return r0, r1, c0, c1

    def _draw_images(self) -> None:
        aspect = self._aspect()
        for ax, rf, title in ((self._sample_ax, self._sample_frame(), "Sample"),
                              (self._reference_ax, self._reference_rf, "Reference phantom")):
            ax.clear()
            img = ax.imshow(_rf_to_display(rf), cmap="gray", aspect=aspect, interpolation="nearest")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Line")
            ax.set_ylabel("Sample")
            _style_axes(ax)
            if ax is self._sample_ax:
                self._sample_img = img
        self._sample_selector = self._make_selector(self._sample_ax, self._sample_roi, SAMPLE_COLOR, self._on_sample_roi)
        self._reference_selector = self._make_selector(self._reference_ax, self._reference_roi, REFERENCE_COLOR,
                                                       self._on_reference_roi)
        self._sample_canvas.draw_idle()
        self._reference_canvas.draw_idle()

    def _update_spectrum(self) -> None:
        ax = self._spectrum_ax
        ax.clear()
        _style_axes(ax)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Power [dB]")
        fs = self._fs_spin.value() * 1e6
        self._result = None
        if fs <= 0:
            self._show_results("Enter the sampling frequency to compute the spectra.")
            self._spectrum_canvas.draw_idle()
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._result = compare_spectra(self._sample_frame(), self._reference_rf, fs,
                                               self._sample_roi, self._reference_roi,
                                               cutoff_db=self._cutoff_spin.value())
        except Exception as e:  # e.g. a degenerate ROI
            self._show_results(f"Could not compute spectra: {e}")
            self._spectrum_canvas.draw_idle()
            return

        r, bw = self._result, self._result.bandwidth
        f = r.freqs / 1e6
        ax.plot(f, r.sample_db, color=SAMPLE_COLOR, label="Sample")
        ax.plot(f, r.reference_db, color=REFERENCE_COLOR, label="Reference")
        ax.axhline(bw.cutoff_db, color="white", linestyle=":", linewidth=1)
        if bw.is_valid:
            ax.axvspan(bw.f_min / 1e6, bw.f_max / 1e6, color="#5fd35f", alpha=0.18, label="Analysis band")
            ax.axvline(bw.f_center / 1e6, color="#5fd35f", linewidth=1.5)
        ax.set_ylim(min(DISPLAY_FLOOR_DB, bw.cutoff_db - 5), 0.5)
        ax.set_xlim(f[0], f[-1])
        ax.grid(True, color=(0.4, 0.4, 0.4), linewidth=0.5)
        legend = ax.legend(loc="lower center", fontsize=8, facecolor=BG, edgecolor="white")
        for text in legend.get_texts():
            text.set_color("white")
        self._spectrum_canvas.draw_idle()

        if bw.is_valid:
            self._show_results(
                f"<b>Analysis band</b> ({bw.cutoff_db:.0f} dB): "
                f"{bw.f_min / 1e6:.2f} – {bw.f_max / 1e6:.2f} MHz<br>"
                f"<b>Center frequency</b>: {bw.f_center / 1e6:.2f} MHz<br>"
                f"Sample peak: {bw.f_peak_sample / 1e6:.2f} MHz<br>"
                f"Reference peak: {bw.f_peak_reference / 1e6:.2f} MHz")
        else:
            self._show_results(f"Sample and reference spectra do not overlap at {bw.cutoff_db:.0f} dB. "
                               "Try a lower cutoff or different ROIs.")

    def _show_results(self, html: str) -> None:
        self._results_label.setText(html)
        self._accept_button.setEnabled(self._result is not None and self._result.bandwidth.is_valid)

    # ------------------------------------------------------------------ handlers
    def _on_sample_roi(self, _press, _release) -> None:
        self._sample_roi = self._extents_to_roi(self._sample_selector.extents, self._sample_frame().shape)
        self._update_spectrum()

    def _on_reference_roi(self, _press, _release) -> None:
        self._reference_roi = self._extents_to_roi(self._reference_selector.extents, self._reference_rf.shape)
        self._update_spectrum()

    def _on_frame_changed(self, value: int) -> None:
        self._frame = int(value)
        self._frame_label.setText(f"{self._frame + 1} / {self._n_frames}")
        self._sample_img.set_data(_rf_to_display(self._sample_frame()))
        self._sample_img.autoscale()
        self._sample_canvas.draw_idle()
        self._update_spectrum()

    def _on_reset_rois(self) -> None:
        self._sample_roi = default_roi(self._sample_frame().shape)
        self._reference_roi = default_roi(self._reference_rf.shape)
        self._sample_selector.extents = self._roi_to_extents(self._sample_roi)
        self._reference_selector.extents = self._roi_to_extents(self._reference_roi)
        self._sample_canvas.draw_idle()
        self._reference_canvas.draw_idle()
        self._update_spectrum()

    def _on_accept(self) -> None:
        if self._result is None or not self._result.bandwidth.is_valid:
            self.show_error("No usable band to apply.")
            return
        self.user_action.emit("spectrum_band_accepted", {
            "bandwidth": self._result.bandwidth,
            "sampling_frequency": self._fs_spin.value() * 1e6,
            "sample_roi": self._result.sample_roi,
            "reference_roi": self._result.reference_roi,
            "frame": self._frame,
        })

    # ------------------------------------------------------------------ BaseViewMixin
    def show_error(self, error_message: str) -> None:
        self._error_label.setText(error_message)
        self._error_label.show()

    def clear_error(self) -> None:
        self._error_label.hide()
