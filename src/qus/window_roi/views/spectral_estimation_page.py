"""Power and Correlation Guided analysis config (S1Study DetermineSpectralParameters ->
ComputePowerSpectra -> DetermineBandwidth, over the whole ROI):

  correlation lengths inside the ROI -> final window; NW from the ROI's first-pass band;
  multitaper spectra on a window grid over the ROI (reference mean + sample); band in every
  window, shared by sample and reference -> analysis band.

Any 2D ROI works. A Power Guided ROI brings its first-pass band (and correlation settings) in its
pickle; for a hand-drawn / loaded ROI the ROI is the A-lines x depths bounding the segmentation and
its first-pass band is computed first (windows.first_pass_band_for_roi), with the probe's preset.

Accepting turns the result into a Custom analysis config (band, sampling frequency, windows,
overlaps), shown in the usual config preview."""

import warnings
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QFrame, QPushButton, QDoubleSpinBox, QComboBox, QLineEdit,
    QProgressBar, QLabel, QScrollArea, QToolButton, QStackedWidget, QGridLayout,
)

from engines.qus.quantus.seg_loading.correlation_length import PROBE_PRESETS, positions_from_image
from engines.qus.quantus.seg_loading.correlation_length.roi import sector_roi_from_rows_lines, valid_band
from engines.qus.quantus.seg_loading.power_spectrum_mtm import power_spectral_estimation_from_image
from src.qus.seg_spectrum_check import seg_bounding_roi
from src.qus.window_roi.probe import detect_probe_preset, known_spectral_nulls
from src.qus.window_roi.windows import first_pass_band_for_roi
from .common import (
    AXIAL, BG, LATERAL, MUTED, ROI, SIDEBAR_WIDTH, TEXT, UNUSABLE, USABLE, Worker, bmode_db, error_label,
    first_frame, heading, hint, results_box, separator, stop_worker, style_axes, style_legend,
)

# analysis band: the strict shared band (valid in every window of the ROI); the 95th / 5th percentile
# band is still computed (in the details) but not offered
# legends sit below each plot, this far under its bottom edge: tick labels (8 pt) + x-axis title (9 pt)
# + padding. The figure uses matplotlib's constrained layout, which makes room for them at any size.
LEGEND_OFFSET_PT = 34
FAILING_SNAP_PX = 15                      # a click this close (pixels) to a failing window selects it
SAMPLE = "#4fe0d0"       # sample spectrum
REFERENCE = "#ffd166"    # reference spectrum


class SnapSpinBox(QDoubleSpinBox):
    """Arrow / wheel steps land on multiples of the step (92 -> 95 / 90, then 5 % steps); typed values
    are kept as typed."""

    def stepBy(self, steps: int) -> None:
        step = self.singleStep()
        v = self.value()
        if steps > 0:
            target = (np.floor(v / step + 1e-9) + steps) * step
        else:
            target = (np.ceil(v / step - 1e-9) + steps) * step
        self.setValue(min(max(target, self.minimum()), self.maximum()))


class SpectralEstimationPage(QWidget):
    """roi_info: the first-pass details saved with a Power Guided ROI (load_roi_pkl_info), or None;
    seg_data: the segmentation, used when there is no roi_info (any 2D ROI)."""

    params_accepted = pyqtSignal(dict)   # Custom config parameters
    back_requested = pyqtSignal()

    def __init__(self, image_data, roi_info: Optional[dict], parent: Optional[QWidget] = None, seg_data=None):
        super().__init__(parent)
        self._image = image_data
        self._info = roi_info
        rf = first_frame(image_data.rf_data)
        self._axial, self._lateral = positions_from_image(image_data, *rf.shape)
        self._bmode = bmode_db(rf)
        self._roi = None
        self._band = None         # (fLow, fHigh) MHz first-pass band of the ROI (saved, or computed)
        if roi_info is not None:
            self._roi = sector_roi_from_rows_lines(self._axial, self._lateral, roi_info["RF Rows"], roi_info["RF Lines"])
            band = roi_info.get("Band MHz")
            if valid_band(band):   # a reversed / missing band (no shared band) is recomputed in the ROI
                self._band = tuple(float(v) for v in band)
        else:
            box = seg_bounding_roi(seg_data, rf.shape)   # r0, r1, c0, c1 (half-open)
            if box is not None:
                r0, r1, c0, c1 = box
                self._roi = sector_roi_from_rows_lines(self._axial, self._lateral, (r0, r1 - 1), (c0, c1 - 1))
        self._result = None
        self._worker: Optional[Worker] = None
        self._sel = None          # selected window (i, j)
        self._grid_view = None    # zoomed (xlim, ylim) of the window grid plot, None = whole ROI
        self._build_ui()
        self._prefill()   # the main frame starts on the parameter confirmation

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        side_frame = QFrame()
        side_frame.setFixedWidth(SIDEBAR_WIDTH + 40)
        side = QVBoxLayout(side_frame)
        side.addWidget(heading("Guided Config Setting Selection"))
        steps = QLabel(
            "<ul style='margin-left:-20px'>"
            "<li><b>Window size</b> from the correlation lengths inside your ROI</li>"
            "<li><b>Analysis band</b> from multitaper power spectra in every window, shared by sample "
            "and reference</li>"
            "<li>Confirm the parameters, then compute</li>"
            "<li>Press <b>Use for Analysis Config</b>: the window sizes, overlaps and "
            "band are passed to the analysis step</li></ul>")
        steps.setTextFormat(Qt.TextFormat.RichText)
        steps.setWordWrap(True)
        steps.setStyleSheet(f"font-size: 14px; color: {TEXT};")
        side.addWidget(steps)
        self._roi_label = hint("")
        self._roi_label.setStyleSheet("font-size: 13px; color: rgb(200, 200, 200);")
        side.addWidget(self._roi_label)
        side.addWidget(separator())

        # results / errors first: what the user needs to read (progress is shown in the main frame)
        self._results = results_box()
        self._results.hide()
        side.addWidget(self._results)
        self._error = error_label()
        side.addWidget(self._error)

        def spin(lo, hi, step, decimals, suffix="", cls=QDoubleSpinBox):
            s = cls()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            return s

        self._win_thresh = spin(1, 100, 5, 0, " %")
        self._win_thresh.setValue(95)
        self._win_thresh.setToolTip("A window is analysed only if more than this share of it lies inside the ROI.")
        nav = QHBoxLayout()
        self._fail_prev = QToolButton()
        self._fail_prev.setArrowType(Qt.ArrowType.LeftArrow)
        self._fail_next = QToolButton()
        self._fail_next.setArrowType(Qt.ArrowType.RightArrow)
        self._fail_label = QLabel("")
        self._fail_label.setStyleSheet(f"color: {UNUSABLE}; font-weight: bold;")
        nav.addWidget(QLabel("Failing windows:"))
        nav.addWidget(self._fail_prev)
        nav.addWidget(self._fail_label)
        nav.addWidget(self._fail_next)
        nav.addStretch(1)
        self._fail_nav = QWidget()
        self._fail_nav.setLayout(nav)
        self._fail_nav.setToolTip("Step through the windows that are not usable in sample and reference")
        self._fail_nav.hide()
        side.addWidget(self._fail_nav)
        self._change_button = QPushButton("Change parameters and recompute...")
        self._change_button.hide()
        side.addWidget(self._change_button)

        # Advanced (collapsed): rarely changed settings and the full taper / grid details
        self._threshold = spin(0.05, 0.95, 0.05, 2)
        self._kappa = spin(0.01, 1.0, 0.05, 2)
        self._kappa.setToolTip("Target multitaper smoothing W as a fraction of the bandwidth (NW = kappa x B x T)")
        self._avoid = QLineEdit()
        self._avoid.setPlaceholderText("none (e.g. 7.5)")
        self._avoid.setToolTip("Known spectral nulls (MHz, comma separated) kept out of the band and the smoothing")
        self._advanced_toggle = QToolButton()
        self._advanced_toggle.setText("Advanced settings")
        self._advanced_toggle.setCheckable(True)
        self._advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._advanced_toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        side.addWidget(self._advanced_toggle)
        self._advanced = QWidget()
        adv = QVBoxLayout(self._advanced)
        adv.setContentsMargins(8, 0, 0, 0)
        adv.addWidget(hint("Uses appropriate presets for each probe. Changes apply the next time you compute "
                           "(window inclusion coverage: when you press Use for Analysis Config)."))
        form = QFormLayout()
        form.addRow("Correlation threshold:", self._threshold)
        form.addRow("Kappa (W / bandwidth):", self._kappa)
        form.addRow("Avoid frequencies (MHz):", self._avoid)
        form.addRow("Window inclusion coverage:", self._win_thresh)
        adv.addLayout(form)
        self._details = results_box()
        self._details.hide()
        adv.addWidget(self._details)
        self._advanced.hide()
        side.addWidget(self._advanced)
        self._advanced_toggle.toggled.connect(self._set_advanced_open)
        side.addStretch()

        self._accept_button = QPushButton("Use for Analysis Config")
        self._back_button = QPushButton("‹  Back: config type")
        for b in (self._accept_button, self._back_button):
            side.addWidget(b)

        scroll = QScrollArea()
        scroll.setWidget(side_frame)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(SIDEBAR_WIDTH + 60)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        # main frame: parameter confirmation -> progress while computing -> results figure
        self._main = QStackedWidget()

        # 1. confirm the parameters
        self._confirm = QWidget()
        conf = QVBoxLayout(self._confirm)
        conf.addStretch(1)
        title = QLabel("Confirm Parameters")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {TEXT};")
        conf.addWidget(title)
        self._confirm_sub = QLabel("")
        self._confirm_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._confirm_sub.setWordWrap(True)
        self._confirm_sub.setStyleSheet(f"font-size: 14px; color: {MUTED};")
        conf.addWidget(self._confirm_sub)
        self._win_len = spin(0.1, 100, 0.5, 2, " mm")
        self._win_wid = spin(0.1, 200, 0.5, 2, " mm")
        self._ax_factor = spin(0.5, 50, 0.5, 1, " x")
        self._lat_factor = spin(0.5, 50, 0.5, 1, " x")
        self._ov_ax = spin(0, 95, 5, 0, " %", cls=SnapSpinBox)
        self._ov_lat = spin(0, 95, 5, 0, " %", cls=SnapSpinBox)
        self._win_len.setToolTip("Size of the region the correlation length is measured in, at each depth")
        self._win_wid.setToolTip("Size of the region the correlation length is measured in, at each depth")
        self._ax_factor.setToolTip("Window size = factor x correlation length (the largest over the ROI depths)")
        self._lat_factor.setToolTip("Window size = factor x correlation length (the largest over the ROI depths)")
        self._ov_ax.setToolTip("Minimum overlap between neighbouring windows")
        self._ov_lat.setToolTip("Minimum overlap between neighbouring windows")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        for col, text in ((1, "Axial"), (2, "Lateral")):
            head = QLabel(text)
            head.setAlignment(Qt.AlignmentFlag.AlignCenter)
            head.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT};")
            grid.addWidget(head, 0, col)
        for row, (text, ax_w, lat_w) in enumerate((
                ("Correlation measurement window", self._win_len, self._win_wid),
                ("Window size factor (x correlation length)", self._ax_factor, self._lat_factor),
                ("Minimum window overlap", self._ov_ax, self._ov_lat)), start=1):
            label = QLabel(text)
            label.setStyleSheet(f"font-size: 14px; color: {TEXT};")
            grid.addWidget(label, row, 0)
            for col, w in ((1, ax_w), (2, lat_w)):
                w.setMinimumWidth(120)
                grid.addWidget(w, row, col)
        grid_box = QFrame()
        grid_box.setObjectName("confirmBox")
        grid_box.setStyleSheet("QFrame#confirmBox { border: 1px solid rgb(90, 90, 90); border-radius: 8px; }")
        grid_box.setLayout(grid)
        grid_box.setMaximumWidth(760)
        conf.addSpacing(12)
        conf.addWidget(grid_box, 0, Qt.AlignmentFlag.AlignHCenter)
        conf.addSpacing(12)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._cancel_change = QPushButton("Back to results")
        self._cancel_change.hide()
        buttons.addWidget(self._cancel_change)
        self._compute_button = QPushButton("Compute Window Sizes and Power Spectra")
        self._compute_button.setStyleSheet(
            "QPushButton { color: white; font-size: 15px; background: rgb(90, 37, 255); border-radius: 15px; "
            "padding: 8px 22px; } QPushButton:disabled { color: rgb(150, 150, 150); background: rgb(70, 70, 70); }")
        buttons.addWidget(self._compute_button)
        buttons.addStretch(1)
        conf.addLayout(buttons)
        conf.addStretch(2)
        self._main.addWidget(self._confirm)

        # 2. progress while computing
        self._busy = QWidget()
        busy = QVBoxLayout(self._busy)
        busy.addStretch(1)
        self._busy_title = QLabel("Computing Window Sizes and Power Spectra")
        self._busy_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._busy_title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {TEXT};")
        busy.addWidget(self._busy_title)
        self._busy_sub = QLabel("The window sizes, overlaps and analysis band are then passed to the "
                                "analysis step.")
        self._busy_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._busy_sub.setWordWrap(True)
        self._busy_sub.setStyleSheet(f"font-size: 14px; color: {MUTED};")
        busy.addWidget(self._busy_sub)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(14)
        self._progress.setMinimumWidth(500)
        self._progress.setMaximumWidth(700)
        busy.addWidget(self._progress, 0, Qt.AlignmentFlag.AlignHCenter)
        self._progress_label = QLabel("")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet(f"font-size: 14px; color: {TEXT};")
        busy.addWidget(self._progress_label)
        busy.addStretch(2)
        self._main.addWidget(self._busy)

        # 3. results
        self._fig = Figure(facecolor=BG)
        self._canvas = FigureCanvas(self._fig)
        self._main.addWidget(self._canvas)
        root.addWidget(self._main, 1)

        self._compute_button.clicked.connect(self._compute)
        self._change_button.clicked.connect(self._show_confirm)
        self._fail_prev.clicked.connect(lambda: self._step_failing(-1))
        self._fail_next.clicked.connect(lambda: self._step_failing(+1))
        self._cancel_change.clicked.connect(lambda: self._main.setCurrentWidget(self._canvas))
        self._accept_button.clicked.connect(self._on_accept)
        self._back_button.clicked.connect(self.back_requested.emit)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._accept_button.setEnabled(False)

    def _failing(self) -> np.ndarray:
        """(row, col) of the windows not usable in sample and reference, top to bottom."""
        return np.argwhere(~self._result.bandwidth.shared_usable)

    def _step_failing(self, step: int) -> None:
        fail = self._failing()
        if not len(fail):
            return
        here = np.flatnonzero((fail == np.asarray(self._sel)).all(axis=1))
        k = (here[0] + step) % len(fail) if len(here) else (0 if step > 0 else len(fail) - 1)
        self._select(fail[k])

    def _update_failing_nav(self) -> None:
        if self._result is None:
            self._fail_nav.hide()
            return
        fail = self._failing()
        self._fail_nav.setVisible(len(fail) > 0)
        here = np.flatnonzero((fail == np.asarray(self._sel)).all(axis=1)) if len(fail) else []
        self._fail_label.setText(f"{here[0] + 1} of {len(fail)}" if len(here) else f"{len(fail)}")

    def _show_confirm(self) -> None:
        """Parameter confirmation in the main frame ('Back to results' once there are results)."""
        self._cancel_change.setVisible(self._result is not None)
        self._main.setCurrentWidget(self._confirm)

    def _prefill(self) -> None:
        cs = dict((self._info or {}).get("Correlation Settings") or {})
        if not cs:   # not a Power Guided ROI: the probe's preset
            cs = {"preset": detect_probe_preset(self._image)}
        preset = PROBE_PRESETS.get(cs.get("preset") or "", {})
        name = cs.get("preset")
        self._confirm_sub.setText(
            (f"Defaults use the appropriate presets for the {name} probe. " if name in PROBE_PRESETS
             else "No preset for this probe: defaults are generic. ")
            + "Check them, then compute.")
        span = (self._roi.depth_range_mm[1] - self._roi.depth_range_mm[0]) if self._roi is not None else 10.0
        default_len = round(min(10.0, max(1.0, span / 3)), 1)
        self._win_len.setValue(cs.get("window_length_mm", preset.get("window_length_mm", default_len)))
        self._win_wid.setValue(cs.get("window_width_mm", preset.get("window_width_mm", default_len)))
        self._threshold.setValue(cs.get("threshold", preset.get("threshold", 0.2)))
        self._ax_factor.setValue(cs.get("ax_window_factor", 10.0))
        self._lat_factor.setValue(cs.get("lat_window_factor", 5.0))
        self._kappa.setValue(0.2)
        nulls = known_spectral_nulls(self._image)   # e.g. L20 on software 13.0.0: 7.5 MHz
        self._avoid.setText(", ".join(f"{v:g}" for v in nulls))
        if nulls:
            self._confirm_sub.setText(self._confirm_sub.text() + " The known spectral null of this probe / "
                                      f"software at {', '.join(f'{v:g}' for v in nulls)} MHz is kept out of the band "
                                      "(Advanced settings).")
        self._ov_ax.setValue(90)
        self._ov_lat.setValue(85)
        if self._roi is None:
            self._roi_label.setText("")
            self._error.setText("This config needs a 2D ROI on the RF data (the segmentation has no usable mask).")
            self._error.show()
            self._compute_button.setEnabled(False)
            return
        self._show_roi_label()

    def _show_roi_label(self) -> None:
        (r1, r2), (l1, l2) = self._roi.rows, self._roi.lines
        d0, d1 = self._roi.depth_range_mm
        source = "bounding A-lines / depths of the segmentation" if self._info is None else "Guided ROI"
        self._roi_label.setText(f"ROI ({source}): A-lines {l1}-{l2}, depth {d0:.1f}-{d1:.1f} mm")

    def _avoid_freqs(self) -> tuple:
        text = self._avoid.text().replace(";", ",").strip()
        if not text:
            return ()
        return tuple(float(v) for v in text.split(",") if v.strip())

    # ------------------------------------------------------------------ compute
    def _compute(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            avoid = self._avoid_freqs()
        except ValueError:
            self._error.setText("Avoid frequencies: numbers in MHz, separated by commas.")
            self._error.show()
            return
        band = self._band
        ckw = dict(window_length_mm=self._win_len.value(), window_width_mm=self._win_wid.value(),
                   threshold=self._threshold.value(), ax_window_factor=self._ax_factor.value(),
                   lat_window_factor=self._lat_factor.value())
        kw = dict(kappa=self._kappa.value(), avoid_freq=avoid, min_overlap_ax=self._ov_ax.value() / 100,
                  min_overlap_lat=self._ov_lat.value() / 100)
        roi = self._roi

        def work(report):
            roi_band = band
            if roi_band is None:   # not saved with the ROI: first-pass band in this ROI first
                roi_band = first_pass_band_for_roi(self._image, roi, ckw,
                                                   report=lambda f, m: report(0.1 * f, m))
                self._band = roi_band
            return power_spectral_estimation_from_image(self._image, roi, roi_band_mhz=roi_band, correlation_kwargs=ckw,
                                                        progress=lambda f, m: report(0.1 + 0.9 * f, m)
                                                        if band is None else report(f, m), **kw)

        self._result = None
        self._error.hide()
        self._accept_button.setEnabled(False)
        self._compute_button.setEnabled(False)
        self._results.hide()
        self._details.hide()
        self._change_button.hide()
        self._fail_nav.hide()
        self._progress.setValue(0)
        self._progress_label.setText("Starting...")
        self._set_busy(True)
        self._worker = Worker(work, self, with_progress=True)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _set_busy(self, busy: bool) -> None:
        """Progress panel in the main frame while computing; afterwards the results figure if there
        are results, else back to the parameter confirmation."""
        if busy:
            self._main.setCurrentWidget(self._busy)
        elif self._result is not None:
            self._main.setCurrentWidget(self._canvas)
        else:
            self._show_confirm()

    def _on_progress(self, frac: float, msg: str) -> None:
        self._progress.setValue(int(1000 * frac))
        self._progress_label.setText(f"{msg}  ({frac:.0%})")

    def _on_done(self, result) -> None:
        self._compute_button.setEnabled(True)
        self._result = result
        self._set_busy(False)
        self._results.show()
        self._change_button.show()
        self._show_roi_label()   # shows the first-pass band if it was computed in this run
        p, g, bw = result.parameters, result.grid, result.bandwidth
        c = p.check
        rows = "".join(
            f"<tr><td>{nw:g}</td><td>{k}</td><td>{w:.2f}</td><td>{wb:.2f}</td><td>{s}</td></tr>"
            for nw, k, w, wb, s in zip(c.nw, c.k, c.w_mhz, c.w_over_b, c.status))
        note = f"<br><span style='color:{UNUSABLE}'><b>{p.note}</b></span>" if p.note else ""
        n_note = (f"<br><span style='color:{MUTED}'>Grid windows are {int(g.ax_lengths[0])} samples; NW chosen "
                  f"for N = {p.n}</span>" if int(g.ax_lengths[0]) != p.n else "")
        usable_colour = TEXT if bw.n_usable == bw.n_windows else UNUSABLE
        self._results.setText(
            f"<b>Window size</b>: <b>{result.ax_window_mm:.1f} mm</b> axial \u00d7 "
            f"<b>{result.lat_window_mm:.1f} mm</b> lateral<br>"
            f"<b>Analysis band</b>: <b>{bw.shared_strict[0]:.1f} - {bw.shared_strict[1]:.1f} MHz</b> "
            f"(valid in every window)<br>"
            f"<span style='color:{usable_colour}'>{bw.n_usable} of {bw.n_windows} windows usable in sample and "
            f"reference</span>{note}")
        self._details.show()
        self._details.setText(
            f"<b>Final window</b> (correlation inside the ROI): <b>{result.ax_window_mm:.1f} mm</b> axial x "
            f"<b>{result.lat_window_mm:.1f} mm</b> lateral<br>"
            f"<b>Band for NW</b>: {p.roi_band_mhz[0]:.2f}-{p.roi_band_mhz[1]:.2f} -> "
            f"{p.band_mhz[0]:.1f}-{p.band_mhz[1]:.1f} MHz<br>"
            f"<b>NW {p.nw:g}</b> (recommended {c.nw_recommended:.2f}), K = {p.k} tapers, W = {p.w_mhz:.2f} MHz "
            f"= {p.w_mhz / (p.band_mhz[1] - p.band_mhz[0]):.2f} x B{note}{n_note}<br>"
            f"<table cellspacing='4'><tr><td><b>NW</b></td><td><b>K</b></td><td><b>W (MHz)</b></td>"
            f"<td><b>W/B</b></td><td><b>Status</b></td></tr>{rows}</table>"
            f"<b>Grid</b>: {g.shape[0]} x {g.shape[1]} windows, {int(g.ax_lengths[0])} samples x "
            f"{int(g.lat_widths.min())}-{int(g.lat_widths.max())} lines, overlap {g.ax_overlap.min():.0%} axial / "
            f"{g.lat_overlap.min():.0%} lateral<br>"
            f"<span style='color:{usable_colour}'><b>{bw.n_usable} of {bw.n_windows}</b> windows usable in sample "
            f"and reference</span><br>"
            f"<b>Shared band</b> (valid in every window): {bw.shared_strict[0]:.1f}-{bw.shared_strict[1]:.1f} MHz")
        self._accept_button.setEnabled(bool(np.all(np.isfinite(self._chosen_band()))))
        self._sel = self._default_window()
        self._grid_view = None
        self._update_failing_nav()
        try:
            self._redraw()
        except Exception as e:  # never let a plotting problem take down the app
            self._error.setText(f"Could not draw the results: {e}")
            self._error.show()

    def _on_failed(self, error) -> None:
        self._compute_button.setEnabled(True)
        self._set_busy(False)
        self._results.hide()
        self._error.setText(str(error))
        self._error.show()
        self._advanced_toggle.setChecked(True)   # the settings are the way to fix it

    def _set_advanced_open(self, open_: bool) -> None:
        self._advanced.setVisible(open_)
        self._advanced_toggle.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)

    def _chosen_band(self) -> tuple:
        """The analysis band: the strict shared band (valid in every window)."""
        return self._result.bandwidth.shared_strict

    def _default_window(self):
        """Deepest row, middle column (as DetermineBW_ROI's example spectrum)."""
        n_ax, n_lat = self._result.grid.shape
        return n_ax - 1, n_lat // 2

    # ------------------------------------------------------------------ figure
    def _legend_below(self, ax, **kwargs) -> None:
        """Legend centred below the axes, LEGEND_OFFSET_PT under its bottom edge (clear of the x-axis
        tick labels and title at any figure size)."""
        off = ScaledTranslation(0, -LEGEND_OFFSET_PT / 72, ax.figure.dpi_scale_trans)
        style_legend(ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                               bbox_transform=ax.transAxes + off, **kwargs))

    def _redraw(self) -> None:
        r = self._result
        fig = self._fig
        fig.clear()
        # constrained layout: room for every title, axis label and legend, recomputed at each draw / resize
        fig.set_layout_engine("constrained", h_pad=0.08, w_pad=0.08, hspace=0.06, wspace=0.06)
        gs = fig.add_gridspec(2, 2)
        # top: window size and band vs depth; bottom: the window grid (click) and that window's spectra
        self._plot_correlation(fig.add_subplot(gs[0, 0]))
        self._plot_band_depth(fig.add_subplot(gs[0, 1]))
        self._ax_grid = fig.add_subplot(gs[1, 0])
        self._plot_grid(self._ax_grid)
        self._ax_spec = fig.add_subplot(gs[1, 1])
        self._plot_spectra(self._ax_spec)
        lo, hi = self._chosen_band()
        fig.suptitle(f"Multitaper Power Spectra in the ROI: NW {r.parameters.nw:g}, {r.ax_window_mm:.1f} x "
                     f"{r.lat_window_mm:.1f} mm Windows\nAnalysis Band {lo:.1f} - {hi:.1f} MHz", color=TEXT,
                     fontsize=11)
        self._canvas.draw_idle()

    def _plot_correlation(self, ax) -> None:
        """Suggested window size (factor x correlation length) at each depth in the ROI; the window
        used is the largest of each (dashed)."""
        r = self._result
        c = r.correlation
        d = c.depths_mm
        for values, final, colour, label in (
                (c.recommended_ax_window_mm, r.ax_window_mm, AXIAL, f"Axial ({c.ax_window_factor:g}x length)"),
                (c.recommended_lat_window_mm, r.lat_window_mm, LATERAL, f"Lateral ({c.lat_window_factor:g}x width)")):
            ax.step(values, d, where="mid", color=colour, linewidth=2, label=label)
            if np.isfinite(final):
                ax.axvline(final, color=colour, linestyle="--", linewidth=1.2,
                           label=f"{label.split(' (')[0]} used: {final:.1f} mm")
        ax.invert_yaxis()
        finals = [v for v in (r.ax_window_mm, r.lat_window_mm) if np.isfinite(v)]
        ax.set_xlim(0, 1.3 * max(finals) if finals else None)
        style_axes(ax, "Suggested Window Size vs. Depth", "Window size (mm)", "Depth (mm)")
        self._legend_below(ax, ncol=2)

    def _plot_grid(self, ax) -> None:
        r = self._result
        ax.pcolormesh(self._lateral, self._axial, self._bmode, cmap="gray", vmin=-60, vmax=0, shading="nearest",
                      rasterized=True)
        x, z = r.grid.centres_mm(self._axial, self._lateral)
        u = r.bandwidth.shared_usable
        # usable windows are not marked (they cover the ROI); only the failing ones are
        if (~u).any():   # failing windows: large markers on top, easy to see and click
            ax.scatter(x[~u], z[~u], s=36, marker="x", color=UNUSABLE, linewidths=1.6, zorder=5,
                       label=f"Not usable ({int((~u).sum())})")
        roi = self._roi
        ax.plot(np.r_[roi.x_mm, roi.x_mm[0]], np.r_[roi.z_mm, roi.z_mm[0]], color=ROI, linewidth=2, label="ROI")
        i, j = self._sel
        g = r.grid
        rows = [g.start_ax[i, j], g.end_ax[i, j]]
        cols = np.arange(g.start_lat[i, j], g.end_lat[i, j] + 1)
        bx = np.r_[self._lateral[rows[0], cols], self._lateral[rows[1], cols[::-1]], self._lateral[rows[0], cols[0]]]
        bz = np.r_[self._axial[rows[0], cols], self._axial[rows[1], cols[::-1]], self._axial[rows[0], cols[0]]]
        ax.plot(bx, bz, color="white", linewidth=1.5, label="Selected window")
        ax.set_aspect("equal", adjustable="datalim")
        if self._grid_view is not None:   # keep the user's zoom across redraws
            ax.set_xlim(*self._grid_view[0])
            ax.set_ylim(*self._grid_view[1])
        else:
            pad = 5.0
            ax.set_xlim(roi.x_mm.min() - pad, roi.x_mm.max() + pad)
            ax.set_ylim(roi.z_mm.max() + pad, max(roi.z_mm.min() - pad, float(np.nanmin(self._axial))))
        style_axes(ax, "Window Grid (Click a Window to Show Its Spectra)" if u.all() else
                   "Window Grid (Click a Window, e.g. a Red \u2715; Scroll to Zoom)", "Lateral (mm)", "Depth (mm)")
        ax.grid(False)
        self._legend_below(ax, ncol=2, markerscale=2)

    def _plot_spectra(self, ax) -> None:
        r = self._result
        i, j = self._sel
        bw, f = r.bandwidth, r.freq
        with np.errstate(divide="ignore"):
            s_db = 10 * np.log10(r.sample_spectra[i, j])
            r_db = 10 * np.log10(r.reference_spectra[i, j])
        ax.plot(f, s_db, color=SAMPLE, linewidth=1.5, label="Sample")
        ax.plot(f, r_db, color=REFERENCE, linewidth=1.5, label="Reference (mean)")
        for lo, hi, colour in ((bw.sample_low[i, j], bw.sample_high[i, j], SAMPLE),
                               (bw.ref_low[i, j], bw.ref_high[i, j], REFERENCE)):
            for v in (lo, hi):
                if np.isfinite(v):
                    ax.axvline(v, color=colour, linestyle="--", linewidth=1)
        slo, shi = bw.shared_low[i, j], bw.shared_high[i, j]
        if np.isfinite(slo) and np.isfinite(shi):
            ax.axvspan(slo, shi, color=USABLE, alpha=0.15, label="Shared band")
        w = r.parameters.w_mhz
        lo, hi = self._chosen_band()
        if np.isfinite(lo):
            ax.axvspan(lo - w, lo, color="white", alpha=0.08)
            ax.axvspan(hi, hi + w, color="white", alpha=0.08, label=f"±W smoothing ({w:.2f} MHz)")
        top = float(np.nanmax(np.r_[s_db[np.isfinite(s_db)], r_db[np.isfinite(r_db)]]))
        ax.set_ylim(top - 70, top + 5)
        fmax = np.nanmax(np.r_[bw.sample_high[np.isfinite(bw.sample_high)], bw.ref_high[np.isfinite(bw.ref_high)], 0])
        ax.set_xlim(0, min(f[-1], 1.5 * fmax) if fmax > 0 else f[-1])
        depth = r.depths_mm[i]
        ok = bool(bw.shared_usable[i, j])
        style_axes(ax, f"Spectra at {depth:.1f} mm, Window Column {j + 1} of {r.grid.shape[1]}"
                       + ("" if ok else "  (Not Usable)"), "Frequency (MHz)", "Power (dB)")
        if not ok:
            ax.title.set_color(UNUSABLE)
        self._legend_below(ax, ncol=2)

    def _plot_band_depth(self, ax) -> None:
        r = self._result
        bw, d = r.bandwidth, r.depths_mm
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows
            lo =np.nanmean(np.where(bw.shared_usable, bw.shared_low, np.nan), axis=1)
            hi = np.nanmean(np.where(bw.shared_usable, bw.shared_high, np.nan), axis=1)
            lo_min = np.nanmax(np.where(bw.shared_usable, bw.shared_low, np.nan), axis=1)
            hi_min = np.nanmin(np.where(bw.shared_usable, bw.shared_high, np.nan), axis=1)
        ax.fill_betweenx(d, lo_min, hi_min, color=USABLE, alpha=0.15, linewidth=0, label="Valid in every window of row")
        ax.plot(lo, d, color=USABLE, linewidth=2, label="Shared band, row mean")
        ax.plot(hi, d, color=USABLE, linewidth=2)
        for k, v in enumerate(bw.shared_strict):
            if np.isfinite(v):
                ax.axvline(v, color="white", linewidth=1.3, label="Analysis band" if k == 0 else None)
        bad_rows = ~bw.shared_usable.all(axis=1)
        if bad_rows.any():
            ax.plot(np.full(bad_rows.sum(), 0.02 * max(np.nanmax(np.r_[hi, 1]), 1)), d[bad_rows], "x",
                    color=UNUSABLE, label="Row with unusable windows")
        ax.invert_yaxis()
        style_axes(ax, "Shared Band vs. Depth", "Frequency (MHz)", "Depth (mm)")
        self._legend_below(ax, ncol=2)

    def _on_scroll(self, event) -> None:
        """Scroll on the window grid: zoom in / out around the cursor."""
        ax = getattr(self, "_ax_grid", None)
        if self._result is None or event.inaxes is not ax or event.xdata is None:
            return
        f = 0.8 if event.step > 0 else 1.25
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        self._grid_view = ((cx + (x0 - cx) * f, cx + (x1 - cx) * f), (cy + (y0 - cy) * f, cy + (y1 - cy) * f))
        ax.set_xlim(*self._grid_view[0])
        ax.set_ylim(*self._grid_view[1])
        self._canvas.draw_idle()

    def _on_click(self, event) -> None:
        if self._result is None or event.inaxes is not getattr(self, "_ax_grid", None) or event.xdata is None:
            return
        if event.dblclick:   # double-click: back to the whole ROI
            self._grid_view = None
            self._redraw()
            return
        x, z = self._result.grid.centres_mm(self._axial, self._lateral)
        # distances in screen pixels, so "near" means the same at any zoom / aspect
        pts = self._ax_grid.transData.transform(np.c_[x.ravel(), z.ravel()])
        d2 = (pts[:, 0] - event.x) ** 2 + (pts[:, 1] - event.y) ** 2
        failing = ~self._result.bandwidth.shared_usable.ravel()
        if failing.any() and d2[failing].min() <= FAILING_SNAP_PX ** 2:
            k = int(np.flatnonzero(failing)[np.argmin(d2[failing])])   # a failing window near the click wins
        else:
            k = int(np.argmin(d2))
        self._select(np.unravel_index(k, x.shape))

    def _select(self, sel) -> None:
        self._sel = tuple(int(v) for v in sel)
        self._update_failing_nav()
        try:
            self._redraw()
        except Exception as e:
            self._error.setText(f"Could not draw the results: {e}")
            self._error.show()

    # ------------------------------------------------------------------ accept
    def config_params(self) -> dict:
        """Custom config parameters from the result (custom config loader format)."""
        r = self._result
        lo, hi = self._chosen_band()
        fs_hz = int(round(r.parameters.fs_mhz * 1e6))
        return {
            "transducer_freq_band": [0, fs_hz // 2],
            "analysis_freq_band": [int(round(lo * 1e6)), int(round(hi * 1e6))],
            "center_frequency": int(round((lo + hi) / 2 * 1e6)),
            "sampling_frequency": fs_hz,
            "ax_win_size": float(r.ax_window_mm),
            "lat_win_size": float(r.lat_window_mm),
            "ax_win_size_units": "mm",
            # TODO(curvilinear, flagged 2026-11-19): the analysis converts this mm size to A-lines with
            # image_data.lateral_res, which for Clarius curvilinear probes is a placeholder (parser.py
            # xResRF = yResRF * scan-converted aspect, 0.035 mm on C3 vs a real 0.39-1.01 mm A-line spacing
            # in the ROI), e.g. 11 mm -> 313 A-lines > 192, so no windows. Options: pass A-lines here
            # (mm / mean A-line spacing in the ROI), or fix xResRF for curvilinear probes in the loader.
            "lat_win_size_units": "mm",
            "window_thresh": float(self._win_thresh.value() / 100),
            "axial_overlap": float(self._ov_ax.value() / 100),
            "lateral_overlap": float(self._ov_lat.value() / 100),
            "speed_of_sound": 1540.0,
        }

    def _on_accept(self) -> None:
        if self._result is None:
            return
        if not np.all(np.isfinite(self._chosen_band())):
            self._error.setText("No shared band - nothing to use for the analysis config.")
            self._error.show()
            return
        self.params_accepted.emit(self.config_params())

    @property
    def result(self):
        return self._result

    # BaseViewMixin-style hooks used by the config coordinator
    def show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.show()

    def clear_error(self) -> None:
        self._error.hide()

    def show_loading(self) -> None:
        self._accept_button.setEnabled(False)

    def hide_loading(self) -> None:
        self._accept_button.setEnabled(self._result is not None)

    def shutdown(self) -> None:
        stop_worker(self._worker)
