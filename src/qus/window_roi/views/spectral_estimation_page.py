"""Power and Correlation Guided analysis config (S1Study DetermineSpectralParameters ->
ComputePowerSpectra -> DetermineBandwidth, over the whole ROI):

  correlation lengths inside the Power Guided ROI -> final window; NW from the ROI's first-pass
  band; multitaper spectra on a window grid over the ROI (reference mean + sample); band in every
  window, shared by sample and reference -> analysis band.

Accepting turns the result into a Custom analysis config (band, sampling frequency, windows,
overlaps), shown in the usual config preview."""

import warnings
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QFrame, QPushButton, QDoubleSpinBox, QComboBox, QLineEdit,
    QProgressBar, QLabel, QScrollArea,
)

from engines.qus.quantus.seg_loading.correlation_length import PROBE_PRESETS, positions_from_image
from engines.qus.quantus.seg_loading.correlation_length.roi import sector_roi_from_rows_lines
from engines.qus.quantus.seg_loading.power_spectrum_mtm import power_spectral_estimation_from_image
from .common import (
    AXIAL, BG, LATERAL, MUTED, ROI, SIDEBAR_WIDTH, TEXT, UNUSABLE, USABLE, Worker, bmode_db, error_label,
    first_frame, heading, hint, results_box, separator, stop_worker, style_axes, style_legend,
)

BAND_STRICT = "Strict (valid in every window)"
BAND_PCTL = "95th / 5th percentile"
SAMPLE = "#4fe0d0"       # sample spectrum
REFERENCE = "#ffd166"    # reference spectrum


class SpectralEstimationPage(QWidget):
    """roi_info: the first-pass details saved with a Power Guided ROI (load_roi_pkl_info), or None."""

    params_accepted = pyqtSignal(dict)   # Custom config parameters
    back_requested = pyqtSignal()

    def __init__(self, image_data, roi_info: Optional[dict], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image = image_data
        self._info = roi_info
        rf = first_frame(image_data.rf_data)
        self._axial, self._lateral = positions_from_image(image_data, *rf.shape)
        self._bmode = bmode_db(rf)
        self._roi = None
        if roi_info is not None:
            self._roi = sector_roi_from_rows_lines(self._axial, self._lateral, roi_info["RF Rows"], roi_info["RF Lines"])
        self._result = None
        self._worker: Optional[Worker] = None
        self._sel = None          # selected window (i, j)
        self._build_ui()
        self._prefill()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        side_frame = QFrame()
        side_frame.setFixedWidth(SIDEBAR_WIDTH + 40)
        side = QVBoxLayout(side_frame)
        side.addWidget(heading("Power and Correlation Guided"))
        side.addWidget(hint(
            "Multitaper power spectra over the ROI with the final window settings: correlation lengths "
            "inside the ROI give the window, the ROI's first-pass band gives the multitaper NW, and the band "
            "is recomputed in every window, shared by sample and reference."))
        self._roi_label = hint("")
        side.addWidget(self._roi_label)
        side.addWidget(separator())

        def spin(lo, hi, step, decimals, suffix=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            return s

        form = QFormLayout()
        self._win_len = spin(0.1, 100, 0.5, 2, " mm")
        self._win_wid = spin(0.1, 200, 0.5, 2, " mm")
        self._threshold = spin(0.05, 0.95, 0.05, 2)
        self._ax_factor = spin(0.5, 50, 0.5, 1, " x")
        self._lat_factor = spin(0.5, 50, 0.5, 1, " x")
        self._kappa = spin(0.01, 1.0, 0.05, 2)
        self._kappa.setToolTip("Target multitaper smoothing W as a fraction of the bandwidth (NW = kappa x B x T)")
        self._avoid = QLineEdit()
        self._avoid.setPlaceholderText("none (e.g. 7.5)")
        self._avoid.setToolTip("Known spectral nulls (MHz, comma separated) kept out of the band and the smoothing")
        self._ov_ax = spin(0, 99, 1, 0, " %")
        self._ov_lat = spin(0, 99, 1, 0, " %")
        form.addRow("Measurement window (axial):", self._win_len)
        form.addRow("Measurement window (lateral):", self._win_wid)
        form.addRow("Correlation threshold:", self._threshold)
        form.addRow("Axial window factor:", self._ax_factor)
        form.addRow("Lateral window factor:", self._lat_factor)
        form.addRow("Kappa (W / bandwidth):", self._kappa)
        form.addRow("Avoid frequencies (MHz):", self._avoid)
        form.addRow("Min axial overlap:", self._ov_ax)
        form.addRow("Min lateral overlap:", self._ov_lat)
        side.addLayout(form)

        self._compute_button = QPushButton("Compute Power Spectra")
        side.addWidget(self._compute_button)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._progress.hide()
        self._progress_label = hint("")
        self._progress_label.hide()
        side.addWidget(self._progress)
        side.addWidget(self._progress_label)
        self._results = results_box()
        side.addWidget(self._results)
        self._error = error_label()
        side.addWidget(self._error)
        side.addStretch()

        out = QFormLayout()
        self._band_choice = QComboBox()
        self._band_choice.addItems([BAND_STRICT, BAND_PCTL])
        self._win_thresh = spin(1, 100, 5, 0, " %")
        self._win_thresh.setValue(95)
        self._win_thresh.setToolTip("Analysis config: share of a window that must lie inside the ROI")
        out.addRow("Analysis band:", self._band_choice)
        out.addRow("Config window threshold:", self._win_thresh)
        side.addLayout(out)
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

        self._fig = Figure(facecolor=BG)
        self._canvas = FigureCanvas(self._fig)
        root.addWidget(self._canvas, 1)

        self._compute_button.clicked.connect(self._compute)
        self._band_choice.currentTextChanged.connect(lambda _: self._result is not None and self._redraw())
        self._accept_button.clicked.connect(self._on_accept)
        self._back_button.clicked.connect(self.back_requested.emit)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._accept_button.setEnabled(False)

    def _prefill(self) -> None:
        cs = dict((self._info or {}).get("Correlation Settings") or {})
        preset = PROBE_PRESETS.get(cs.get("preset") or "", {})
        span = (self._roi.depth_range_mm[1] - self._roi.depth_range_mm[0]) if self._roi is not None else 10.0
        default_len = round(min(10.0, max(1.0, span / 3)), 1)
        self._win_len.setValue(cs.get("window_length_mm", preset.get("window_length_mm", default_len)))
        self._win_wid.setValue(cs.get("window_width_mm", preset.get("window_width_mm", default_len)))
        self._threshold.setValue(cs.get("threshold", preset.get("threshold", 0.2)))
        self._ax_factor.setValue(cs.get("ax_window_factor", 10.0))
        self._lat_factor.setValue(cs.get("lat_window_factor", 5.0))
        self._kappa.setValue(0.2)
        self._ov_ax.setValue(92)
        self._ov_lat.setValue(85)
        if self._info is None or self._roi is None:
            self._roi_label.setText("")
            self._error.setText("This config needs an ROI from Power Guided ROI Selection (it uses the ROI's "
                                "first-pass band). Go back to segmentation and choose Power Guided ROI Selection, "
                                "or load an ROI saved by it.")
            self._error.show()
            self._compute_button.setEnabled(False)
            return
        band = self._info.get("Band MHz")
        (r1, r2), (l1, l2) = self._roi.rows, self._roi.lines
        d0, d1 = self._roi.depth_range_mm
        self._roi_label.setText(
            f"ROI: A-lines {l1}-{l2}, depth {d0:.1f}-{d1:.1f} mm\n"
            + (f"First-pass band: {band[0]:.2f}-{band[1]:.2f} MHz" if band else "First-pass band: none saved"))
        if not band or not np.all(np.isfinite(band)):
            self._error.setText("The ROI has no first-pass band; redraw it in Power Guided ROI Selection.")
            self._error.show()
            self._compute_button.setEnabled(False)

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
        band = tuple(float(v) for v in self._info["Band MHz"])
        ckw = dict(window_length_mm=self._win_len.value(), window_width_mm=self._win_wid.value(),
                   threshold=self._threshold.value(), ax_window_factor=self._ax_factor.value(),
                   lat_window_factor=self._lat_factor.value())
        kw = dict(kappa=self._kappa.value(), avoid_freq=avoid, min_overlap_ax=self._ov_ax.value() / 100,
                  min_overlap_lat=self._ov_lat.value() / 100)
        roi = self._roi

        def work(report):
            return power_spectral_estimation_from_image(self._image, roi, roi_band_mhz=band, correlation_kwargs=ckw,
                                                        progress=report, **kw)

        self._result = None
        self._error.hide()
        self._accept_button.setEnabled(False)
        self._compute_button.setEnabled(False)
        self._results.setText("Computing (multitaper spectra take a few minutes)...")
        self._progress.setValue(0)
        self._progress.show()
        self._progress_label.show()
        self._worker = Worker(work, self, with_progress=True)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, frac: float, msg: str) -> None:
        self._progress.setValue(int(1000 * frac))
        self._progress_label.setText(f"{msg}  ({frac:.0%})")

    def _on_done(self, result) -> None:
        self._compute_button.setEnabled(True)
        self._progress.hide()
        self._progress_label.hide()
        self._result = result
        p, g, bw = result.parameters, result.grid, result.bandwidth
        c = p.check
        rows = "".join(
            f"<tr><td>{nw:g}</td><td>{k}</td><td>{w:.2f}</td><td>{wb:.2f}</td><td>{s}</td></tr>"
            for nw, k, w, wb, s in zip(c.nw, c.k, c.w_mhz, c.w_over_b, c.status))
        note = f"<br><span style='color:{UNUSABLE}'><b>{p.note}</b></span>" if p.note else ""
        n_note = (f"<br><span style='color:{MUTED}'>Grid windows are {int(g.ax_lengths[0])} samples; NW chosen "
                  f"for N = {p.n} (as the MATLAB)</span>" if int(g.ax_lengths[0]) != p.n else "")
        usable_colour = TEXT if bw.n_usable == bw.n_windows else UNUSABLE
        self._results.setText(
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
            f"<b>Shared band</b>: strict {bw.shared_strict[0]:.1f}-{bw.shared_strict[1]:.1f} MHz, "
            f"95/5 pct {bw.shared_pctl[0]:.1f}-{bw.shared_pctl[1]:.1f} MHz")
        self._accept_button.setEnabled(bool(np.all(np.isfinite(self._chosen_band()))))
        self._sel = self._default_window()
        try:
            self._redraw()
        except Exception as e:  # never let a plotting problem take down the app
            self._error.setText(f"Could not draw the results: {e}")
            self._error.show()

    def _on_failed(self, error) -> None:
        self._compute_button.setEnabled(True)
        self._progress.hide()
        self._progress_label.hide()
        self._results.setText("")
        self._error.setText(str(error))
        self._error.show()

    def _chosen_band(self) -> tuple:
        bw = self._result.bandwidth
        return bw.shared_strict if self._band_choice.currentText() == BAND_STRICT else bw.shared_pctl

    def _default_window(self):
        """Deepest row, middle column (as DetermineBW_ROI's example spectrum)."""
        n_ax, n_lat = self._result.grid.shape
        return n_ax - 1, n_lat // 2

    # ------------------------------------------------------------------ figure
    def _redraw(self) -> None:
        r = self._result
        fig = self._fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.42, left=0.07, right=0.97, top=0.9, bottom=0.1)
        self._plot_correlation(fig.add_subplot(gs[0, 0]))
        self._ax_grid = fig.add_subplot(gs[0, 1])
        self._plot_grid(self._ax_grid)
        self._ax_spec = fig.add_subplot(gs[1, 0])
        self._plot_spectra(self._ax_spec)
        self._plot_band_depth(fig.add_subplot(gs[1, 1]))
        lo, hi = self._chosen_band()
        fig.suptitle(f"Multitaper Power Spectra in the ROI: NW {r.parameters.nw:g}, {r.ax_window_mm:.1f} x "
                     f"{r.lat_window_mm:.1f} mm Windows\nAnalysis Band {lo:.1f} - {hi:.1f} MHz", color=TEXT,
                     fontsize=11)
        self._canvas.draw_idle()

    def _plot_correlation(self, ax) -> None:
        c = self._result.correlation
        d = c.depths_mm
        for mean, sd, colour, label in ((c.ax_mean_mm, c.ax_std_mm, AXIAL, "Axial"),
                                        (c.lat_mean_mm, c.lat_std_mm, LATERAL, "Lateral")):
            ax.plot(mean, d, color=colour, linewidth=2, label=label)
            ax.fill_betweenx(d, mean - sd, mean + sd, color=colour, alpha=0.2, linewidth=0)
        ax.invert_yaxis()
        style_axes(ax, "Correlation Lengths in the ROI (Mean ± SD)", "Correlation length (mm)", "Depth (mm)")
        style_legend(ax.legend(fontsize=8, loc="lower right"))

    def _plot_grid(self, ax) -> None:
        r = self._result
        ax.pcolormesh(self._lateral, self._axial, self._bmode, cmap="gray", vmin=-60, vmax=0, shading="nearest",
                      rasterized=True)
        x, z = r.grid.centres_mm(self._axial, self._lateral)
        u = r.bandwidth.shared_usable
        ax.scatter(x[u], z[u], s=1.5, color=USABLE, linewidths=0, label="Usable window centre", rasterized=True)
        if (~u).any():
            ax.scatter(x[~u], z[~u], s=4, color=UNUSABLE, linewidths=0, label="Not usable", rasterized=True)
        roi = self._roi
        ax.plot(np.r_[roi.x_mm, roi.x_mm[0]], np.r_[roi.z_mm, roi.z_mm[0]], color=ROI, linewidth=2, label="ROI")
        i, j = self._sel
        g = r.grid
        rows = [g.start_ax[i, j], g.end_ax[i, j]]
        cols = np.arange(g.start_lat[i, j], g.end_lat[i, j] + 1)
        bx = np.r_[self._lateral[rows[0], cols], self._lateral[rows[1], cols[::-1]], self._lateral[rows[0], cols[0]]]
        bz = np.r_[self._axial[rows[0], cols], self._axial[rows[1], cols[::-1]], self._axial[rows[0], cols[0]]]
        ax.plot(bx, bz, color="white", linewidth=1.5, label="Selected window")
        pad = 5.0
        ax.set_xlim(roi.x_mm.min() - pad, roi.x_mm.max() + pad)
        ax.set_ylim(roi.z_mm.max() + pad, max(roi.z_mm.min() - pad, float(np.nanmin(self._axial))))
        ax.set_aspect("equal", adjustable="datalim")
        style_axes(ax, "Window Grid (Click a Window to Show Its Spectra)", "Lateral (mm)", "Depth (mm)")
        ax.grid(False)
        style_legend(ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, markerscale=4))

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
        style_legend(ax.legend(fontsize=7, loc="upper right"))

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
        for band, style, label in ((bw.shared_strict, "-", "Strict"), (bw.shared_pctl, "--", "95/5 pct")):
            for k, v in enumerate(band):
                if np.isfinite(v):
                    ax.axvline(v, color="white", linestyle=style, linewidth=1.3, label=label if k == 0 else None)
        bad_rows = ~bw.shared_usable.all(axis=1)
        if bad_rows.any():
            ax.plot(np.full(bad_rows.sum(), 0.02 * max(np.nanmax(np.r_[hi, 1]), 1)), d[bad_rows], "x",
                    color=UNUSABLE, label="Row with unusable windows")
        ax.invert_yaxis()
        style_axes(ax, "Shared Band vs. Depth", "Frequency (MHz)", "Depth (mm)")
        style_legend(ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3))

    def _on_click(self, event) -> None:
        if self._result is None or event.inaxes is not getattr(self, "_ax_grid", None) or event.xdata is None:
            return
        x, z = self._result.grid.centres_mm(self._axial, self._lateral)
        k = int(np.argmin((x - event.xdata) ** 2 + (z - event.ydata) ** 2))
        self._sel = np.unravel_index(k, x.shape)
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
