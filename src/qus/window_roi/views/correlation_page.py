"""Page 1 of the Window Sizes & ROI step: speckle correlation lengths on the reference phantom ->
recommended QUS window (quantus.seg_loading.correlation_length)."""

from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QFrame, QComboBox, QDoubleSpinBox, QCheckBox, QPushButton,
)

from engines.qus.quantus.seg_loading.correlation_length import PROBE_PRESETS, correlation_lengths_from_image, positions_from_image
from engines.qus.quantus.seg_loading.correlation_length.correlation import _central_region, _mround
from .common import (
    AXIAL, BG, LATERAL, MUTED, REGION, SIDEBAR_WIDTH, TEXT, Worker, bmode_db, error_label, first_frame, heading, hint,
    results_box, separator, stop_worker, style_axes, style_legend,
)

CUSTOM = "Custom"


class CorrelationPage(QWidget):
    """Settings sidebar + depth-aligned plots of the correlation lengths and recommended windows."""

    next_requested = pyqtSignal(object)   # CorrelationLengths
    back_requested = pyqtSignal()

    def __init__(self, image_data, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image = image_data
        self._result = None
        self._worker: Optional[Worker] = None
        self._pending = False
        self._gs = self._ax0 = None
        ref = first_frame(image_data.get_phantom_frames() if hasattr(image_data, "get_phantom_frames")
                          else image_data.phantom_rf_data)
        self._axial, self._lateral = positions_from_image(image_data, *ref.shape)
        self._bmode = bmode_db(ref)
        self._build_ui()
        self._set_custom_defaults()
        self._compute()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        side_frame = QFrame()
        side_frame.setFixedWidth(SIDEBAR_WIDTH)
        side = QVBoxLayout(side_frame)
        side.addWidget(heading("Window Sizes"))
        side.addWidget(hint(f"Reference phantom: {self._image.phantom_name}\n"
                            "Speckle correlation lengths vs. depth give the QUS window size "
                            "(factor x mean length, the largest over the depth range)."))
        side.addWidget(separator())

        form = QFormLayout()
        self._preset = QComboBox()
        self._preset.addItems([CUSTOM] + list(PROBE_PRESETS))
        form.addRow("Probe preset:", self._preset)

        def spin(lo, hi, step, decimals, suffix=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            return s

        max_depth = float(np.nanmax(self._axial))
        self._top = spin(0, max_depth, 0.5, 1, " mm")
        self._bottom = spin(0, max_depth, 0.5, 1, " mm")
        self._win_len = spin(0.1, 100, 0.5, 2, " mm")
        self._win_wid = spin(0.1, 200, 0.5, 2, " mm")
        self._threshold = spin(0.05, 0.95, 0.05, 2)
        self._central = QCheckBox("Measure central part of image only")
        self._fraction = spin(5, 100, 5, 0, " % of width")
        central_tip = ("Measurement windows are placed every half window across the central part of the image "
                       "(e.g. 50% = the middle half of the A-lines), away from the image edges / sector walls, "
                       "and averaged with the frames. Unchecked: only the centre A-line, as the MATLAB.")
        self._central.setToolTip(central_tip)
        self._fraction.setToolTip(central_tip)
        self._ax_factor = spin(0.5, 50, 0.5, 1, " x")
        self._lat_factor = spin(0.5, 50, 0.5, 1, " x")
        form.addRow("Windows from:", self._top)
        form.addRow("Windows to:", self._bottom)
        form.addRow("Measurement window (axial):", self._win_len)
        form.addRow("Measurement window (lateral):", self._win_wid)
        form.addRow("Correlation threshold:", self._threshold)
        form.addRow(self._central)
        form.addRow("Central part:", self._fraction)
        form.addRow("Axial window factor:", self._ax_factor)
        form.addRow("Lateral window factor:", self._lat_factor)
        side.addLayout(form)

        self._compute_button = QPushButton("Recompute")
        side.addWidget(self._compute_button)
        self._results = results_box()
        side.addWidget(self._results)
        self._error = error_label()
        side.addWidget(self._error)
        side.addStretch()
        self._next_button = QPushButton("Next: first pass && ROI  ›")
        self._back_button = QPushButton("‹  Back: segmentation type")
        for b in (self._next_button, self._back_button):
            side.addWidget(b)
        root.addWidget(side_frame)

        self._fig = Figure(facecolor=BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.mpl_connect("resize_event", self._fit_bmode_width)
        root.addWidget(self._canvas, 1)

        self._preset.currentTextChanged.connect(self._on_preset)
        self._central.toggled.connect(self._fraction.setEnabled)
        self._compute_button.clicked.connect(self._compute)
        self._next_button.clicked.connect(lambda: self._result is not None and self.next_requested.emit(self._result))
        self._back_button.clicked.connect(self.back_requested.emit)

    def _set_custom_defaults(self) -> None:
        depth = float(np.nanmax(self._axial))
        self._top.setValue(min(5.0, depth / 4))
        self._bottom.setValue(depth)
        length = round(min(10.0, max(1.0, depth / 6)), 1)
        self._win_len.setValue(length)
        self._win_wid.setValue(length)
        self._threshold.setValue(0.2)
        self._central.setChecked(True)
        self._fraction.setValue(50)
        self._ax_factor.setValue(10.0)
        self._lat_factor.setValue(5.0)

    def _on_preset(self, name: str) -> None:
        if name == CUSTOM:
            return
        p = PROBE_PRESETS[name]
        top, bottom = p.get("depth_range_mm", (None, None))
        if top is not None:
            self._top.setValue(top)
        if bottom is not None:
            self._bottom.setValue(min(bottom, self._bottom.maximum()))
        self._win_len.setValue(p["window_length_mm"])
        self._win_wid.setValue(p["window_width_mm"])
        self._threshold.setValue(p.get("threshold", 0.5))
        frac = p.get("lateral_fraction")
        self._central.setChecked(frac is not None)
        if frac is not None:
            self._fraction.setValue(100 * frac)
        self._compute()

    def preset_name(self) -> Optional[str]:
        name = self._preset.currentText()
        return None if name == CUSTOM else name

    def settings(self) -> dict:
        return dict(window_length_mm=self._win_len.value(), window_width_mm=self._win_wid.value(),
                    depth_range_mm=(self._top.value(), self._bottom.value()), threshold=self._threshold.value(),
                    lateral_fraction=self._fraction.value() / 100 if self._central.isChecked() else None,
                    ax_window_factor=self._ax_factor.value(), lat_window_factor=self._lat_factor.value())

    # ------------------------------------------------------------------ compute
    def _compute(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._pending = True   # settings changed mid-run (e.g. preset picked): rerun when it ends
            return
        self._pending = False
        kw = self.settings()
        self._error.hide()
        self._next_button.setEnabled(False)
        self._compute_button.setEnabled(False)
        self._results.setText("Computing correlation lengths...")
        self._worker = Worker(lambda: correlation_lengths_from_image(self._image, source="phantom", **kw), self)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result) -> None:
        self._compute_button.setEnabled(True)
        if self._pending:
            self._compute()
            return
        self._result = result
        w = result.recommended_window()
        ok = np.isfinite(w["ax_window_mm"]) and np.isfinite(w["lat_window_mm"])
        self._next_button.setEnabled(bool(ok))
        if ok:
            self._results.setText(
                f"<b>Suggested Window Size for First Pass:</b><br>"
                f"<b>Axial window</b>: {w['ax_window_mm']:.1f} mm ({w['ax_window_samples']:.0f} samples, "
                f"max at {w['ax_mm_at_depth_mm']:.1f} mm)<br>"
                f"<b>Lateral window</b>: {w['lat_window_mm']:.1f} mm (max at {w['lat_mm_at_depth_mm']:.1f} mm); "
                f"{w['lat_window_lines']:.0f} A-lines (max at {w['lat_at_depth_mm']:.1f} mm)<br>"
                f"<span style='color:{MUTED}'>{result._describe_data()}, depths "
                f"{w['depth_range_mm'][0]:.1f}-{w['depth_range_mm'][1]:.1f} mm</span>")
        else:
            self._results.setText("No recommended window: the correlation never dropped below the threshold.")
        try:
            self._plot()
        except Exception as e:  # never let a plotting problem take down the app (exceptions in slots abort)
            self._error.setText(f"Could not draw the plots: {e}")
            self._error.show()

    def _on_failed(self, error) -> None:
        self._compute_button.setEnabled(True)
        if self._pending:
            self._compute()
            return
        self._results.setText("")
        self._error.setText(str(error))
        self._error.show()

    # ------------------------------------------------------------------ plots
    def _region_outline(self, r):
        """Outline (mm) of the region the measurement windows covered: depth range x lines."""
        n_samples, n_lines = self._axial.shape
        mid0 = _mround(n_lines / 2) - 1
        if r.lateral_fraction is None:   # centre A-line only: the widest window around it
            half_lat = int((np.max(r.window_lines) - 1) // 2)
            first, last = max(mid0 - half_lat, 0), min(mid0 + half_lat, n_lines - 1)
        else:
            first, last = _central_region(n_lines, mid0, r.lateral_fraction)
        centre = self._axial[:, mid0]
        half = (r.window_samples[0] - 1) // 2
        r1 = int(np.argmin(np.abs(centre - r.depths_mm.min()))) - half
        r2 = int(np.argmin(np.abs(centre - r.depths_mm.max()))) + half
        r1, r2 = max(r1, 0), min(r2, n_samples - 1)
        cols, rows = np.arange(first, last + 1), np.arange(r1, r2 + 1)
        ax, lat = self._axial, self._lateral
        x = np.concatenate([lat[r1, cols], lat[rows, last], lat[r2, cols[::-1]], lat[rows[::-1], first]])
        z = np.concatenate([ax[r1, cols], ax[rows, last], ax[r2, cols[::-1]], ax[rows[::-1], first]])
        return x, z

    def _plot(self) -> None:
        r = self._result
        fig = self._fig
        fig.clear()
        gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.08, left=0.05, right=0.985,
                              top=0.82, bottom=0.12)
        ax0 = fig.add_subplot(gs[0])
        axs = [fig.add_subplot(gs[i], sharey=ax0) for i in (1, 2, 3)]
        d = r.depths_mm

        ax0.pcolormesh(self._lateral, self._axial, self._bmode, cmap="gray", vmin=-60, vmax=0, shading="nearest",
                       rasterized=True)
        x, z = self._region_outline(r)
        ax0.plot(np.r_[x, x[0]], np.r_[z, z[0]], color=REGION, linestyle="--", linewidth=1.5,
                 label="Measured region (window edges)")
        self._region_half_width = float(np.nanmax(np.abs(x)))
        self._gs, self._ax0 = gs, ax0
        self._fit_bmode_width()
        style_axes(ax0, "Reference Phantom", "Lateral (mm)", "Depth (mm)")
        style_legend(ax0.legend(loc="lower center", fontsize=8))

        for ax, mean, sd, color, name, unit_line in (
            (axs[0], r.lat_mean_lines, r.lat_std_lines, LATERAL, "Lateral Width", "A-lines"),
            (axs[1], r.ax_mean_samples, r.ax_std_samples, AXIAL, "Axial Length", "samples"),
        ):
            ax.fill_betweenx(d, mean - sd, mean + sd, color=color, alpha=0.25, linewidth=0, label="± 1 SD")
            ax.plot(mean, d, color=color, linewidth=2, marker="o", markersize=3, label="Mean")
            style_axes(ax, name, f"{name.capitalize()} ({unit_line})")
            style_legend(ax.legend(loc="lower right", fontsize=8))
            ax.tick_params(labelleft=False)
        axs[0].axvline(1, color=MUTED, linestyle=":", linewidth=1)

        w = r.recommended_window()
        ax = axs[2]
        ax.step(r.recommended_lat_window_mm, d, where="mid", color=LATERAL, linewidth=2,
                label=f"Lateral ({r.lat_window_factor:g}x)")
        ax.step(r.recommended_ax_window_mm, d, where="mid", color=AXIAL, linewidth=2,
                label=f"Axial ({r.ax_window_factor:g}x)")
        for value, color in ((w["lat_window_mm"], LATERAL), (w["ax_window_mm"], AXIAL)):
            if np.isfinite(value):
                ax.axvline(value, color=color, linestyle="--", linewidth=1.2)
                ax.text(value, 0.99, f"{value:.1f} mm ", color=color, fontsize=9, fontweight="bold",
                        transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right")
        style_axes(ax, "Recommended Window", "Window (mm)")
        style_legend(ax.legend(loc="lower right", fontsize=8))
        ax.tick_params(labelleft=False)
        ax.set_xlim(left=0)

        ax0.set_ylim(float(np.nanmax(self._axial)), float(min(np.nanmin(self._axial), 0)))
        fig.suptitle(f"Speckle Correlation Lengths\n"
                     f"Threshold {r.threshold:g}, measurement window {r.window_length_mm:g} x {r.window_width_mm:g} mm\n"
                     f"{r._describe_data()}", color=TEXT, fontsize=11)
        self._canvas.draw_idle()

    def _fit_bmode_width(self, _event=None) -> None:
        """Size the B-mode panel for equal mm scaling with the same height (depth axis) as the other
        panels: the whole image width if it fits in ~half the figure, otherwise the measured region
        plus 40% either side (so the image visibly extends past the outline)."""
        gs = getattr(self, "_gs", None)
        if gs is None or self._ax0 is None:
            return
        w, h = self._fig.get_size_inches() * self._fig.dpi
        plot_h = (gs.top - gs.bottom) * h
        avail = (gs.right - gs.left) * w
        x0, x1 = float(np.nanmin(self._lateral)), float(np.nanmax(self._lateral))
        z_span = float(np.nanmax(self._axial) - min(np.nanmin(self._axial), 0))
        cap = 0.55 * avail
        fit_span = cap * z_span / max(plot_h, 1.0)      # widest lateral range that fits at equal mm
        if (x1 - x0) > fit_span:
            half = max(fit_span / 2, 1.1 * self._region_half_width)
            x0, x1 = max(x0, -half), min(x1, half)
        w0 = min(plot_h * (x1 - x0) / z_span, cap)
        rest = max((avail - w0) / 3, 1.0)
        gs.set_width_ratios([w0, rest, rest, rest])
        for ax in self._fig.axes:   # existing axes don't follow new ratios on their own
            spec = ax.get_subplotspec()
            if spec is not None:
                ax.set_position(spec.get_position(self._fig))
        self._ax0.set_xlim(x0, x1)
        self._ax0.set_aspect("auto")   # equal mm by construction (up to the cap)
        if _event is not None:
            self._canvas.draw_idle()

    @property
    def result(self):
        return self._result

    def shutdown(self) -> None:
        stop_worker(self._worker)
