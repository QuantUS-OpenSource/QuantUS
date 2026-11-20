"""Power Guided ROI screen: first-pass power check with the correlation-length window, then draw the
ROI with a live band check (Python version of DrawROIWithBand.m). The window comes from the
correlation lengths (computed automatically; 'Adjust window sizes...' opens their settings)."""

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QFrame, QPushButton, QLineEdit, QMessageBox, QDoubleSpinBox,
    QComboBox, QLabel, QCheckBox, QSlider, QScrollArea,
)

from engines.qus.quantus.seg_loading.correlation_length import (
    NyquistError, calc_first_pass_from_image, intersect_first_pass, positions_from_image,
)
from engines.qus.quantus.seg_loading.correlation_length.first_pass import (
    ABOVE_NOISE_DB, MIN_SIGNAL_POWER_DB, with_usable_level,
)
from engines.qus.quantus.seg_loading.nakagami_confidence import VESSEL_THRESHOLD, nakagami_confidence_from_image
from engines.qus.quantus.seg_loading.correlation_length.roi import check_sector_roi, save_roi_pkl, sector_roi_from_box
from src.qus.window_roi import POWER_GUIDED_ROI
from .common import (
    BG, MUTED, ROI, SIDEBAR_WIDTH, TEXT, UNUSABLE, USABLE, Worker, bmode_db, error_label, first_frame, heading,
    hint, results_box, separator, stop_worker, style_axes, style_legend,
)

NOTE_YELLOW = "#ffd84d"   # "quick power check only" note
RIGHT_TITLE = "Power Spectra vs Frequency vs Depth\n(Laterally Averaged, Entire Image)"
VESSEL = "#3fe0ff"        # vessel outline (poor Nakagami fit, darker)
BRIGHT = "#ff9f43"        # bright-reflector outline (poor Nakagami fit, brighter)


class RoiBandPage(QWidget):
    """First pass on reference + sample, usable region on the sample B-mode, box ROI + band check."""

    accepted = pyqtSignal(object)        # dict: seg_path, roi, first pass summary
    back_requested = pyqtSignal()
    adjust_requested = pyqtSignal()      # open the correlation-length / first-pass settings
    first_pass_done = pyqtSignal(object) # FirstPassIntersect (usable depths -> window depth range)

    def __init__(self, image_data, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image = image_data
        rf = first_frame(image_data.rf_data)
        self._axial, self._lateral = positions_from_image(image_data, *rf.shape)
        self._bmode = bmode_db(rf)
        self._intersect = None
        self._roi = None
        self._roi_extents = None
        self._windows = None
        self.correlation_settings: Optional[dict] = None   # correlation page settings, saved with the ROI
        self._job = None   # settings of the first pass that is running / last ran
        self._worker: Optional[Worker] = None
        self._selector: Optional[RectangleSelector] = None
        self._naka = None                 # NakagamiConfidence of the sample frame (vessels)
        self._naka_error: Optional[str] = None
        self._vessel_artists = []
        self._build_ui()
        self._naka_worker: Optional[Worker] = Worker(lambda: nakagami_confidence_from_image(self._image), self)
        self._naka_worker.done.connect(self._on_nakagami)
        self._naka_worker.failed.connect(self._on_nakagami_failed)
        self._naka_worker.start()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        side_frame = QFrame()
        side_frame.setFixedWidth(340)
        side = QVBoxLayout(side_frame)
        side.addWidget(heading("Guided ROI"))
        side.addWidget(hint(f"Sample: {self._image.scan_name}\nReference: {self._image.phantom_name}"))
        side.addWidget(separator())
        self._window_box = results_box()
        self._window_box.setText("Computing correlation lengths on the reference phantom...")
        side.addWidget(self._window_box)
        self._adjust_button = QPushButton("Adjust window sizes...")
        self._adjust_button.setToolTip("Correlation-length settings and plots, first-pass depth step, save location")
        side.addWidget(self._adjust_button)

        # first-pass step / rerun / save path: shown in the Adjust dialog (see advanced_widget)
        self._advanced = QWidget()
        adv = QFormLayout(self._advanced)
        self._step = QDoubleSpinBox()
        self._step.setRange(0.01, 20)
        self._step.setDecimals(2)
        self._step.setSingleStep(0.05)
        self._step.setSuffix(" mm")
        self._step.setKeyboardTracking(False)
        adv.addRow("First-pass depth step:", self._step)
        self._rerun = QPushButton("Rerun first pass")
        adv.addRow(self._rerun)
        self._save_path = QLineEdit(str(self._default_save_path()))
        self._save_path.setToolTip("If the file exists, a numbered name is used (_2, _3, ...)")
        adv.addRow("Save ROI to:", self._save_path)

        form = QFormLayout()
        self._spectrum_src = QComboBox()
        self._spectrum_src.addItems(["Sample", "Reference"])
        self._spectrum_src.setToolTip("Power spectrum shown behind the bandwidth plot (all lines averaged, dB)")
        form.addRow("Spectrum shown:", self._spectrum_src)
        self._show_vessels = QCheckBox("Show possible vessels")
        self._show_vessels.setChecked(True)
        self._show_vessels.setToolTip("Outline regions whose echo amplitudes deviate from a Nakagami fit (KS test) "
                                      "and that are darker than the tissue at that depth")
        form.addRow(self._show_vessels)
        self._show_bright = QCheckBox("Show possible bright reflectors")
        self._show_bright.setChecked(True)
        self._show_bright.setToolTip("Outline regions whose echo amplitudes deviate from a Nakagami fit (KS test) "
                                     "and that are brighter than the tissue at that depth (vessel walls, septa, "
                                     "ligaments): they bias the backscatter up and the Nakagami m down")
        form.addRow(self._show_bright)
        side.addLayout(form)

        # thresholds: draggable
        self._power_label = QLabel()
        self._power_slider = QSlider(Qt.Orientation.Horizontal)
        self._power_slider.setRange(int(ABOVE_NOISE_DB), 30)           # total dB above the noise floor
        self._power_slider.setValue(int(round(ABOVE_NOISE_DB + MIN_SIGNAL_POWER_DB)))
        self._power_slider.setToolTip("A block is usable (magenta) if its peak is at least this far above the "
                                      "noise floor in both sample and reference (default 15 dB). Band edges stay "
                                      f"at the noise floor + {ABOVE_NOISE_DB:g} dB.")
        side.addWidget(self._power_label)
        side.addWidget(self._power_slider)
        self._naka_label = QLabel()
        self._naka_slider = QSlider(Qt.Orientation.Horizontal)
        self._naka_slider.setRange(2, 30)                               # x 0.01
        self._naka_slider.setValue(int(round(VESSEL_THRESHOLD * 100)))
        self._naka_slider.setToolTip("Nakagami fit deviation (KS test), 0-1: the largest difference between a "
                                     "window's echo-amplitude distribution and the fitted Nakagami distribution "
                                     "(Kolmogorov-Smirnov statistic). Above it: possible vessel (darker) or bright "
                                     "reflector (brighter). Default 0.10.")
        side.addWidget(self._naka_label)
        side.addWidget(self._naka_slider)
        self._update_threshold_labels()
        self._fp_box = results_box()
        side.addWidget(self._fp_box)
        self._roi_box = results_box()
        side.addWidget(self._roi_box)
        self._error = error_label()
        side.addWidget(self._error)
        side.addStretch()
        self._accept_button = QPushButton("Accept ROI")
        self._back_button = QPushButton("‹  Back: segmentation type")
        for b in (self._accept_button, self._back_button):
            side.addWidget(b)
        scroll = QScrollArea()            # scrolls only if the window is too short for the sidebar
        scroll.setWidget(side_frame)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(side_frame.width() + 20)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        # drawing instructions right above the image, where the user is looking
        main = QVBoxLayout()
        instructions = QLabel()
        self._instructions = instructions
        instructions.setTextFormat(Qt.TextFormat.RichText)
        instructions.setWordWrap(True)
        instructions.setStyleSheet(f"color: {TEXT}; font-size: 14px; padding: 8px 12px; "
                                   "border: 1px solid rgb(90, 90, 90); border-radius: 6px;")
        main.addWidget(instructions)
        self._update_instructions()
        self._fig = Figure(facecolor=BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        main.addWidget(self._canvas, 1)
        root.addLayout(main, 1)

        self._rerun.clicked.connect(self._run_first_pass)
        self._step.valueChanged.connect(lambda _: self._run_first_pass())
        self._adjust_button.clicked.connect(self.adjust_requested.emit)
        self._spectrum_src.currentTextChanged.connect(lambda _: self._intersect is not None and self._draw())
        self._show_vessels.toggled.connect(lambda _: self._update_vessels())
        self._show_bright.toggled.connect(lambda _: self._update_vessels())
        self._power_slider.valueChanged.connect(self._on_power_slider)
        self._power_slider.sliderReleased.connect(self._apply_power_level)
        self._naka_slider.valueChanged.connect(self._on_naka_slider)
        self._naka_slider.sliderReleased.connect(self._update_vessels)
        self._accept_button.clicked.connect(self._on_accept)
        self._back_button.clicked.connect(self.back_requested.emit)
        # Enter accepts only while the image has focus (not while typing in the step / save path)
        self._canvas.mpl_connect("key_press_event", lambda e: e.key == "enter" and self._on_accept())
        self._accept_button.setEnabled(False)

    # ------------------------------------------------------------------ thresholds (sliders)
    def _vthr(self) -> float:
        """Nakagami fit deviation (KS test) threshold."""
        return self._naka_slider.value() / 100

    def _power_total_db(self) -> float:
        """Usable level: peak at least this many dB above the noise floor."""
        return float(self._power_slider.value())

    def _update_threshold_labels(self) -> None:
        self._power_label.setText(f"Usable signal: peak \u2265 noise floor + <b>{self._power_total_db():.0f} dB</b>")
        self._naka_label.setText(f"Nakagami fit deviation (KS test) threshold: <b>{self._vthr():.2f}</b>")

    def _update_instructions(self) -> None:
        self._instructions.setText(
            f"<span style='color:{USABLE}; font-weight:bold'>Magenta outline</span>: enough signal in sample "
            f"AND reference (peak &ge; noise floor + {self._power_total_db():.0f} dB).&nbsp;&nbsp;<b>Drag a box</b> "
            "on the B-mode to draw the ROI (it follows the beam lines). <b>Drag inside it to move it</b>, drag its "
            "handles to resize; the band check updates when you let go. <b>Accept ROI</b> (or Enter on the image) "
            "to accept. The sliders on the left set both thresholds.<br>"
            f"<span style='color:{VESSEL}; font-weight:bold'>Cyan outline</span>: possible vessel (Nakagami fit "
            f"deviation (KS test) above {self._vthr():.2f}, and darker than the tissue at that depth).<br>"
            f"<span style='color:{BRIGHT}; font-weight:bold'>Orange outline</span>: possible bright reflector (Nakagami "
            f"fit deviation above {self._vthr():.2f}, brighter: vessel wall, septum, ligament). The ROI box shows how "
            "much of the ROI each covers.")

    def _on_naka_slider(self, _value) -> None:
        self._update_threshold_labels()
        self._update_instructions()
        if self._naka_slider.isSliderDown():      # live while dragging: outlines only
            if self._intersect is not None:
                self._draw_vessels()
                self._canvas.draw_idle()
        else:                                      # keyboard / click: full update
            self._update_vessels()

    def _on_power_slider(self, _value) -> None:
        self._update_threshold_labels()
        self._update_instructions()
        if not self._power_slider.isSliderDown():  # keyboard / click; dragging applies on release
            self._apply_power_level()

    def _leveled(self, intersect):
        """The first pass with the slider's usable level (stored signal powers, nothing recomputed)."""
        m = self._power_total_db() - intersect.sample.above_noise_db
        return intersect_first_pass(with_usable_level(intersect.sample, m), with_usable_level(intersect.reference, m))

    def _apply_power_level(self) -> None:
        raw = getattr(self, "_intersect_raw", None)
        if raw is None:
            return
        self._intersect = self._leveled(raw)
        self._show_first_pass_summary()
        try:
            self._draw()
        except Exception as e:
            self.show_error(f"Could not draw the first pass: {e}")

    def _show_first_pass_summary(self) -> None:
        inter = self._intersect
        s, r = inter.sample, inter.reference
        self._fp_box.setText(
            f"<b>First pass</b>: {s.window_length_mm:.1f} mm axial window ({s.window_samples} samples), "
            f"{s.block_width_mm:.1f} mm blocks ({s.block_lines} lines), fs {s.fs_mhz:g} MHz<br>"
            f"Usable blocks (peak &ge; noise + {self._power_total_db():.0f} dB): sample {s.usable_2d.mean():.0%}, "
            f"reference {r.usable_2d.mean():.0%}, <b>both {inter.usable_both_2d.mean():.0%}</b>")

    def _default_save_path(self) -> Path:
        scan = Path(getattr(self._image, "scan_path", "") or ".")
        folder = scan.parent if scan.suffix else scan
        return folder / f"{self._image.scan_name}_FirstPassROI.pkl"

    @property
    def advanced_widget(self) -> QWidget:
        """First-pass step, rerun and save path, for the Adjust dialog."""
        return self._advanced

    def set_window_summary(self, html: str) -> None:
        self._window_box.setText(html)

    def show_window_error(self, message: str) -> None:
        self._window_box.setText(f"<span style='color:{UNUSABLE}'>Window sizes not available: {message}</span>"
                                 "<br>Use 'Adjust window sizes...' to change the settings.")
        self._fp_box.setText("")

    # ------------------------------------------------------------------ first pass
    def set_windows(self, ax_mm: float, lat_mm: float) -> None:
        """Called when the page is shown with the correlation-length window; reruns the first pass
        if the window changed."""
        if self._windows == (ax_mm, lat_mm) and self._intersect is not None:
            return
        self._windows = (ax_mm, lat_mm)
        self._step.blockSignals(True)   # one run below, not one per change
        self._step.setValue(round(0.2 * ax_mm, 2))
        self._step.blockSignals(False)
        self._run_first_pass()

    def _settings(self) -> tuple:
        return (self._windows, round(self._step.value(), 4))

    def _run_first_pass(self) -> None:
        if self._windows is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return  # _on_first_pass reruns if the settings changed meanwhile
        ax_mm, lat_mm = self._windows
        fs = getattr(self._image, "sampling_frequency", None)
        kw = dict(window_length_mm=ax_mm, block_width_mm=lat_mm, ax_step_mm=self._step.value(),
                  sampling_frequency_mhz=float(fs) / 1e6 if fs else None)
        self._job = self._settings()

        def work():
            ref = calc_first_pass_from_image(self._image, "phantom", **kw)
            # same lateral blocks as the reference even if the sample's imaging depth differs
            smp = calc_first_pass_from_image(self._image, "scan", block_lines=ref.block_lines, **kw)
            return intersect_first_pass(smp, ref)

        self._intersect, self._roi = None, None
        self._error.hide()
        self._accept_button.setEnabled(False)
        self._rerun.setEnabled(False)
        self._fp_box.setText(f"Running the first pass ({ax_mm:.1f} mm axial window, {lat_mm:.1f} mm blocks)...")
        self._roi_box.setText("")
        self._worker = Worker(work, self)
        self._worker.done.connect(self._on_first_pass)
        self._worker.failed.connect(self._on_first_pass_failed)
        self._worker.start()

    def _on_first_pass(self, intersect) -> None:
        self._rerun.setEnabled(True)
        if self._job != self._settings():   # window / step changed while this one ran: run again
            self._run_first_pass()
            return
        self._intersect_raw = intersect            # as computed (default level)
        self._intersect = self._leveled(intersect)  # with the slider's usable level
        self._show_first_pass_summary()
        self._roi_box.setText("Drag a box on the B-mode to draw the ROI.")
        try:
            self._draw()
        except Exception as e:  # never let a plotting problem take down the app (exceptions in slots abort)
            self.show_error(f"Could not draw the first pass: {e}")
        self.first_pass_done.emit(intersect)

    def _on_first_pass_failed(self, error) -> None:
        self._rerun.setEnabled(True)
        if self._job != self._settings():
            self._run_first_pass()
            return
        self._fp_box.setText("")
        msg = str(error)
        if isinstance(error, NyquistError):
            msg = "<b>Not usable for spectral analysis</b><br>" + msg
        self._error.setText(msg)
        self._error.show()
        self._fig.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ figure
    def _draw(self) -> None:
        inter = self._intersect
        s = inter.sample
        fig = self._fig
        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.22, left=0.06, right=0.93, top=0.8, bottom=0.2)
        self._ax_b = fig.add_subplot(gs[0])
        self._ax_r = fig.add_subplot(gs[1], sharey=self._ax_b)
        ax = self._ax_b
        x, z = s.block_x, s.block_z

        ax.pcolormesh(self._lateral, self._axial, self._bmode, cmap="gray", vmin=-60, vmax=0, shading="nearest",
                      rasterized=True)
        bad = np.ma.masked_where(inter.usable_both_2d, np.ones_like(x))
        ax.pcolormesh(x, z, bad, cmap=_solid(UNUSABLE), vmin=0, vmax=1, alpha=0.25, shading="nearest",
                      edgecolors="none", linewidth=0, antialiased=False, rasterized=True)
        if np.any(inter.usable_both_2d) and min(x.shape) >= 2:
            # outline of the usable blocks, including along the edges of the evaluated grid
            # (a 1-block-wide or 1-row grid has no 2-D outline; the red shading still shows it)
            px, pz, pm = _padded_grid(x, z, inter.usable_both_2d)
            ax.contour(px, pz, pm.astype(float), [0.5], colors=[USABLE], linewidths=2)
        gx = np.concatenate([x[0], x[:, -1], x[-1, ::-1], x[::-1, 0]])
        gz = np.concatenate([z[0], z[:, -1], z[-1, ::-1], z[::-1, 0]])
        ax.plot(gx, gz, color="white", linestyle=":", linewidth=1, label="Evaluated region")
        self._outline, = ax.plot([], [], color=ROI, linewidth=2.5, label="ROI")
        ax.plot([], [], color=USABLE, linewidth=2, label="Usable in sample & reference")
        ax.fill([], [], color=UNUSABLE, alpha=0.25, label="Not usable")
        ax.set_aspect("equal", adjustable="datalim")
        style_axes(ax, "Sample B-Mode: Drag a Box to Draw the ROI", "Lateral (mm)", "Depth (mm)")
        self._vessel_artists = []
        self._draw_vessels()

        axr = self._ax_r
        d = s.depths
        # power vs frequency vs depth (all lines averaged), absolute dB, 60 dB below the maximum
        src = inter.reference if self._spectrum_src.currentText() == "Reference" else s
        with np.errstate(divide="ignore", invalid="ignore"):
            p_db = 10 * np.log10(src.mean_ps)
        top = float(np.nanmax(p_db[np.isfinite(p_db)])) if np.isfinite(p_db).any() else 0.0
        mesh = axr.pcolormesh(src.freq_axis, d, p_db, cmap="viridis", vmin=top - 60, vmax=top, shading="nearest",
                              rasterized=True)
        cbar = fig.colorbar(mesh, ax=axr, fraction=0.06, pad=0.02)
        cbar.set_label(f"{self._spectrum_src.currentText()} power (dB)", color=TEXT, fontsize=8)
        cbar.ax.tick_params(colors=TEXT, labelsize=8)
        lo = np.where(inter.usable_both, inter.f_low_both, np.nan)
        hi = np.where(inter.usable_both, inter.f_high_both, np.nan)
        axr.plot(lo, d, color="white", linestyle="--", linewidth=1.3, label="Shared band, all lines")
        axr.plot(hi, d, color="white", linestyle="--", linewidth=1.3)
        self._row_lo, = axr.plot([], [], color=USABLE, linewidth=2.5, label="Shared band, ROI")
        self._row_hi, = axr.plot([], [], color=USABLE, linewidth=2.5)
        self._band_box, = axr.plot([], [], color="white", linewidth=2.5, label="ROI band")
        self._bad_rows, = axr.plot([], [], "x", color=UNUSABLE, markersize=8, mew=1.5, label="Unusable row")
        fmax = np.nanmax(np.r_[s.f_high, inter.reference.f_high, 0])
        axr.set_xlim(0, min(1.25 * fmax, s.freq_axis[-1]) if fmax > 0 else s.freq_axis[-1])
        style_axes(axr, RIGHT_TITLE, "Frequency (MHz)", "Depth (mm)")
        axr.grid(False)
        axr.tick_params(labelleft=True)
        style_legend(axr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8))
        ax.set_ylim(float(np.nanmax(self._axial)), float(min(np.nanmin(self._axial), 0)))

        self._selector = RectangleSelector(
            ax, self._on_box, useblit=True, button=[1], minspanx=0.5, minspany=0.5, spancoords="data",
            interactive=True, drag_from_anywhere=True,   # drag inside the box to move it
            props=dict(edgecolor="white", facecolor="none", linewidth=1, linestyle="--"))
        fig.suptitle(f"First Pass\n{s.window_length_mm:.1f} mm window, {s.block_lines}-line blocks "
                     f"({s.block_width_mm:.1f} mm)\nUsable in sample & reference: {inter.usable_both_2d.mean():.0%} "
                     f"of blocks", color=TEXT, fontsize=11, y=0.995)
        fig.text(0.5, 0.905, "Quick power check only: spectra are recomputed with the final window settings "
                 "for parameter estimation.", color=NOTE_YELLOW, fontsize=10.5, fontweight="bold", ha="center",
                 va="top")
        if self._roi_extents is not None:            # redraw (e.g. spectrum switched): keep the ROI
            self._selector.extents = self._roi_extents
            self._on_box(None, None)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ vessels (Nakagami confidence)
    def _on_nakagami(self, result) -> None:
        self._naka = result
        self._update_vessels()

    def _on_nakagami_failed(self, error) -> None:
        self._naka_error = str(error)
        self._update_vessels()

    def _structure_masks(self):
        """(vessel, bright) masks: poor Nakagami fit and darker / brighter than the tissue at the same
        depth, inside the first pass's evaluated depths (leaves out the transducer ring-down)."""
        if self._naka is None:
            return None, None
        thr = self._vthr()
        vessel = self._naka.vessel_mask(thr, darker_only=True)
        bright = self._naka.bright_mask(thr)
        if self._intersect is not None:
            z = self._intersect.sample.block_z
            inside = (self._axial >= np.nanmin(z)) & (self._axial <= np.nanmax(z))
            vessel, bright = vessel & inside, bright & inside
        return vessel, bright

    def _vessel_mask(self):
        return self._structure_masks()[0]

    def _draw_vessels(self) -> None:
        """Vessel / bright-reflector outlines on the B-mode (if shown and computed) and the legend."""
        ax = getattr(self, "_ax_b", None)
        if ax is None:
            return
        for art in self._vessel_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._vessel_artists = []
        vessel, bright = self._structure_masks()
        # short legend labels (the B-mode legend must fit under the plot); the criterion - Nakagami fit
        # deviation (KS test) above the threshold - is in the instructions and the threshold setting
        layers = ((self._show_vessels, vessel, VESSEL, "Possible vessel"),
                  (self._show_bright, bright, BRIGHT, "Possible bright reflector"))
        handles, labels = ax.get_legend_handles_labels()
        handles = [h for h, l in zip(handles, labels) if not l.startswith("Possible ")]
        for box, mask, colour, label in layers:
            if not box.isChecked():
                continue
            if mask is not None and mask.any():
                self._vessel_artists.append(
                    ax.contour(self._lateral, self._axial, mask.astype(float), [0.5], colors=[colour], linewidths=1.3))
            proxy, = ax.plot([], [], color=colour, linewidth=1.3, label=label)
            self._vessel_artists.append(proxy)
            handles.append(proxy)
        style_legend(ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8))

    def _update_vessels(self) -> None:
        if self._intersect is None:
            return
        self._draw_vessels()
        if self._roi is not None and self._roi.band is not None:
            self._on_box(None, None)          # refresh the ROI box (vessel share)
        else:
            self._canvas.draw_idle()

    def _vessel_line(self, roi) -> str:
        """ROI box lines: share of the ROI that is possible vessel / possible bright reflector."""
        if self._naka_error:
            return f"<br><span style='color:{MUTED}'>Vessel check not available: {self._naka_error}</span>"
        vessel, bright = self._structure_masks()
        if vessel is None:
            return f"<br><span style='color:{MUTED}'>Vessel / bright-reflector check: computing...</span>"
        (r1, r2), (l1, l2) = roi.rows, roi.lines
        out = ""
        for mask, name in ((vessel, "Possible vessel"), (bright, "Possible bright reflector")):
            part = mask[r1:r2 + 1, l1:l2 + 1]
            share = float(part.mean()) if part.size else 0.0
            colour = NOTE_YELLOW if share > 0 else "white"
            out += f"<br><span style='color:{colour}'>{name}: {share:.0%} of the ROI</span>"
        return out   # criteria (Nakagami fit deviation threshold, darker / brighter) are in the legend / instructions

    # ------------------------------------------------------------------ ROI
    def _on_box(self, press, release) -> None:
        if self._intersect is None:
            return
        x0, x1, z0, z1 = self._selector.extents
        if abs(x1 - x0) < 1e-6 or abs(z1 - z0) < 1e-6:   # a click, not a drag: keep the previous ROI
            if self._roi_extents is not None:
                self._selector.extents = self._roi_extents
                self._canvas.draw_idle()
            return
        self._roi_extents = (x0, x1, z0, z1)
        roi = check_sector_roi(self._intersect, sector_roi_from_box(self._axial, self._lateral, x0, x1, z0, z1))
        self._roi = roi
        self._outline.set_data(np.r_[roi.x_mm, roi.x_mm[0]], np.r_[roi.z_mm, roi.z_mm[0]])
        b = roi.band
        axr = self._ax_r
        if b is None:
            for h in (self._row_lo, self._row_hi, self._band_box, self._bad_rows):
                h.set_data([], [])
            axr.set_title(f"{RIGHT_TITLE}\nROI: no first-pass blocks inside", color=UNUSABLE, fontsize=10)
            self._roi_box.setText(f"<span style='color:{UNUSABLE}'>No first-pass blocks inside the ROI - make it "
                                  "larger.</span>")
            self._accept_button.setEnabled(False)
            self._canvas.draw_idle()
            return
        self._row_lo.set_data(b.f_low_row, b.depths)
        self._row_hi.set_data(b.f_high_row, b.depths)
        zt, zb = b.depths.min(), b.depths.max()
        if np.isfinite(b.bw_mhz):
            self._band_box.set_data([b.f_low_roi, b.f_high_roi, b.f_high_roi, b.f_low_roi, b.f_low_roi],
                                    [zt, zt, zb, zb, zt])
        else:
            self._band_box.set_data([], [])
        bad = ~b.row_usable
        bx = np.where(np.isnan(b.f_low_row[bad]), 0.02 * axr.get_xlim()[1], b.f_low_row[bad])
        self._bad_rows.set_data(bx, b.depths[bad])
        ok = b.all_rows_usable and np.isfinite(b.bw_mhz)
        line1 = (f"ROI band {b.f_low_roi:.2f} - {b.f_high_roi:.2f} MHz ({b.bw_mhz:.2f} MHz)" if np.isfinite(b.bw_mhz)
                 else "No shared band across the usable rows")
        line3 = (f"All {b.depths.size} depth rows usable" if ok
                 else f"{int(bad.sum())} of {b.depths.size} depth rows NOT usable")
        axr.set_title(f"{RIGHT_TITLE}\n{line1}; {line3[0].lower() + line3[1:]}", color=TEXT if ok else UNUSABLE,
                      fontsize=10)
        (r1, r2), (l1, l2) = roi.rows, roi.lines
        colour = "white" if ok else UNUSABLE
        self._roi_box.setText(
            f"<b>ROI</b>: lines {l1}-{l2}, samples {r1}-{r2} ({roi.depth_range_mm[0]:.1f}-"
            f"{roi.depth_range_mm[1]:.1f} mm)<br><b>{line1}</b><br>"
            f"f<sub>low</sub> set at {b.f_low_limit_depth:.1f} mm, f<sub>high</sub> at {b.f_high_limit_depth:.1f} mm<br>"
            f"Min signal power above band level: sample {np.nanmin(b.signal_power_s1):.1f} dB, reference "
            f"{np.nanmin(b.signal_power_l1):.1f} dB (need {self._intersect.sample.min_signal_power_db:g})<br>"
            f"<span style='color:{colour}'>{line3}</span>" + self._vessel_line(roi))
        self._accept_button.setEnabled(True)
        self._canvas.draw_idle()

    def _on_accept(self) -> None:
        if not self.isVisible() or not self._accept_button.isEnabled():
            return
        roi = self._roi
        if roi is None or roi.band is None:
            self._error.setText("Draw an ROI with first-pass blocks inside it first.")
            self._error.show()
            return
        b = roi.band
        if not b.all_rows_usable or not np.isfinite(b.bw_mhz):
            answer = QMessageBox.question(
                self, "ROI check", "The ROI contains depth rows that are not usable in the sample and reference "
                "(or has no shared band). Keep it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        target = _unused_path(Path(self._save_path.text().strip()))
        try:
            s = self._intersect.sample
            path = save_roi_pkl(target, self._image, roi, frame=0, extra={
                "First Pass Window mm": (s.window_length_mm, s.block_width_mm),
                "First Pass Block Lines": s.block_lines,
                "Correlation Settings": dict(self.correlation_settings or {}),
                "Usable Level dB Above Noise": self._power_total_db(),
                "Nakagami Fit Deviation Threshold": self._vthr(),
            })
        except Exception as e:
            self._error.setText(f"Could not save the ROI: {e}")
            self._error.show()
            return
        self._error.hide()
        self._accept_button.setEnabled(False)   # one accept per visit (loading opens the preview)
        self.accepted.emit({"seg_path": str(path), "roi": roi, "intersect": self._intersect,
                            "seg_type": POWER_GUIDED_ROI})

    def showEvent(self, event) -> None:
        """Coming back (e.g. from the preview): the current ROI can be accepted again."""
        super().showEvent(event)
        self._accept_button.setEnabled(self._intersect is not None and self._roi is not None
                                       and self._roi.band is not None)

    def shutdown(self) -> None:
        stop_worker(self._worker)
        stop_worker(self._naka_worker)

    def show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.show()


def _unused_path(path: Path) -> Path:
    """path, or path with _2, _3, ... before the suffix if it already exists (never overwrite an ROI)."""
    if not path.exists():
        return path
    n = 2
    while (candidate := path.with_name(f"{path.stem}_{n}{path.suffix}")).exists():
        n += 1
    return candidate


def _padded_grid(x: np.ndarray, z: np.ndarray, mask: np.ndarray):
    """Block-centre grid extended by one block on every side (linear extrapolation) with the mask
    False there, so a contour at 0.5 closes around usable blocks on the grid edge too (halfway
    to the next block, i.e. about the edge of the block's coverage)."""
    def pad(a):
        a = np.pad(a, 1, mode="edge").astype(float)
        a[0, :] = 2 * a[1, :] - a[2, :]
        a[-1, :] = 2 * a[-2, :] - a[-3, :]
        a[:, 0] = 2 * a[:, 1] - a[:, 2]
        a[:, -1] = 2 * a[:, -2] - a[:, -3]
        return a

    if min(x.shape) < 2:
        return x, z, mask
    return pad(x), pad(z), np.pad(mask, 1, constant_values=False)


def _solid(color: str):
    from matplotlib.colors import ListedColormap

    return ListedColormap([color])
