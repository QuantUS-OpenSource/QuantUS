"""Page 2 of the Window Sizes & ROI step: first-pass power check with the correlation-length window,
then draw the ROI with a live band check (Python version of DrawROIWithBand.m)."""

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QFrame, QPushButton, QLineEdit, QMessageBox, QDoubleSpinBox,
    QComboBox,
)

from engines.qus.quantus.seg_loading.correlation_length import (
    NyquistError, calc_first_pass_from_image, intersect_first_pass, positions_from_image,
)
from engines.qus.quantus.seg_loading.correlation_length.roi import check_sector_roi, save_roi_pkl, sector_roi_from_box
from src.qus.window_roi import POWER_GUIDED_ROI
from .common import (
    BG, MUTED, ROI, SIDEBAR_WIDTH, TEXT, UNUSABLE, USABLE, Worker, bmode_db, error_label, first_frame, heading,
    hint, results_box, separator, stop_worker, style_axes, style_legend,
)


class RoiBandPage(QWidget):
    """First pass on reference + sample, usable region on the sample B-mode, box ROI + band check."""

    accepted = pyqtSignal(object)        # dict: seg_path, roi, first pass summary
    back_requested = pyqtSignal()

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
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        side_frame = QFrame()
        side_frame.setFixedWidth(340)
        side = QVBoxLayout(side_frame)
        side.addWidget(heading("First Pass & ROI"))
        side.addWidget(hint(f"Sample: {self._image.scan_name}\nReference: {self._image.phantom_name}\n"
                            "Magenta: enough signal in sample AND reference (peak >= noise floor + 15 dB). "
                            "Drag a box on the B-mode to draw the ROI (it follows the beam lines); the band "
                            "check updates when you let go. 'Accept ROI' (or Enter on the image) to accept."))
        side.addWidget(separator())
        form = QFormLayout()
        self._step = QDoubleSpinBox()
        self._step.setRange(0.01, 20)
        self._step.setDecimals(2)
        self._step.setSingleStep(0.05)
        self._step.setSuffix(" mm")
        self._step.setKeyboardTracking(False)
        form.addRow("First-pass depth step:", self._step)
        self._spectrum_src = QComboBox()
        self._spectrum_src.addItems(["Sample", "Reference"])
        self._spectrum_src.setToolTip("Power spectrum shown behind the bandwidth plot (all lines averaged, dB)")
        form.addRow("Spectrum shown:", self._spectrum_src)
        side.addLayout(form)
        self._rerun = QPushButton("Rerun first pass")
        side.addWidget(self._rerun)
        self._fp_box = results_box()
        side.addWidget(self._fp_box)
        self._roi_box = results_box()
        side.addWidget(self._roi_box)
        self._error = error_label()
        side.addWidget(self._error)
        side.addStretch()
        save_form = QFormLayout()
        self._save_path = QLineEdit(str(self._default_save_path()))
        save_form.addRow("Save ROI to:", self._save_path)
        side.addLayout(save_form)
        self._accept_button = QPushButton("Accept ROI")
        self._back_button = QPushButton("‹  Back: window sizes")
        for b in (self._accept_button, self._back_button):
            side.addWidget(b)
        root.addWidget(side_frame)

        self._fig = Figure(facecolor=BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        root.addWidget(self._canvas, 1)

        self._rerun.clicked.connect(self._run_first_pass)
        self._spectrum_src.currentTextChanged.connect(lambda _: self._intersect is not None and self._draw())
        self._accept_button.clicked.connect(self._on_accept)
        self._back_button.clicked.connect(self.back_requested.emit)
        # Enter accepts only while the image has focus (not while typing in the step / save path)
        self._canvas.mpl_connect("key_press_event", lambda e: e.key == "enter" and self._on_accept())
        self._accept_button.setEnabled(False)

    def _default_save_path(self) -> Path:
        scan = Path(getattr(self._image, "scan_path", "") or ".")
        folder = scan.parent if scan.suffix else scan
        return folder / f"{self._image.scan_name}_FirstPassROI.pkl"

    # ------------------------------------------------------------------ first pass
    def set_windows(self, ax_mm: float, lat_mm: float) -> None:
        """Called when the page is shown with the correlation-length window; reruns the first pass
        if the window changed."""
        if self._windows == (ax_mm, lat_mm) and self._intersect is not None:
            return
        self._windows = (ax_mm, lat_mm)
        self._step.setValue(round(0.2 * ax_mm, 2))
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
        self._intersect = intersect
        s, r = intersect.sample, intersect.reference
        self._fp_box.setText(
            f"<b>First pass</b>: {s.window_length_mm:.1f} mm axial window ({s.window_samples} samples), "
            f"{s.block_width_mm:.1f} mm blocks ({s.block_lines} lines), fs {s.fs_mhz:g} MHz<br>"
            f"Usable blocks: sample {s.usable_2d.mean():.0%}, reference {r.usable_2d.mean():.0%}, "
            f"<b>both {intersect.usable_both_2d.mean():.0%}</b>")
        self._roi_box.setText("Drag a box on the B-mode to draw the ROI.")
        try:
            self._draw()
        except Exception as e:  # never let a plotting problem take down the app (exceptions in slots abort)
            self.show_error(f"Could not draw the first pass: {e}")

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
        style_legend(ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8))

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
        style_axes(axr, "Draw an ROI", "Frequency (MHz)", "Depth (mm)")
        axr.grid(False)
        axr.tick_params(labelleft=True)
        style_legend(axr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8))
        ax.set_ylim(float(np.nanmax(self._axial)), float(min(np.nanmin(self._axial), 0)))

        self._selector = RectangleSelector(
            ax, self._on_box, useblit=True, button=[1], minspanx=0.5, minspany=0.5, spancoords="data",
            interactive=True, props=dict(edgecolor="white", facecolor="none", linewidth=1, linestyle="--"))
        fig.suptitle(f"First Pass\n{s.window_length_mm:.1f} mm window, {s.block_lines}-line blocks "
                     f"({s.block_width_mm:.1f} mm)\nUsable in sample & reference: {inter.usable_both_2d.mean():.0%} "
                     f"of blocks", color=TEXT, fontsize=11, y=0.995)
        fig.text(0.5, 0.905, "Quick power check only: spectra are recomputed with the final window settings "
                 "for parameter estimation.", color=UNUSABLE, fontsize=10.5, fontweight="bold", ha="center",
                 va="top")
        if self._roi_extents is not None:            # redraw (e.g. spectrum switched): keep the ROI
            self._selector.extents = self._roi_extents
            self._on_box(None, None)
        self._canvas.draw_idle()

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
            axr.set_title(f"No First-Pass Blocks Inside the ROI\n{roi.band_error}", color=UNUSABLE, fontsize=10)
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
        axr.set_title(f"{line1}\n{line3}", color=TEXT if ok else UNUSABLE, fontsize=10)
        (r1, r2), (l1, l2) = roi.rows, roi.lines
        colour = "white" if ok else UNUSABLE
        self._roi_box.setText(
            f"<b>ROI</b>: lines {l1}-{l2}, samples {r1}-{r2} ({roi.depth_range_mm[0]:.1f}-"
            f"{roi.depth_range_mm[1]:.1f} mm)<br><b>{line1}</b><br>"
            f"f<sub>low</sub> set at {b.f_low_limit_depth:.1f} mm, f<sub>high</sub> at {b.f_high_limit_depth:.1f} mm<br>"
            f"Min signal power above band level: sample {np.nanmin(b.signal_power_s1):.1f} dB, reference "
            f"{np.nanmin(b.signal_power_l1):.1f} dB (need {self._intersect.sample.min_signal_power_db:g})<br>"
            f"<span style='color:{colour}'>{line3}</span>")
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
        target = Path(self._save_path.text().strip())
        if target.exists():
            answer = QMessageBox.question(
                self, "Save ROI", f"{target.name} already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            s = self._intersect.sample
            path = save_roi_pkl(target, self._image, roi, frame=0, extra={
                "First Pass Window mm": (s.window_length_mm, s.block_width_mm),
                "First Pass Block Lines": s.block_lines,
                "Correlation Settings": dict(self.correlation_settings or {}),
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

    def show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.show()


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
