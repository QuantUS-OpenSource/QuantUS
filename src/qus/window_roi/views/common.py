"""Shared look and helpers for the Window Sizes & ROI step (dark theme matching the QuantUS GUI)."""

from typing import Callable

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QLabel, QFrame

BG = (42 / 255, 42 / 255, 42 / 255)
PANEL = (52 / 255, 52 / 255, 52 / 255)
GRID = (0.38, 0.38, 0.38)
TEXT = "white"
MUTED = "#bdbdbd"
LATERAL = "#ffb347"     # lateral width / window
AXIAL = "#4fa3ff"       # axial length / window
USABLE = "#ff4fd8"      # usable in sample and reference / shared band (DrawROIWithBand magenta)
UNUSABLE = "#ff3b3b"    # not usable / unusable rows
ROI = "#ff5a5a"         # ROI outline (DrawROIWithBand light red)
REGION = "#7cd992"      # measured region on the reference

SIDEBAR_WIDTH = 340


def style_axes(ax, title: str = None, xlabel: str = None, ylabel: str = None) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color((0.55, 0.55, 0.55))
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    if title is not None:
        ax.set_title(title, color=TEXT, fontsize=10, pad=6)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)


def style_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor(BG)
    legend.get_frame().set_edgecolor((0.45, 0.45, 0.45))
    for text in legend.get_texts():
        text.set_color(TEXT)


def bmode_db(rf_2d: np.ndarray, dynamic_range_db: float = 60.0) -> np.ndarray:
    """Log-compressed Hilbert envelope (dB re max, clipped to the dynamic range) of one RF frame."""
    from engines.qus.quantus.correlation_length.correlation import envelope_frames

    env = envelope_frames(np.asarray(rf_2d, dtype=float)[None])[0]
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(env / env.max())
    return np.clip(np.nan_to_num(db, neginf=-dynamic_range_db), -dynamic_range_db, 0)


def first_frame(rf: np.ndarray) -> np.ndarray:
    rf = np.asarray(rf)
    return rf[0] if rf.ndim == 3 else rf


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-size: 22px; font-weight: bold;")
    return label


def hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: rgb(190, 190, 190);")
    return label


def results_box() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setStyleSheet("font-size: 13px; padding: 6px; border: 1px solid rgb(80, 80, 80);")
    return label


def error_label() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setStyleSheet("color: rgb(255, 110, 110);")
    label.hide()
    return label


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: rgb(80, 80, 80);")
    return line


def stop_worker(worker) -> None:
    """Disconnect a Worker so its result is ignored (it finishes in the background)."""
    if worker is None:
        return
    for sig in (worker.done, worker.failed):
        try:
            sig.disconnect()
        except TypeError:
            pass
    # no wait(): a running Worker keeps itself alive (_RUNNING) and finishes in the background,
    # so leaving a page never blocks the GUI; results go nowhere once disconnected


_RUNNING = set()  # running workers keep themselves alive, so deleting a page never destroys a running QThread
_QUIT_HOOKED = False


def _wait_all_on_quit() -> None:
    for worker in list(_RUNNING):
        worker.wait()


class Worker(QThread):
    """Runs fn() off the GUI thread; emits done(result) or failed(exception). Has no Qt parent:
    it holds a reference to itself from start() until it finishes, and running workers are waited
    for when the application quits."""

    done = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, fn: Callable, parent=None):
        super().__init__(None)
        self._fn = fn
        self.finished.connect(lambda: _RUNNING.discard(self))

    def start(self, *args) -> None:
        global _QUIT_HOOKED
        _RUNNING.add(self)  # before the thread runs, so there is no window where it can be collected
        if not _QUIT_HOOKED:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(_wait_all_on_quit)
                _QUIT_HOOKED = True
        super().start(*args)

    def run(self) -> None:
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.done.emit(self._fn())
        except Exception as e:  # reported in the page
            self.failed.emit(e)
