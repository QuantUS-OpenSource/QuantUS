"""
Analysis Parameters Widget for Analysis Loading

This widget allows users to configure parameters required for the selected analysis functions.
It dynamically creates input fields based on the required parameters.
"""

from typing import List, Optional, Dict, Any
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QPushButton, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt

from engines.ceus.src.data_objs.image import UltrasoundImage
from engines.ceus.src.data_objs.seg import CeusSeg


class AnalysisParamsWidget(QWidget):
    """
    Widget for configuring analysis parameters.

    For analyses with no required parameters (e.g. TIC), this widget acts as
    a simple confirmation screen. For analyses that require parameters, input
    fields would be added here in future.
    """

    # Signals for communicating with controller
    params_configured = pyqtSignal(dict)  # analysis_params
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, image_data: UltrasoundImage, seg_data: CeusSeg, config_data,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data

        self._required_params: List[str] = []
        self._selected_functions: List[str] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the static UI."""
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(16)

        # Title
        title = QLabel("Analysis Parameters")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        root.addWidget(title)

        # Image info
        scan_name = getattr(self._image_data, 'scan_name', 'Unknown') if self._image_data else 'Unknown'
        info = QLabel(f"Image: {scan_name}")
        info.setStyleSheet("color: rgb(180, 180, 180); font-size: 12px;")
        root.addWidget(info)

        # MC info
        use_mc = getattr(self._seg_data, 'use_mc', False) if self._seg_data else False
        mc_label = QLabel("Motion compensation: " + ("enabled" if use_mc else "disabled"))
        mc_label.setStyleSheet(
            "color: rgb(100, 220, 100); font-size: 12px;" if use_mc
            else "color: rgb(180, 180, 180); font-size: 12px;"
        )
        root.addWidget(mc_label)

        # Dynamic parameter area (populated by set_required_params)
        self._params_area = QVBoxLayout()
        self._params_area.setSpacing(8)
        root.addLayout(self._params_area)

        root.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(self._button_style(accent=False))
        back_btn.clicked.connect(self._on_back_clicked)
        btn_row.addWidget(back_btn)

        btn_row.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setStyleSheet(self._button_style(accent=True))
        self._run_btn.clicked.connect(self._on_run_analysis_clicked)
        btn_row.addWidget(self._run_btn)

        root.addLayout(btn_row)

    @staticmethod
    def _button_style(accent: bool) -> str:
        if accent:
            return ("QPushButton { background-color: rgb(0, 120, 215); color: white; "
                    "border-radius: 6px; padding: 8px 20px; font-size: 13px; font-weight: bold; }"
                    "QPushButton:hover { background-color: rgb(0, 140, 240); }"
                    "QPushButton:disabled { background-color: rgb(80, 80, 80); color: rgb(130,130,130); }")
        return ("QPushButton { background-color: rgb(70, 70, 70); color: white; "
                "border-radius: 6px; padding: 8px 20px; font-size: 13px; }"
                "QPushButton:hover { background-color: rgb(90, 90, 90); }")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_required_params(self, required_params: List[str], selected_functions: List[str]) -> None:
        """
        Set required parameters and refresh the parameter input area.

        Args:
            required_params: List of required parameter names
            selected_functions: List of selected function names
        """
        self._required_params = required_params
        self._selected_functions = selected_functions
        self._refresh_params_area()

    def setup_ui(self) -> None:
        """No-op: UI is built in __init__. Kept for interface compatibility."""
        pass

    def connect_signals(self) -> None:
        """No-op: signals connected in __init__. Kept for interface compatibility."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_params_area(self) -> None:
        """Rebuild the dynamic parameter area."""
        # Clear existing widgets
        while self._params_area.count():
            item = self._params_area.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._required_params:
            note = QLabel("No additional parameters required.")
            note.setStyleSheet("color: rgb(180, 180, 180); font-size: 12px; font-style: italic;")
            self._params_area.addWidget(note)
        else:
            # Future: add input widgets per parameter
            for param in self._required_params:
                lbl = QLabel(f"Parameter required: {param}")
                lbl.setStyleSheet("color: rgb(220, 150, 50); font-size: 12px;")
                self._params_area.addWidget(lbl)

    def _on_run_analysis_clicked(self) -> None:
        """Collect parameters and emit params_configured signal."""
        params: Dict[str, Any] = {}
        self.params_configured.emit(params)

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()

    # ------------------------------------------------------------------
    # Compatibility stubs (called by view coordinator)
    # ------------------------------------------------------------------

    def show_error(self, error_message: str) -> None:
        print(f"AnalysisParamsWidget error: {error_message}")

    def clear_error(self) -> None:
        pass

    def show_loading(self) -> None:
        pass

    def hide_loading(self) -> None:
        pass
