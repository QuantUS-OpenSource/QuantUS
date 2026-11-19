"""
Config Type Selection Widget for Config Loading

Option cards for the analysis configuration (the generated .ui's dropdown is hidden and replaced):
  Power & Correlation Guided  multitaper spectra over the ROI (any 2D ROI on the RF data)
  Probe Preset                one card for every scanner preset loader; a dropdown on the card
                              starts on the preset matching the scan's probe (if known)
  Custom                      parameters entered by hand
  Load from File              a saved analysis config (.pkl)
Accept emits the same config type display name as the old dropdown did.
"""

import re
from typing import Dict, List, Optional

from PyQt6.QtCore import QEvent, QObject, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.config_loading.ui.config_type_selection_ui import Ui_configTypeSelection
from src.qus.widgets.option_cards import OptionCardGrid
from src.qus.window_roi import POWER_CORRELATION_GUIDED
from src.qus.window_roi.probe import detect_probe_preset
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg

CUSTOM = "Custom"
LOAD_FROM_FILE = "Pkl RF"   # display name of the pkl_rf loader
# config types that are not scanner presets; every other loader goes into the Probe Preset card
NOT_PRESETS = {CUSTOM, LOAD_FROM_FILE, POWER_CORRELATION_GUIDED}
# probe detected from the scan (window_roi.probe) -> preset display name
PROBE_TO_PRESET = {"C3": "Clarius C3", "L15": "Clarius L15"}

ACCEPT_STYLE = """
QPushButton {
    color: white;
    font-size: 16px;
    background: rgb(90, 37, 255);
    border-radius: 15px;
}
QPushButton:disabled {
    color: rgb(150, 150, 150);
    background: rgb(70, 70, 70);
}
"""


def preset_label(display_name: str) -> str:
    """'Clarius c3 Config' -> 'Clarius C3', 'Philips 3d Config' -> 'Philips 3D'."""
    name = re.sub(r"\s*config$", "", display_name, flags=re.IGNORECASE)
    return " ".join(w.upper() if any(ch.isdigit() for ch in w) else w for w in name.split())


class _SelectOnPress(QObject):
    """Clicking the dropdown on a card also selects that card."""

    def __init__(self, grid: OptionCardGrid, key: str):
        super().__init__(grid)
        self._grid, self._key = grid, key

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress and self._grid.selected() != self._key:
            self._grid.select(self._key)
        return False


class ConfigTypeSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting the type of analysis configuration to load.

    This is the first step in the config loading process where users
    choose the type of configuration they want to load or create.
    Designed to be used within the main application widget stack.
    """

    # Signals for communicating with controller
    config_type_selected = pyqtSignal(str)  # config_type_name
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_configTypeSelection()
        self._image_data = image_data
        self._seg_data = seg_data
        self._loader_names: List[str] = []
        self._presets: Dict[str, str] = {}          # card label -> config type display name
        self._power_guided_roi = False              # a 2D ROI on the RF data (any segmentation type)
        self._detected_probe = detect_probe_preset(image_data)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)

        # Configure layout for config type selection only
        self.setLayout(self._ui.full_screen_layout)

        # Configure stretch factors for type selection
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.config_type_layout, 10)

        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

        self._build_cards()

    def _build_cards(self) -> None:
        """Swap the generated dropdown for the option cards."""
        self._preset_combo = QComboBox()
        self._preset_combo.setPlaceholderText("Choose a preset...")
        self._cards = OptionCardGrid([
            {"key": "power", "title": "Power & Correlation Guided",
             "description": "Correlation lengths and multitaper spectra in your ROI set the window sizes and band",
             "badge": "guided"},
            {"key": "preset", "title": "Probe Preset",
             "description": "Standard analysis settings for a scanner / probe",
             "badge": "preset", "extra": self._preset_combo},
            {"key": "custom", "title": "Custom",
             "description": "Enter the analysis parameters yourself", "badge": "manual"},
            {"key": "file", "title": "Load from File",
             "description": "Import saved analysis settings from a file", "badge": "file · .pkl"},
        ])
        self._preset_combo.installEventFilter(_SelectOnPress(self._cards, "preset"))

        container = QWidget()
        container.setObjectName("cardContainer")
        container.setStyleSheet("QWidget#cardContainer { background: transparent; }")  # not inherited by children
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._cards)

        layout = self._ui.config_type_layout
        layout.replaceWidget(self._ui.config_type_dropdown, container)
        self._ui.config_type_dropdown.hide()
        layout.setSpacing(28)
        layout.setStretch(0, 1)  # title
        layout.setStretch(1, 6)  # cards
        layout.setStretch(2, 1)  # accept
        layout.setStretch(3, 1)  # spacer

        self._ui.accept_config_button.setStyleSheet(ACCEPT_STYLE)
        self._ui.accept_config_button.setEnabled(False)
        self._update_cards()

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.accept_config_button.clicked.connect(self._on_config_accepted)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        self._cards.selection_changed.connect(lambda _key: self._update_accept())
        self._preset_combo.currentIndexChanged.connect(lambda _i: self._on_preset_changed())

    def set_config_loaders(self, loader_names: List[str]) -> None:
        """
        Set the available config loaders (display names) and enable the matching cards.

        Args:
            loader_names: List of formatted config loader names
        """
        self._loader_names = list(loader_names)
        self._presets = {preset_label(n): n for n in loader_names if n not in NOT_PRESETS}
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItems(list(self._presets))
        default = self._default_preset()
        self._preset_combo.setCurrentIndex(self._preset_combo.findText(default) if default else -1)
        self._preset_combo.blockSignals(False)
        self._update_cards()

    def set_power_guided_roi(self, available: bool) -> None:
        """Whether Power & Correlation Guided can run on this segmentation (a 2D ROI on the RF data)."""
        self._power_guided_roi = bool(available)
        self._update_cards()

    def _default_preset(self) -> Optional[str]:
        """Preset matching the scan: the Clarius probe from rf.yml, Philips 3D for 3D volumes."""
        if getattr(self._image_data, "spatial_dims", 2) == 3:
            wanted = "Philips 3D"
        else:
            wanted = PROBE_TO_PRESET.get(self._detected_probe)
        return wanted if wanted in self._presets else None

    def _update_cards(self) -> None:
        names = set(self._loader_names)
        if POWER_CORRELATION_GUIDED not in names:
            power = (False, "Not available for this scan")
        elif not self._power_guided_roi:
            power = (False, "Needs a 2D ROI on the RF data")
        else:
            power = (True, "")
        self._cards.set_available("power", *power)
        self._cards.set_available("preset", bool(self._presets), "No presets available")
        self._cards.set_available("custom", CUSTOM in names, "Not available")
        self._cards.set_available("file", LOAD_FROM_FILE in names, "Not available")
        self._on_preset_changed()

    def _on_preset_changed(self) -> None:
        card = self._cards.card("preset")
        label = self._preset_combo.currentText()
        default = self._default_preset()
        probe = PROBE_TO_PRESET.get(self._detected_probe, self._detected_probe)
        if label and default and label != default:
            card.set_note(f"This scan looks like {default}")
        elif label and not default and probe:
            card.set_note(f"No preset for the detected probe ({probe})")
        else:
            card.set_note("")
        self._update_accept()

    def _update_accept(self) -> None:
        self._ui.accept_config_button.setEnabled(bool(self.get_selected_config_type()))

    def get_selected_config_type(self) -> str:
        """The selected config type display name ('' if none / no preset chosen)."""
        key = self._cards.selected()
        if key == "power":
            return POWER_CORRELATION_GUIDED
        if key == "preset":
            return self._presets.get(self._preset_combo.currentText(), "")
        if key == "custom":
            return CUSTOM
        if key == "file":
            return LOAD_FROM_FILE
        return ""

    def _on_config_accepted(self) -> None:
        """Handle config type selection acceptance."""
        selected_type = self.get_selected_config_type()
        if selected_type:
            self.config_type_selected.emit(selected_type)
        else:
            print("Config Type Selection Error: Please select a configuration type")

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
