"""
Segmentation Type Selection Widget for Segmentation Loading
"""

from typing import Optional
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from PyQt6.QtCore import pyqtSignal

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.seg_loading.ui.seg_type_selection_ui import Ui_constructRoi
from src.qus.widgets.option_cards import SegTypeCardGrid, SpectrumCheckToggleRow
from src.qus.spectrum_check.spectrum_check_controller import spectrum_check_supported
from src.qus.window_roi import POWER_GUIDED_ROI
from engines.qus.quantus.data_objs import UltrasoundRfImage

# Cards shown on this screen. "seg_type" is the display name the model's set_seg_type() expects
# (backend loaders pkl_roi / nifti_voi are shown as "Pkl roi" / "Nifti voi").
SEG_TYPE_CARDS = [
    {"key": "power", "seg_type": POWER_GUIDED_ROI, "title": "Guided ROI",
     "description": "Draw with guides: usable power signal, speckle-based window sizes, and possible "
                    "vessels and bright reflectors",
     "badge": "guided"},
    {"key": "manual", "seg_type": "Manual Segmentation", "title": "Manual Segmentation",
     "description": "Draw an ROI or VOI directly on the B-mode image", "badge": "draw"},
    {"key": "load_roi", "seg_type": "Pkl roi", "title": "Load ROI",
     "description": "Import a saved 2D region of interest from a file", "badge": "2D · file"},
    {"key": "load_voi", "seg_type": "Nifti voi", "title": "Load VOI",
     "description": "Import a saved 3D volume of interest from a file", "badge": "3D · file"},
]
# Guided ROI has its own band check, so the spectrum check toggle is hidden for it
NO_SPECTRUM_TOGGLE = {"power"}

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


class SegTypeSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting segmentation type.
    
    This is the first step in the segmentation loading process where users
    choose the type of segmentation they want to load or create.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    seg_type_selected = pyqtSignal(str, bool)  # seg_type_name, include_spectrum_check
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_constructRoi()
        self._image_data = image_data
        self._cards: Optional[SegTypeCardGrid] = None
        self._spectrum_toggle: Optional[SpectrumCheckToggleRow] = None
        self._spectrum_check_available = False
        
    def setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for segmentation type selection only - use the main layout
        self.setLayout(self._ui.main_layout)
        
        # Configure stretch factors for type selection
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.select_type_layout, 10)
        
        # Ensure the layout fills the entire widget
        self._ui.main_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.main_layout.setSpacing(0)
        self._ui.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.full_screen_layout.setSpacing(0)

        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)
        
        self._build_cards()

    def _build_cards(self) -> None:
        """Swap the generated dropdown for the option-card grid + spectrum check toggle row."""
        self._cards = SegTypeCardGrid(SEG_TYPE_CARDS)
        self._spectrum_toggle = SpectrumCheckToggleRow()
        container = QWidget()
        container.setObjectName("cardContainer")
        container.setStyleSheet("QWidget#cardContainer { background: transparent; }")  # not inherited by children
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(16)
        column.addWidget(self._cards)
        column.addWidget(self._spectrum_toggle)

        layout = self._ui.select_type_layout
        layout.replaceWidget(self._ui.seg_type_dropdown, container)
        self._ui.seg_type_dropdown.hide()
        layout.setSpacing(28)
        layout.setStretch(0, 1)  # title
        layout.setStretch(1, 6)  # cards
        layout.setStretch(2, 1)  # accept
        layout.setStretch(3, 1)  # spacer

        self._ui.accept_type_button.setStyleSheet(ACCEPT_STYLE)
        self._ui.accept_type_button.setEnabled(False)
        self._spectrum_check_available = spectrum_check_supported(self._image_data)
        self._update_card_availability([])
        
    def connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.accept_type_button.clicked.connect(self._on_type_accepted)
        self._cards.selection_changed.connect(self._on_card_selected)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        
    def update_display(self, data) -> None:
        """Update the view with new data."""
        # This widget doesn't need to update with external data
        pass
        
    def set_seg_loaders(self, loader_names: list) -> None:
        """
        Enable the cards whose segmentation type the model offers for this image.
        
        Args:
            loader_names: List of formatted segmentation loader names
        """
        self._update_card_availability(loader_names)

    def _update_card_availability(self, loader_names: list) -> None:
        dims = getattr(self._image_data, "spatial_dims", 2)
        for card in SEG_TYPE_CARDS:
            key = card["key"]
            if card["seg_type"] not in loader_names:
                ok, reason = False, "Not available for this scan"
            elif key == "load_roi" and dims != 2:
                ok, reason = False, "ROI files are 2D; this scan is 3D"
            elif key == "load_voi" and dims != 3:
                ok, reason = False, "VOI files are 3D; this scan is 2D"
            else:
                ok, reason = True, ""
            self._cards.set_available(key, ok, reason)
        if self._cards.selected() is None:
            self._on_card_selected("")

    def get_selected_seg_type(self) -> str:
        """Get the currently selected segmentation type ('' if none)."""
        key = self._cards.selected() if self._cards else None
        return next((c["seg_type"] for c in SEG_TYPE_CARDS if c["key"] == key), "")

    def include_spectrum_check(self) -> bool:
        """Whether the user asked for the spectrum check after segmentation."""
        return self._spectrum_toggle is not None and self._spectrum_toggle.is_checked()

    def _on_card_selected(self, key: str) -> None:
        self._ui.accept_type_button.setEnabled(bool(key))
        show_toggle = bool(key) and key not in NO_SPECTRUM_TOGGLE and self._spectrum_check_available
        self._spectrum_toggle.set_shown(show_toggle)

    def _on_type_accepted(self) -> None:
        """Handle segmentation type selection acceptance."""
        selected_type = self.get_selected_seg_type()
        if selected_type:
            self.seg_type_selected.emit(selected_type, self.include_spectrum_check())
        else:
            print("Seg Type Selection Error: Please select a segmentation type")

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
