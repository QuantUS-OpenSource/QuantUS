"""
Scan Type Selection Widget for Image Loading
"""

from typing import Optional
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from ...mvc.base_view import BaseViewMixin
from ..ui.scan_type_ui import Ui_select_scan_type


class ScanTypeSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting scan type.
    
    This is the first step in the image loading process where users
    choose the type of scan they want to load. Designed to be used
    within the main application widget stack.
    """
    
    # Signals for communicating with controller
    scan_type_selected = pyqtSignal(str)  # scan_type_name
    close_requested = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_select_scan_type()

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for scan type selection only
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors for scan type selection
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.select_type_layout, 8)
        
        # Set focus policy
        self._ui.scan_type_dropdown.setFocusPolicy(self._ui.scan_type_dropdown.focusPolicy())

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.accept_type_button.clicked.connect(self._on_type_accepted)
            
    def _on_type_accepted(self) -> None:
        """Handle scan type selection acceptance."""
        selected_type = self._ui.scan_type_dropdown.currentText()
        if selected_type:
            self.scan_type_selected.emit(selected_type)
        else:
            print("Please select a scan type")

    def set_scan_loaders(self, loader_names: list) -> None:
        """
        Set available scan loaders in the dropdown.
        
        Args:
            loader_names: List of formatted scan loader names
        """
        self._ui.scan_type_dropdown.clear()
        self._ui.scan_type_dropdown.addItems(loader_names)
