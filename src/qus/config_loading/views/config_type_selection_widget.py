"""
Config Type Selection Widget for Config Loading

This widget allows users to select the type of analysis configuration to load.
It provides a dropdown with available config loaders and handles the selection.
"""

from typing import List, Optional
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.config_loading.ui.config_type_selection_ui import Ui_configTypeSelection
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg


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
        
        # Set focus policy
        self._ui.config_type_dropdown.setFocusPolicy(self._ui.config_type_dropdown.focusPolicy())

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.accept_config_button.clicked.connect(self._on_config_accepted)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        
    def set_config_loaders(self, loader_names: List[str]) -> None:
        """
        Set available config loaders in the dropdown.
        
        Args:
            loader_names: List of formatted config loader names
        """
        self._ui.config_type_dropdown.clear()
        self._ui.config_type_dropdown.addItems(loader_names)
        
    def get_selected_config_type(self) -> str:
        """Get the currently selected config type."""
        return self._ui.config_type_dropdown.currentText()
            
    def _on_config_accepted(self) -> None:
        """Handle config type selection acceptance."""
        selected_type = self._ui.config_type_dropdown.currentText()
        if selected_type:
            self.config_type_selected.emit(selected_type)
        else:
            print("Config Type Selection Error: Please select a configuration type")

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit() 