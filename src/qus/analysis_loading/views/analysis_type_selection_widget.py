"""
Analysis Type Selection Widget for Analysis Loading

This widget allows users to select which analysis type to run.
It provides a dropdown menu for analysis type options.
"""

from typing import Optional
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QMessageBox

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.analysis_loading.ui.analysis_type_selection_ui import Ui_analysisTypeSelection
from engines.qus.quantus.data_objs import UltrasoundRfImage

class AnalysisTypeSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting which analysis type to select functions from.
    
    This widget displays available analysis types in a dropdown menu.
    """
    
    # Signals for communicating with controller
    analysis_type_selected = pyqtSignal(str)  
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, analysis_types: dict, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_analysisTypeSelection()
        self._image_data = image_data
        self._analysis_types = analysis_types

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for type selection
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.analysis_type_layout, 10)

        # Update image and phantom paths
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.next_button.clicked.connect(self._on_next_clicked)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
            
    def _on_next_clicked(self) -> None:
        """Handle next button click."""
        selected_type = self._ui.analysis_type_options.currentText()
        if selected_type:
            self.analysis_type_selected.emit(selected_type)
        else:
            QMessageBox.critical(self, "Error", "Please select an analysis type.")

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()

    def set_type_options(self, analysis_types: dict) -> None:
        """
        Set available analysis types in the dropdown.
        
        Args:
            analysis_types: Dict 
        """
        self._ui.analysis_type_options.clear()
        self._ui.analysis_type_options.addItems(analysis_types)
