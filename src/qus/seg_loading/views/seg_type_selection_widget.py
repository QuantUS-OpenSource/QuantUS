"""
Segmentation Type Selection Widget for Segmentation Loading
"""

from typing import Optional
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.seg_loading.ui.seg_type_selection_ui import Ui_constructRoi
from engines.qus.quantus.data_objs import UltrasoundRfImage


class SegTypeSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting segmentation type.
    
    This is the first step in the segmentation loading process where users
    choose the type of segmentation they want to load or create.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    seg_type_selected = pyqtSignal(str)  # seg_type_name
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_constructRoi()
        self._image_data = image_data
        
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
        
        # Set focus policy
        self._ui.seg_type_dropdown.setFocusPolicy(self._ui.seg_type_dropdown.focusPolicy())
        
    def connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.accept_type_button.clicked.connect(self._on_type_accepted)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        
    def update_display(self, data) -> None:
        """Update the view with new data."""
        # This widget doesn't need to update with external data
        pass
        
    def set_seg_loaders(self, loader_names: list) -> None:
        """
        Set available segmentation loaders in the dropdown.
        
        Args:
            loader_names: List of formatted segmentation loader names
        """
        self._ui.seg_type_dropdown.clear()
        self._ui.seg_type_dropdown.addItems(loader_names)
        
    def get_selected_seg_type(self) -> str:
        """Get the currently selected segmentation type."""
        return self._ui.seg_type_dropdown.currentText()
            
    def _on_type_accepted(self) -> None:
        """Handle segmentation type selection acceptance."""
        selected_type = self._ui.seg_type_dropdown.currentText()
        if selected_type:
            self.seg_type_selected.emit(selected_type)
        else:
            print("Seg Type Selection Error: Please select a segmentation type")

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
