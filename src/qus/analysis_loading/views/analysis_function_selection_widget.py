"""
Analysis Function Selection Widget for Analysis Loading

This widget allows users to select which analysis functions to run.
It provides a dropdown menu for function selection and displays descriptions.
"""

from typing import List, Optional
from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtWidgets import QWidget, QMessageBox, QListWidgetItem

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.analysis_loading.ui.analysis_function_selection_ui import Ui_analysisFunctionSelection
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig


class AnalysisFunctionSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting which analysis functions to run.
    
    This widget displays available analysis functions in a dropdown menu
    and shows descriptions for the selected function.
    """
    
    # Signals for communicating with controller
    functions_selected = pyqtSignal(list)  # list of selected function names
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, config_data: RfAnalysisConfig, 
                 available_functions: dict, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_analysisFunctionSelection()
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        self._available_functions = available_functions
        
        # Organize functions by type
        self._paramap_functions = {}
        self._bmode_functions = {}
        self._organize_functions_by_type()

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for function selection
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.analysis_function_layout, 10)

        # Update image and phantom paths
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

        # Populate paramap functions list
        for func_name in self._paramap_functions.keys():
            if func_name == "compute_power_spectra":
                continue  # skip this function for now
            formatted_name = func_name.replace('_', '-').title()
            item = QListWidgetItem(formatted_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setSizeHint(QSize(0, 30))  # increase item height
            self._ui.paramap_list.addItem(item)

        # Populate bmode functions list
        for func_name in self._bmode_functions.keys():
            if func_name == "compute_power_spectra":
                continue  # skip this function for now
            formatted_name = func_name.replace('_', '-').title()
            item = QListWidgetItem(formatted_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setSizeHint(QSize(0, 30))  # increase item height
            self._ui.bmode_list.addItem(item)

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.next_button.clicked.connect(self._on_next_clicked)
        self._ui.back_button.clicked.connect(self._on_back_clicked)

    def _organize_functions_by_type(self) -> None:
        """Organize available functions by type (paramap vs bmode)."""
        for func_name, func_info in self._available_functions.items():
            # Get the module path of the function
            module_path = getattr(func_info, '__module__', '')
            
            # Check if function belongs to paramap or bmode based on module path
            if 'paramap' in module_path.lower():
                self._paramap_functions[func_name] = func_info
            elif 'bmode' in module_path.lower():
                self._bmode_functions[func_name] = func_info
            else:
                # Fallback: check function name for keywords
                if 'radiomics' in func_name.lower():
                    self._bmode_functions[func_name] = func_info
                else:
                    # Default to bmode if unclear
                    self._bmode_functions[func_name] = func_info

    def _get_selected_functions(self) -> List[str]:
        """Retrieve the list of selected analysis functions from both lists."""
        selected = []
        
        # Get selected from paramap list
        for i in range(self._ui.paramap_list.count()):
            item = self._ui.paramap_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                func_name = item.text().replace('-', '_').lower()
                selected.append(func_name)
        
        # Get selected from bmode list
        for i in range(self._ui.bmode_list.count()):
            item = self._ui.bmode_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                func_name = item.text().replace('-', '_').lower()
                selected.append(func_name)
        
        return selected
            
    def _on_next_clicked(self) -> None:
        """Handle next button click."""
        selected_functions = self._get_selected_functions()
        if selected_functions:
            self.functions_selected.emit(selected_functions)
        else:
            QMessageBox.critical(self, "Error", "Please select at least one analysis method to run.")

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
