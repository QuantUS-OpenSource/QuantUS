"""
Visualization Function Selection Widget for QuantUS GUI

This widget allows users to select visualization functions with recommendations
based on the analysis results.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import QWidget, QListWidgetItem, QFileDialog, QMessageBox
from PyQt6.QtCore import pyqtSignal, Qt, QSize

from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis
from engines.qus.quantus.data_objs.analysis_config import RfAnalysisConfig
from engines.qus.quantus.data_objs.image import UltrasoundRfImage
from engines.qus.quantus.data_objs.seg import BmodeSeg
from ...mvc.base_view import BaseViewMixin
from ..ui.visualization_function_selection_ui import Ui_visualizationFunctionSelection


class VisualizationFunctionSelectionWidget(QWidget, BaseViewMixin):
    """Widget for selecting visualization functions."""
    
    # Signals for communicating with controller
    visualization_info = pyqtSignal(dict)  # Emitted when visualization functions are selected
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, 
                 config_data: RfAnalysisConfig, analysis_data: ParamapAnalysis, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_visualizationFunctionSelection()
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        self._analysis_data = analysis_data
        self._available_functions: List[str] = []

        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the user interface."""
        self._ui.setupUi(self)

        # Configure layout for function selection
        self.setLayout(self._ui.full_screen_layout)

        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.visualization_function_layout, 10)

        # Update image and phantom paths
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

        # Set default destination folder
        folder = Path(__file__).parents[3] / "Visualization_Results"
        self._ui.dest_path_input.setText(str(folder))

    def _connect_signals(self):
        """Connect signals."""
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        self._ui.next_button.clicked.connect(self._on_next_clicked)
        self._ui.choose_dest_path_button.clicked.connect(self._on_choose_folder) 
        self._ui.clear_dest_path_button.clicked.connect(self._ui.dest_path_input.clear)

    def _get_selected_functions(self) -> List[str]:
        """Retrieve the list of selected visualization functions."""
        selected = []
        for i in range(self._ui.funcs_list.count()):
            item = self._ui.funcs_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                func_name = item.text()
                if func_name in ["Parametric Maps", "B-Mode Statistical Maps"]:
                    # These categories both map to 'paramaps' visualization in the engine
                    if "paramaps" not in selected:
                        selected.append("paramaps")
                else:
                    func_name = func_name.replace('-', '_').lower()
                    selected.append(func_name)
        return selected

    def _on_next_clicked(self) -> None:
        """Handle next button click."""
        if not (dest_folder := self._ui.dest_path_input.text()):
            QMessageBox.critical(self, "Error", "Please specify a destination folder")
            return
        
        if dest_folder == str(Path(__file__).parents[4] / "Visualization_Results"):
            # If using default folder, remove old contents
            try:
                import shutil
                shutil.rmtree(dest_folder)
            except FileNotFoundError:
                pass  # Folder does not exist, no need to remove
        
        selected_functions = self._get_selected_functions()
        self.visualization_info.emit({
            "functions": selected_functions,
            "dest_folder": dest_folder,
        })

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()

    def _on_choose_folder(self):
        """Select folder to save VOI to."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._ui.dest_path_input.setText(folder)

    def update_available_functions(self, available_func_names: List[str]) -> None:
        """Update the list of available functions."""
        self._available_functions = available_func_names
        self._ui.funcs_list.clear()

        # Check for presence of both B-mode and standard Parametric results
        has_bmode = any(name.startswith('bmode') for name in self._analysis_data.results_names)
        has_paramap = any(not name.startswith('bmode') for name in self._analysis_data.results_names)

        if has_paramap:
            item = QListWidgetItem("Parametric Maps")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Checked)
            item.setSizeHint(QSize(0, 30))
            self._ui.funcs_list.addItem(item)
            
        if has_bmode:
            item = QListWidgetItem("B-Mode Statistical Maps")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Checked)
            item.setSizeHint(QSize(0, 30))
            self._ui.funcs_list.addItem(item)

        for func_name in self._available_functions:
            if func_name == "paramaps":
                continue  # skip this function as it's added as the categories above
            else:
                formatted_name = func_name.replace('_', '-').title()
                item = QListWidgetItem(formatted_name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setSizeHint(QSize(0, 30))
                self._ui.funcs_list.addItem(item)
