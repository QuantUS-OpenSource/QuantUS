"""
Analysis Parameters Widget for Analysis Loading

This widget allows users to configure parameters required for the selected analysis functions.
It dynamically creates input fields based on the required parameters.
"""

from typing import List, Optional, Dict, Any
from PyQt6.QtWidgets import (QWidget, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, 
                            QCheckBox, QComboBox, QFormLayout,
                            QGroupBox, QTextEdit)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer

from ...mvc.base_view import BaseViewMixin
from ..ui.analysis_params_ui import Ui_analysisParams
from engines.ceus.src.data_objs.image import UltrasoundImage
from engines.ceus.src.data_objs.seg import CeusSeg


class AnalysisParamsWidget(QWidget, BaseViewMixin):
    """
    Widget for configuring analysis parameters.
    
    This widget dynamically creates input fields based on the required parameters
    for the selected analysis functions.
    """
    
    # Signals for communicating with controller
    params_configured = pyqtSignal(dict)  # analysis_params
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundImage, seg_data: CeusSeg, config_data, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_analysisParams()
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        
        # Track parameter inputs
        self._param_inputs: Dict[str, QWidget] = {}
        self._required_params: List[str] = []
        self._selected_functions: List[str] = []

    def setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for parameters configuration (assuming similar structure to QUS)
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.analysis_params_layout, 10)
        
        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(getattr(self._image_data, 'scan_name', "No image loaded"))
        self._ui.phantom_path_input.setText(getattr(self._image_data, 'phantom_name', "No phantom loaded"))

    def connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.run_analysis_button.clicked.connect(self._on_run_analysis_clicked)
        self._ui.back_button.clicked.connect(self._on_back_clicked)

    def set_required_params(self, required_params: List[str], selected_functions: List[str]) -> None:
        """
        Set required parameters and create input fields.
        
        Args:
            required_params: List of required parameter names
            selected_functions: List of selected function names
        """
        print(f"DEBUG: AnalysisParamsWidget.set_required_params called")
        print(f"DEBUG: required_params = {required_params}")
        print(f"DEBUG: selected_functions = {selected_functions}")
        self._required_params = required_params
        self._selected_functions = selected_functions
        self._create_parameter_inputs()
        
    def _create_parameter_inputs(self) -> None:
        """Create input fields for each required parameter."""
        print(f"DEBUG: AnalysisParamsWidget._create_parameter_inputs called")
        
        # Clear existing inputs
        self._clear_params_layout()
        self._param_inputs = {}
        
        # If no params required, provide a small delay and auto-transition?
        # Or just show the screen with a "Continue" button
        if not self._required_params:
            print(f"DEBUG: No required params found")
            if hasattr(self._ui, 'run_analysis_button'):
                 self._ui.run_analysis_button.setText("Continue to Execution")
                 self._ui.run_analysis_button.setVisible(True)
                 self._ui.run_analysis_button.setEnabled(True)
            self._ui.analysis_running_label.hide()
            self._ui.analysis_execution_label.hide()
            return

        # Show normal parameter labels
        self._ui.analysis_params_label.show()
        self._ui.run_analysis_button.setText("Run Analysis")
        self._ui.run_analysis_button.setVisible(True)
        self._ui.run_analysis_button.setEnabled(True)
        self._ui.analysis_running_label.hide()
        self._ui.analysis_execution_label.hide()

        # Ideally would dynamically create inputs based on CEUS requirements
        form_layout = QFormLayout()
        for param_name in self._required_params:
            label = QLabel(param_name.replace("_", " ").title() + ":")
            label.setStyleSheet("color: white; font-size: 14px;")
            
            # Simple line edit for now
            line_edit = QLineEdit()
            line_edit.setStyleSheet("color: white; background-color: rgb(60, 60, 60); border: 1px solid gray; padding: 5px;")
            
            form_layout.addRow(label, line_edit)
            self._param_inputs[param_name] = line_edit
            
        self._ui.params_layout.addLayout(form_layout)
        
    def _clear_params_layout(self) -> None:
        """Clear all widgets from the params container."""
        if hasattr(self._ui, 'params_layout') and self._ui.params_layout is not None:
            while self._ui.params_layout.count():
                child = self._ui.params_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    # Recursively clear sub-layouts
                    def clear_sub_layout(l):
                        while l.count():
                            c = l.takeAt(0)
                            if c.widget(): c.widget().deleteLater()
                            elif c.layout(): clear_sub_layout(c.layout())
                    clear_sub_layout(child.layout())
        
    def _on_run_analysis_clicked(self) -> None:
        """Handle run analysis button click."""
        print(f"DEBUG: AnalysisParamsWidget._on_run_analysis_clicked called")
        # Collect parameters (simplified)
        params = {}
        # TODO: Collect actual values from dynamically created widgets
        
        print(f"DEBUG: Emitting params_configured with {params}")
        self.params_configured.emit(params)

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
