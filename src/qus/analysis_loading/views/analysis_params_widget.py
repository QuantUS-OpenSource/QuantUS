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

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.analysis_loading.ui.analysis_params_ui import Ui_analysisParams
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig
from engines.qus.quantus.reference_phantoms import get_reference_phantoms, ReferencePhantom

CUSTOM_PHANTOM_LABEL = "Custom (enter values manually)"
# Power-law attenuation is only meaningful when all three terms are on the form; a method that
# takes `ref_atten` alone (e.g. bsc_single_freq, linear dB/cm/MHz) is left for manual entry.
ATTEN_KWARGS = ("ref_atten", "ref_atten_exp", "ref_atten_offset")


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
    
    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, config_data: RfAnalysisConfig, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_analysisParams()
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        
        # Track parameter inputs
        self._param_inputs: Dict[str, QWidget] = {}
        self._required_params: Dict[str, Dict[str, str]] = {}     # function name, param name, default value
        self._selected_functions: List[str] = []
        self._execution_widgets: List[QWidget] = []
        self._param_selection_widgets: List[QWidget] = []
        self._phantom_combo: Optional[QComboBox] = None
        self._phantom_filled: List[str] = []      # params currently locked to the selected phantom

        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for parameters configuration
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.analysis_params_layout, 10)

        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

        # Hide execution widgets
        self._execution_widgets = [self._ui.analysis_execution_label, self._ui.analysis_running_label]
        self._param_selection_widgets = [self._ui.analysis_params_label, self._ui.run_analysis_button]
        for widget in self._execution_widgets:
            widget.hide()

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.run_analysis_button.clicked.connect(self._on_run_analysis_clicked)
        self._ui.back_button.clicked.connect(self._on_back_clicked)

    def set_required_params(self, required_params: Dict[str, Dict[str, str]], selected_functions: List[str]) -> None:
        """
        Set required parameters and create input fields.
        
        Args:
            required_params: Dictionary of required parameter names and their default values organized by function name
            selected_functions: List of selected function names
        """
        self._required_params = required_params
        self._selected_functions = selected_functions
        self._create_parameter_inputs()
        
    def _create_parameter_inputs(self) -> None:
        """Create input fields for each required parameter."""
        # Clear existing inputs
        self._clear_params_layout()
        self._param_inputs.clear()
        self._phantom_combo = None
        self._phantom_filled = []

        # Get the layout to add inputs to
        layout = self._ui.params_layout

        n_params = sum(len(params) for params in self._required_params.values())
        if not n_params:
            # No parameters required
            no_params_label = QLabel("No additional parameters required for the selected functions.")
            no_params_label.setStyleSheet("""
                QLabel {
                    color: rgb(200, 200, 200);
                    font-size: 14px;
                    background-color: transparent;
                    padding: 20px;
                }
            """)
            no_params_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(no_params_label)
            return

        for func_name, params in self._required_params.items():
            if len(params):
                # Create a group box for related parameters
                group_box = self._create_parameter_group(func_name, params)
                layout.addWidget(group_box)

        # Reference phantom selector goes on top, once the fields it fills exist
        phantom_group = self._create_phantom_group()
        if phantom_group is not None:
            layout.insertWidget(0, phantom_group)

        # Add stretch at the end
        layout.addStretch()

    def _phantom_fillable_values(self, phantom: ReferencePhantom) -> Dict[str, Any]:
        """Values the phantom supplies for params present on this form."""
        values = phantom.to_analysis_kwargs()
        if not all(k in self._param_inputs for k in ATTEN_KWARGS):
            for k in ATTEN_KWARGS:
                values.pop(k, None)
        return {k: v for k, v in values.items() if k in self._param_inputs}

    def _create_phantom_group(self) -> Optional[QGroupBox]:
        """Dropdown of library reference phantoms, shown when a selected method uses one."""
        try:
            self._phantoms = get_reference_phantoms()
        except Exception as e:  # a malformed phantom.yaml must not break the params page
            print(f"Error loading reference phantoms: {e}")
            return None
        if not any(self._phantom_fillable_values(p) for p in self._phantoms.values()):
            return None

        group_box = self._create_parameter_group("reference_phantom", {})
        combo = QComboBox()
        combo.addItem(CUSTOM_PHANTOM_LABEL, None)
        for name, phantom in self._phantoms.items():
            combo.addItem(phantom.display_name, name)
        combo.setStyleSheet(self._get_input_style())
        combo.currentIndexChanged.connect(self._on_phantom_changed)

        self._phantom_info = QLabel("")
        self._phantom_info.setWordWrap(True)
        self._phantom_info.setStyleSheet("QLabel { color: rgb(200, 200, 200); font-size: 11px; "
                                         "font-weight: normal; background-color: transparent; }")

        form = group_box.layout()
        label, _ = self._create_parameter_field("phantom", "")
        form.addRow(label, combo)
        form.addRow(self._phantom_info)
        self._phantom_combo = combo
        return group_box

    def _on_phantom_changed(self, _index: int) -> None:
        """Fill and lock the params the selected phantom provides; unlock them for Custom."""
        for name in self._phantom_filled:
            self._param_inputs[name].setEnabled(True)
        self._phantom_filled = []

        key = self._phantom_combo.currentData()
        if key is None:
            self._phantom_info.setText("")
            return
        phantom = self._phantoms[key]
        for name, value in self._phantom_fillable_values(phantom).items():
            if self._set_widget_value(self._param_inputs[name], value):
                self._param_inputs[name].setEnabled(False)
                self._phantom_filled.append(name)

        info = [phantom.description.strip()] if phantom.description else []
        if phantom.transmission_path is None and phantom.membrane_thickness_um is None \
                and "cirs_mode" in self._param_inputs:
            info.append("No membrane transmission data for this phantom: set Cirs Mode / Cirs Path manually.")
        self._phantom_info.setText("\n".join(info))

    @staticmethod
    def _set_widget_value(widget: QWidget, value: Any) -> bool:
        """Write a value into an input widget; False if this widget type can't hold it."""
        try:
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(f"{value:.6e}" if isinstance(value, float) and value != 0
                               and not 1e-3 <= abs(value) < 1e4 else str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                return False
        except (TypeError, ValueError):
            return False
        return True

    def _create_parameter_group(self, group_name: str, params: List[str]) -> QGroupBox:
        """
        Create a group box for related parameters.
        
        Args:
            group_name: Name of the parameter group
            params: List of parameter names in the group
            
        Returns:
            QGroupBox containing the parameter inputs
        """
        group_box = QGroupBox(group_name.replace('_', ' ').title())
        group_box.setStyleSheet("""
            QGroupBox {
                color: rgb(255, 255, 255);
                font-size: 14px;
                font-weight: bold;
                border: 2px solid rgb(120, 120, 120);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: transparent;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background-color: rgb(60, 60, 60);
                border-radius: 4px;
            }
        """)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        
        for param_name, default_val in params.items():
            label, input_widget = self._create_parameter_field(param_name, default_val)
            if label and input_widget:
                form_layout.addRow(label, input_widget)
                self._param_inputs[param_name] = input_widget

        group_box.setLayout(form_layout)
        return group_box

    def _create_parameter_field(self, param_name: str, default_val: Any) -> tuple[Optional[QLabel], Optional[QWidget]]:
        """
        Create a label and input widget for a parameter.
        
        Args:
            param_name: Name of the parameter
            default_val: Default value for the parameter

        Returns:
            Tuple of (label, input_widget)
        """
        label = QLabel(param_name.replace('_', ' ').title() + ":")
        label.setStyleSheet("""
            QLabel {
                color: rgb(255, 255, 255);
                font-size: 12px;
                background-color: transparent;
            }
        """)
        
        # Create appropriate input widget based on parameter type
        input_widget = self._create_input_widget(default_val)
        if not input_widget:
            return None, None
            
        return label, input_widget
        
    def _create_input_widget(self, default_val: Any) -> Optional[QWidget]:
        """
        Create appropriate input widget based on default parameter value.

        Args:
            default_val: Default value for the parameter

        Returns:
            Appropriate input widget
        """
        if type(default_val) == int:
            spin_box = QSpinBox()
            spin_box.setRange(-1000000, 1000000)
            spin_box.setValue(int(default_val))
            spin_box.setStyleSheet(self._get_input_style())
            return spin_box
        elif type(default_val) == float:
            # Use scientific notation for very small or very large values
            abs_val = abs(default_val)
            if abs_val != 0 and (abs_val < 1e-3 or abs_val >= 1e4):
                line_edit = QLineEdit()
                line_edit.setText(f"{default_val:.6e}")
                line_edit.setStyleSheet(self._get_input_style())
                return line_edit

            spin_box = QDoubleSpinBox()
            spin_box.setRange(-1000000.0, 1000000.0)
            spin_box.setDecimals(12)
            spin_box.setSingleStep(0.1)
            spin_box.setValue(float(default_val))
            spin_box.setStyleSheet(self._get_input_style())
            return spin_box
        elif type(default_val) == bool:
            checkbox = QCheckBox()
            checkbox.setChecked(bool(default_val))
            checkbox.setStyleSheet("""
                QCheckBox {
                    color: rgb(255, 255, 255);
                    background-color: transparent;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 2px solid rgb(120, 120, 120);
                    border-radius: 3px;
                    background-color: rgb(80, 80, 80);
                }
                QCheckBox::indicator:checked {
                    background-color: rgb(0, 120, 215);
                    border-color: rgb(0, 100, 195);
                }
            """)
            return checkbox
        else:
            # Default to line edit for string/number parameters
            line_edit = QLineEdit()
            line_edit.setText(str(default_val) if default_val else "")
            line_edit.setStyleSheet(self._get_input_style())
            return line_edit
            
    def _get_input_style(self) -> str:
        """Get common style for input widgets."""
        return """
            QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit, QComboBox {
                background-color: rgb(80, 80, 80);
                border: 2px solid rgb(120, 120, 120);
                border-radius: 5px;
                color: rgb(255, 255, 255);
                padding: 5px;
                font-size: 11px;
            }

            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QTextEdit:focus, QComboBox:focus {
                border-color: rgb(100, 150, 255);
            }

            QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {
                background-color: rgb(60, 60, 60);
                color: rgb(170, 170, 170);
            }
        """
        
    def _clear_params_layout(self) -> None:
        """Clear all widgets from the parameters layout."""
        layout = self._ui.params_layout
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def _get_configured_params(self) -> Dict[str, Any]:
        """Get the configured parameter values."""
        params = {}
        
        for param_name, input_widget in self._param_inputs.items():
            try:
                if isinstance(input_widget, QLineEdit):
                    value = input_widget.text().strip()
                    # Try to convert to number if possible
                    try:
                        if value == "":
                            pass
                        elif any(ch in value.lower() for ch in ('e',)):
                            value = float(value)
                        elif '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                        
                elif isinstance(input_widget, (QDoubleSpinBox, QSpinBox)):
                    value = input_widget.value()
                    
                elif isinstance(input_widget, QCheckBox):
                    value = input_widget.isChecked()
                    
                elif isinstance(input_widget, QTextEdit):
                    text = input_widget.toPlainText()
                    # Try to parse as JSON
                    try:
                        import json
                        value = json.loads(text)
                    except (json.JSONDecodeError, ValueError):
                        value = text
                        
                elif isinstance(input_widget, QComboBox):
                    value = input_widget.currentText()
                    
                else:
                    value = str(input_widget)  # Fallback
                    
                params[param_name] = value
                
            except Exception as e:
                # Handle any widget-specific errors
                print(f"Error getting value for {param_name}: {e}")
                params[param_name] = None
                
        return params

    def _on_run_analysis_clicked(self) -> None:
        """Handle run analysis button click."""
        configured_params = self._get_configured_params()
        self.params_configured.emit(configured_params)

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
        
    def show_loading(self) -> None:
        """Show loading state in the UI."""
        super().show_loading()
        # Disable widgets during loading
        for widget in self._param_selection_widgets:
            widget.hide()
        
        # Show loading message after a small delay
        self._loading_timer = getattr(self, '_loading_timer', None)
        if self._loading_timer:
            self._loading_timer.stop()
        
        self._loading_timer = QTimer()
        self._loading_timer.singleShot(200, self._show_loading_message)  # 200ms delay

    def _show_loading_message(self) -> None:
        """Show loading message if still in loading state."""
        for widget in self._execution_widgets:
            widget.show()

    def hide_loading(self) -> None:
        """Hide loading state in the UI."""
        super().hide_loading()
        
        # Cancel loading timer if it exists
        loading_timer = getattr(self, '_loading_timer', None)
        if loading_timer:
            loading_timer.stop()
        
        # Re-show widgets after loading
        for widget in self._param_selection_widgets:
            widget.show()
        for widget in self._execution_widgets:
            widget.hide()
