"""
Analysis Function Selection Widget for Analysis Loading

This widget allows users to select which analysis functions to run.
It provides a dropdown menu for function selection and displays descriptions.
"""

from typing import List, Optional, Dict
from PyQt6.QtWidgets import QWidget, QComboBox, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import pyqtSignal, Qt

from quantus.gui.mvc.base_view import BaseViewMixin
from quantus.gui.analysis_loading.ui.analysis_function_selection_ui import Ui_analysisFunctionSelection
from quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig


class AnalysisFunctionSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting which analysis functions to run.
    
    This widget displays available analysis functions in a dropdown menu
    and shows descriptions for the selected function.
    """
    
    # Signals for communicating with controller
    functions_selected = pyqtSignal(dict)  # {'selected_functions': list, 'metadata': dict}
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, config_data: RfAnalysisConfig, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_analysisFunctionSelection()
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        
        # Track available functions and selected function
        self._available_functions: Dict = {}
        self._selected_function: Optional[str] = None
        
        # UI components for function selection
        self._function_combo_box: Optional[QComboBox] = None
        self._description_label: Optional[QLabel] = None
        
    def setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for function selection
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.analysis_function_layout, 10)

        # Update labels to reflect inputted image and phantom
        if self._image_data is not None:
            self._ui.image_path_input.setText(self._image_data.scan_name or "No image loaded")
            self._ui.phantom_path_input.setText(self._image_data.phantom_name or "No phantom loaded")
        else:
            self._ui.image_path_input.setText("No image loaded")
            self._ui.phantom_path_input.setText("No phantom loaded")
        
        # Configure the functions layout for dropdown and description
        self._ui.functions_layout.setSpacing(15)  # Increase spacing for better layout
        self._ui.functions_layout.setContentsMargins(20, 20, 20, 20)  # Add margins
        
        # Style the functions content widget
        self._ui.functions_content.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        
        # Ensure the functions content widget can expand to fill available space
        self._ui.functions_content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
    def connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.next_button.clicked.connect(self._on_continue_clicked)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        
    def update_display(self, data) -> None:
        """Update the view with new data."""
        # This widget doesn't need to update with external data
        pass
        
    def set_available_functions(self, available_functions: Dict) -> None:
        """
        Set available analysis functions and create dropdown menu.
        
        Args:
            available_functions: Dictionary of available functions for the selected analysis type
        """
        self._available_functions = available_functions
        self._create_function_selection_ui()
        
    def _create_function_selection_ui(self) -> None:
        """Create dropdown menu and description label for function selection."""
        # Clear existing widgets
        self._clear_function_layout()
        
        # Get the layout to add widgets to
        layout = self._ui.functions_layout
        
        # Count available functions excluding compute_power_spectra
        available_count = len([f for f in self._available_functions.keys() if f != 'compute_power_spectra'])
        print(f"DEBUG: Creating dropdown for {available_count} functions (excluding compute_power_spectra)")
        
        # Create instruction label
        instruction_label = QLabel("Select an analysis function:")
        instruction_label.setStyleSheet("""
            QLabel {
                color: rgb(255, 255, 255);
                font-size: 14px;
                font-weight: bold;
                background-color: transparent;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(instruction_label)
        
        # Create combo box for function selection
        self._function_combo_box = QComboBox()
        self._function_combo_box.setStyleSheet("""
            QComboBox {
                color: rgb(255, 255, 255);
                font-size: 12px;
                font-weight: bold;
                background-color: rgb(60, 60, 60);
                border: 2px solid rgb(120, 120, 120);
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 300px;
                max-width: 500px;
            }
            
            QComboBox:hover {
                border-color: rgb(150, 150, 150);
                background-color: rgb(70, 70, 70);
            }
            
            QComboBox:focus {
                border-color: rgb(0, 120, 215);
            }
            
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid rgb(255, 255, 255);
                margin-right: 10px;
            }
            
            QComboBox QAbstractItemView {
                color: rgb(255, 255, 255);
                background-color: rgb(60, 60, 60);
                border: 2px solid rgb(120, 120, 120);
                border-radius: 6px;
                selection-background-color: rgb(0, 120, 215);
                outline: none;
            }
        """)
        
        # Add placeholder item
        self._function_combo_box.addItem("-- Select a function --", None)
        
        # Add function options (filter out compute_power_spectra as it's a dependency function)
        for func_name, func_info in self._available_functions.items():
            # Skip compute_power_spectra as it's a dependency function, not a user-selectable analysis
            if func_name == 'compute_power_spectra':
                continue
            print(f"DEBUG: Adding function to dropdown: {func_name}")
            display_name = self._format_function_name(func_name)
            self._function_combo_box.addItem(display_name, func_name)
        
        # Connect signal for selection change
        self._function_combo_box.currentIndexChanged.connect(self._on_function_selection_changed)
        
        # Add combo box to layout
        layout.addWidget(self._function_combo_box)
        
        # Create description label
        self._description_label = QLabel("")
        self._description_label.setStyleSheet("""
            QLabel {
                color: rgb(200, 200, 200);
                font-size: 12px;
                background-color: rgb(40, 40, 40);
                border: 1px solid rgb(80, 80, 80);
                border-radius: 6px;
                padding: 15px;
                margin-top: 10px;
                line-height: 1.5;
                min-height: 60px;
            }
        """)
        self._description_label.setWordWrap(True)
        self._description_label.setMaximumWidth(500)
        self._description_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        # Add description label to layout
        layout.addWidget(self._description_label)
        
        # Add stretch at the end
        layout.addStretch()
        
        # Update button state
        self._update_continue_button_state()
        
    def _clear_function_layout(self) -> None:
        """Clear all widgets from the function list layout."""
        layout = self._ui.functions_layout
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def _format_function_name(self, func_name: str) -> str:
        """
        Format function name for display.
        
        Args:
            func_name: Internal function name
            
        Returns:
            Formatted display name
        """
        # Replace underscores with spaces and capitalize
        formatted = func_name.replace('_', ' ').title()
        
        # Handle special cases
        name_mappings = {
            'Bsc': 'BSC',
            'Bsc Stft': 'BSC STFT',
            'Lizzi Feleppa': 'Lizzi-Feleppa',
            'Hscan': 'H-scan',
            'Nakagami Params': 'Nakagami Parameters'
        }
        
        return name_mappings.get(formatted, formatted)
        
    def get_selected_functions(self) -> List[str]:
        """Get list of selected function names (will be only one or empty)."""
        if self._function_combo_box and self._selected_function:
            return [self._selected_function]
        return []
            
    def _on_continue_clicked(self) -> None:
        """Handle continue button click."""
        selected_functions = self.get_selected_functions()
        if selected_functions:
            # Count available functions excluding compute_power_spectra
            available_count = len([f for f in self._available_functions.keys() if f != 'compute_power_spectra'])
            function_data = {
                'selected_functions': selected_functions,
                'metadata': {
                    'total_available': available_count,
                    'total_selected': len(selected_functions)
                }
            }
            self.functions_selected.emit(function_data)
        else:
            self.show_error("Please select one analysis function")

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
        
    def _on_function_selection_changed(self, index: int) -> None:
        """Handle function selection change in dropdown."""
        if index > 0:  # Skip the placeholder item
            self._selected_function = self._function_combo_box.itemData(index)
            description = self._get_function_description(self._selected_function)
            self._description_label.setText(description)
            print(f"DEBUG: Function selected: {self._selected_function}")
        else:
            self._selected_function = None
            self._description_label.setText("")
            print("DEBUG: No function selected")
        
        self._update_continue_button_state()
        
    def _update_continue_button_state(self) -> None:
        """Update the enabled state of the continue button."""
        has_selection = len(self.get_selected_functions()) > 0
        self._ui.next_button.setEnabled(has_selection)

    def _get_function_description(self, func_name: str) -> str:
        """
        Get description for a specific analysis function.
        
        Args:
            func_name: Internal function name
            
        Returns:
            Description of the function
        """
        descriptions = {
            'compute_power_spectra': 'Computes power spectra and normalized power spectrum (NPS) for frequency-domain analysis',
            'lizzi_feleppa': 'Calculates Midband Fit (MBF), Spectral Slope (SS), and Spectral Intercept (SI) for tissue characterization',
            'attenuation_coef': 'Estimates local attenuation coefficient using Spectral Difference Method (dB/cm/MHz)',
            'bsc': 'Calculates Backscatter Coefficient using reference phantom method (1/cm-sr)',
            'nakagami_params': 'Computes Nakagami distribution parameters (m and Ω) for statistical tissue characterization',
            'hscan': 'Performs H-scan analysis using Hermite-Gaussian wavelets for tissue composition mapping',
            'central_freq_shift': 'Measures central frequency shift between scan and reference for tissue property detection'
        }
        
        return descriptions.get(func_name, 'No description available for this function.')
