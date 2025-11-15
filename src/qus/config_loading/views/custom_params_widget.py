"""
Custom Parameters Widget for Config Loading

This widget allows users to manually configure analysis parameters
for custom analysis configurations.
"""

from typing import Dict, Any, Optional
from PyQt6.QtWidgets import QWidget, QMessageBox, QGroupBox, QSpinBox, QDoubleSpinBox, QLabel, QHBoxLayout, QFormLayout, QPushButton, QVBoxLayout, QScrollArea
from PyQt6.QtCore import pyqtSignal, Qt

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.config_loading.ui.custom_params_ui import Ui_customParams
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg


class CustomParamsWidget(QWidget, BaseViewMixin):
    """
    Widget for configuring custom analysis parameters.
    
    This is the custom parameters configuration step in the config loading process 
    where users manually configure analysis parameters. Designed to be used within 
    the main application widget stack.
    """
    
    # Signals for communicating with controller
    params_configured = pyqtSignal(dict)  # params_dict
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_customParams()
        self._image_data = image_data
        self._seg_data = seg_data
        self._param_widgets: Dict[str, QWidget] = {}

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for custom params only
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors for custom params
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.custom_params_layout, 10)

        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)
        
        # Create and add parameter groups to the UI
        self._create_parameter_widgets()
        
    def _create_parameter_widgets(self) -> None:
        """Create and add parameter widgets to the UI."""
        # Clear existing widgets from the scroll area
        while self._ui.scrollAreaLayout.count():
            child = self._ui.scrollAreaLayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add parameter groups to the existing scroll area layout
        freq_group = self._create_frequency_group()
        self._ui.scrollAreaLayout.addWidget(freq_group)
        
        window_group = self._create_windowing_group()
        self._ui.scrollAreaLayout.addWidget(window_group)

        if self._image_data.spatial_dims == 3:
            group_3d = self._create_3d_group()
            self._ui.scrollAreaLayout.addWidget(group_3d)

        # Add some spacing at the bottom
        self._ui.scrollAreaLayout.addStretch()

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.accept_params_button.clicked.connect(self._on_create_config_clicked)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        
    def _create_frequency_group(self) -> QGroupBox:
        """Create the frequency parameters group."""
        group = QGroupBox("Frequency Parameters")
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # Transducer frequency band (min, max)
        self._transducer_min = QSpinBox()
        self._transducer_min.setRange(0, 100000000)
        self._transducer_min.setValue(0)
        self._transducer_min.setSuffix(" Hz")
        self._transducer_min.setSingleStep(1000000)
        self._param_widgets['transducer_freq_band_min'] = self._transducer_min
        
        self._transducer_max = QSpinBox()
        self._transducer_max.setRange(0, 100000000)
        self._transducer_max.setValue(8000000)
        self._transducer_max.setSuffix(" Hz")
        self._transducer_max.setSingleStep(1000000)
        self._param_widgets['transducer_freq_band_max'] = self._transducer_max
        
        freq_band_layout = QHBoxLayout()
        freq_band_layout.addWidget(QLabel("Min:"))
        freq_band_layout.addWidget(self._transducer_min)
        freq_band_layout.addWidget(QLabel("Max:"))
        freq_band_layout.addWidget(self._transducer_max)
        freq_band_layout.addStretch()
        
        layout.addRow("Transducer Frequency Band:", freq_band_layout)
        
        # Analysis frequency band (lower, upper)
        self._analysis_lower = QSpinBox()
        self._analysis_lower.setRange(0, 100000000)
        self._analysis_lower.setValue(3000000)
        self._analysis_lower.setSuffix(" Hz")
        self._analysis_lower.setSingleStep(1000000)
        self._param_widgets['analysis_freq_band_lower'] = self._analysis_lower
        
        self._analysis_upper = QSpinBox()
        self._analysis_upper.setRange(0, 100000000)
        self._analysis_upper.setValue(5000000)
        self._analysis_upper.setSuffix(" Hz")
        self._analysis_upper.setSingleStep(1000000)
        self._param_widgets['analysis_freq_band_upper'] = self._analysis_upper
        
        analysis_band_layout = QHBoxLayout()
        analysis_band_layout.addWidget(QLabel("Lower:"))
        analysis_band_layout.addWidget(self._analysis_lower)
        analysis_band_layout.addWidget(QLabel("Upper:"))
        analysis_band_layout.addWidget(self._analysis_upper)
        analysis_band_layout.addStretch()
        
        layout.addRow("Analysis Frequency Band:", analysis_band_layout)
        
        # Center frequency
        self._center_freq = QSpinBox()
        self._center_freq.setRange(0, 100000000)
        self._center_freq.setValue(4000000)
        self._center_freq.setSuffix(" Hz")
        self._center_freq.setSingleStep(1000000)
        self._param_widgets['center_frequency'] = self._center_freq
        layout.addRow("Center Frequency:", self._center_freq)
        
        # Sampling frequency
        self._sampling_freq = QSpinBox()
        self._sampling_freq.setRange(0, 1000000000)
        self._sampling_freq.setValue(53330000)
        self._sampling_freq.setSuffix(" Hz")
        self._sampling_freq.setSingleStep(1000000)
        self._param_widgets['sampling_frequency'] = self._sampling_freq
        layout.addRow("Sampling Frequency:", self._sampling_freq)
        
        return group
        
    def _create_windowing_group(self) -> QGroupBox:
        """Create the windowing parameters group."""
        group = QGroupBox("Windowing Parameters")
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # Axial window size
        self._ax_win_size = QDoubleSpinBox()
        self._ax_win_size.setRange(0.1, 99.0)
        self._ax_win_size.setValue(3.85)
        self._ax_win_size.setSuffix(" mm")
        self._ax_win_size.setDecimals(2)
        self._param_widgets['ax_win_size'] = self._ax_win_size
        layout.addRow("Axial Window Size:", self._ax_win_size)
        
        # Lateral window size
        self._lat_win_size = QDoubleSpinBox()
        self._lat_win_size.setRange(0.1, 99.0)
        self._lat_win_size.setValue(3.85)
        self._lat_win_size.setSuffix(" mm")
        self._lat_win_size.setDecimals(2)
        self._param_widgets['lat_win_size'] = self._lat_win_size
        layout.addRow("Lateral Window Size:", self._lat_win_size)
        
        # Window threshold
        self._window_thresh = QDoubleSpinBox()
        self._window_thresh.setRange(0, 99)
        self._window_thresh.setValue(95)
        self._window_thresh.setSuffix(" %")
        self._window_thresh.setDecimals(2)
        self._param_widgets['window_thresh'] = self._window_thresh
        layout.addRow("Window Threshold:", self._window_thresh)
        
        # Axial overlap
        self._axial_overlap = QDoubleSpinBox()
        self._axial_overlap.setRange(0.0, 99.0)
        self._axial_overlap.setValue(50)
        self._axial_overlap.setSuffix(" %")
        self._axial_overlap.setDecimals(2)
        self._param_widgets['axial_overlap'] = self._axial_overlap
        layout.addRow("Axial Overlap:", self._axial_overlap)
        
        # Lateral overlap
        self._lateral_overlap = QDoubleSpinBox()
        self._lateral_overlap.setRange(0.0, 99.0)
        self._lateral_overlap.setValue(50)
        self._lateral_overlap.setSuffix(" %")
        self._lateral_overlap.setDecimals(2)
        self._param_widgets['lateral_overlap'] = self._lateral_overlap
        layout.addRow("Lateral Overlap:", self._lateral_overlap)
        
        return group
        
    def _create_3d_group(self) -> QGroupBox:
        """Create the 3D parameters group."""
        group = QGroupBox("3D Parameters")
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # Coronal window size
        self._cor_win_size = QDoubleSpinBox()
        self._cor_win_size.setRange(0.1, 100.0)
        self._cor_win_size.setValue(20.0)
        self._cor_win_size.setSuffix(" mm")
        self._cor_win_size.setDecimals(2)
        self._cor_win_size.setSpecialValueText("Not used")
        self._param_widgets['cor_win_size'] = self._cor_win_size
        layout.addRow("Coronal Window Size:", self._cor_win_size)
        
        # Coronal overlap
        self._coronal_overlap = QDoubleSpinBox()
        self._coronal_overlap.setRange(0.0, 99.0)
        self._coronal_overlap.setValue(50)
        self._coronal_overlap.setSuffix(" %")
        self._coronal_overlap.setDecimals(2)
        self._coronal_overlap.setSpecialValueText("Not used")
        self._param_widgets['coronal_overlap'] = self._coronal_overlap
        layout.addRow("Coronal Overlap:", self._coronal_overlap)
        
        return group
        

    # ===================================================================
    # PUBLIC INTERFACE
    # ============================================================================
    
    def clear_error(self) -> None:
        """Clear any error messages."""
        # Could add error display functionality here
        pass
        
    def show_error(self, error_message: str) -> None:
        """
        Show error message.
        
        Args:
            error_message: Error message to display
        """
        QMessageBox.critical(self, "Error", error_message)
        
    def show_loading(self) -> None:
        """Show loading state."""
        self._ui.accept_params_button.setEnabled(False)
        self._ui.accept_params_button.setText("Creating...")
        
    def hide_loading(self) -> None:
        """Hide loading state."""
        self._ui.accept_params_button.setEnabled(True)
        self._ui.accept_params_button.setText("Accept")
        
    # ============================================================================
    # EVENT HANDLERS
    # ============================================================================
    
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
        
    def _on_create_config_clicked(self) -> None:
        """Handle create configuration button click."""
        try:
            params = self._collect_parameters()
            self._validate_parameters(params)
            self.params_configured.emit(params)
        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            
    # ============================================================================
    # PRIVATE METHODS
    # ============================================================================
    
    def _collect_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the form."""
        params = {
            'transducer_freq_band': [
                self._transducer_min.value(),
                self._transducer_max.value()
            ],
            'analysis_freq_band': [
                self._analysis_lower.value(),
                self._analysis_upper.value()
            ],
            'center_frequency': self._center_freq.value(),
            'sampling_frequency': self._sampling_freq.value(),
            'ax_win_size': self._ax_win_size.value(),
            'lat_win_size': self._lat_win_size.value(),
            'window_thresh': self._window_thresh.value()/100.0,
            'axial_overlap': self._axial_overlap.value()/100.0,
            'lateral_overlap': self._lateral_overlap.value()/100.0,
        }
        
        # Add 3D parameters if they are set
        if self._image_data.spatial_dims == 3:
            params['cor_win_size'] = self._cor_win_size.value()
            params['coronal_overlap'] = self._coronal_overlap.value()/100.0
            
        return params
        
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate the collected parameters.
        
        Args:
            params: Dictionary of parameters to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate frequency bands
        if params['transducer_freq_band'][0] >= params['transducer_freq_band'][1]:
            raise ValueError("Transducer frequency band min must be less than max")
            
        if params['analysis_freq_band'][0] >= params['analysis_freq_band'][1]:
            raise ValueError("Analysis frequency band lower must be less than upper")
            
        if (params['analysis_freq_band'][0] < params['transducer_freq_band'][0] or 
            params['analysis_freq_band'][1] > params['transducer_freq_band'][1]):
            raise ValueError("Analysis frequency band must be within transducer frequency band")
            
        # Validate window sizes
        if params['ax_win_size'] <= 0 or params['lat_win_size'] <= 0:
            raise ValueError("Window sizes must be positive")
            
        # Validate overlaps
        if not (0 <= params['axial_overlap'] <= 1) or not (0 <= params['lateral_overlap'] <= 1):
            raise ValueError("Overlap values must be between 0 and 1")
            
        if not (0 <= params['window_thresh'] <= 1):
            raise ValueError("Window threshold must be between 0 and 1") 
