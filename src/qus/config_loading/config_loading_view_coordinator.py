"""
Config Loading View Coordinator for MVC architecture

This coordinator manages the workflow between config type selection, file selection,
custom parameters configuration, and preview widgets, providing a unified interface for the controller.
It manages widgets that are designed to be embedded in the main application widget stack.
"""

from typing import Any, List, Optional
from PyQt6.QtWidgets import QWidget, QStackedWidget
from PyQt6.QtCore import pyqtSignal

from src.qus.mvc.base_view import BaseViewMixin
from .views.config_type_selection_widget import ConfigTypeSelectionWidget
from .views.config_file_selection_widget import ConfigFileSelectionWidget
from .views.custom_params_widget import CustomParamsWidget
from .views.config_preview_widget import ConfigPreviewWidget
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig
from src.qus.window_roi import POWER_CORRELATION_GUIDED


class ConfigLoadingViewCoordinator(QStackedWidget, BaseViewMixin):
    """
    Coordinator for config loading widgets.
    
    Manages the workflow between config type selection, file selection,
    custom parameters configuration, and preview widgets using a QStackedWidget. 
    This allows embedding in the main application widget stack for a seamless navigation experience.
    """
    
    # ============================================================================
    # SIGNALS - Communication with controller
    # ============================================================================
    
    user_action = pyqtSignal(str, object)  # action_name, action_data
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    
    
    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.__init_base_view__(parent)
        self._image_data = image_data
        self._seg_data = seg_data
        
        # Widget instances
        self._config_type_widget: Optional[ConfigTypeSelectionWidget] = None
        self._config_file_widget: Optional[ConfigFileSelectionWidget] = None
        self._custom_params_widget: Optional[CustomParamsWidget] = None
        self._frequency_prefill: dict = {}  # from the spectrum check step, applied to the Custom form
        self._config_preview_widget: Optional[ConfigPreviewWidget] = None
        self._power_config_widget = None  # Power and Correlation Guided (src.qus.window_roi)
        self._input_file_extensions: List[str] = []
        
        # Current state
        self._selected_config_type: Optional[str] = None
        self._config_data: Optional[RfAnalysisConfig] = None

        # Start with config type selection
        self.show_config_type_selection()

    # ============================================================================
    # CONTROLLER INPUT ROUTING - Route inputs from controller to appropriate widget
    # ============================================================================
    
    def set_config_loaders(self, loader_names: list) -> None:
        """
        Set available config loaders in the dropdown.
        
        Args:
            loader_names: List of formatted config loader names
        """
        if self._config_type_widget:
            self._config_type_widget.set_config_loaders(loader_names)

    def set_power_guided_roi(self, available: bool) -> None:
        """Whether Power and Correlation Guided can run on this segmentation (a 2D ROI on the RF data)."""
        if self._config_type_widget:
            self._config_type_widget.set_power_guided_roi(available)

    # ============================================================================
    # GENERAL WIDGET OPERATIONS - Loading states, errors, etc.
    # ============================================================================

    def show_loading(self) -> None:
        """Show loading state in the current widget."""
        current_widget: BaseViewMixin = self.currentWidget()
        current_widget.show_loading()

    def hide_loading(self) -> None:
        """Hide loading state in the current widget."""
        current_widget: BaseViewMixin = self.currentWidget()
        current_widget.hide_loading()
    
    def show_error(self, error_message: str) -> None:
        """
        Display error message to user in the current widget.
        
        Args:
            error_message: Error message to display
        """
        current_widget: BaseViewMixin = self.currentWidget()
        current_widget.show_error(error_message)

    def clear_error(self) -> None:
        """Clear error message in the current widget."""
        current_widget: BaseViewMixin = self.currentWidget()
        current_widget.clear_error()

    # ============================================================================
    # NAVIGATION METHODS - Methods to show different widgets
    # ============================================================================

    def reset_to_config_type_selection(self) -> None:
        """Reset to config type selection and clear all state."""
        self._selected_config_type = None
        self._config_data = None
        self.show_config_type_selection()

    def show_config_type_selection(self) -> None:
        """Show the config type selection widget."""
        self._input_file_extensions = []; self._loading_options = []; self._default_option_vals = []
        if self._config_type_widget is None:
            self._config_type_widget = ConfigTypeSelectionWidget(self._image_data, self._seg_data)
            self._config_type_widget.config_type_selected.connect(self._on_config_type_selected)
            self._config_type_widget.close_requested.connect(self.close_requested.emit)
            self._config_type_widget.back_requested.connect(self.back_requested.emit)
            self.addWidget(self._config_type_widget)
        
        self.setCurrentWidget(self._config_type_widget)
        self._config_type_widget.clear_error()

    def show_file_selection(self, file_extensions: list, loading_options: list, default_option_vals: list) -> None:
        """
        Show the config file selection widget.
        
        Args:
            file_extensions: List of allowed file extensions
            loading_options: List of loading options for the config type
            default_option_vals: List of default values for loading options
        """
        self._input_file_extensions = file_extensions
        self._loading_options = loading_options
        self._default_option_vals = default_option_vals
        if self._config_file_widget is not None:
            self.removeWidget(self._config_file_widget)
            self._config_file_widget.deleteLater()
            self._config_file_widget = None

        self._config_file_widget = ConfigFileSelectionWidget(self._selected_config_type, self._image_data,
                self._seg_data, self._input_file_extensions, self._loading_options, self._default_option_vals)

        self._config_file_widget.file_selected.connect(self._on_file_selected)
        self._config_file_widget.back_requested.connect(self._on_file_selection_back)
        self._config_file_widget.close_requested.connect(self.close_requested.emit)
        self.addWidget(self._config_file_widget)
        
        self.setCurrentWidget(self._config_file_widget)
        self._config_file_widget.clear_error()

    def show_custom_params(self) -> None:
        """Show the custom parameters configuration widget."""
        if self._custom_params_widget is None:
            self._custom_params_widget = CustomParamsWidget(self._image_data, self._seg_data)
            
            self._custom_params_widget.params_configured.connect(self._on_custom_params_configured)
            self._custom_params_widget.back_requested.connect(self._on_custom_params_back)
            self.addWidget(self._custom_params_widget)
            if self._frequency_prefill:
                self._custom_params_widget.prefill_frequencies(**self._frequency_prefill)
        
        self.setCurrentWidget(self._custom_params_widget)
        self._custom_params_widget.clear_error()

    def show_power_correlation_guided(self, roi_info) -> None:
        """Show the Power and Correlation Guided page (multitaper spectra over the ROI); accepting it
        creates a Custom config. roi_info: load_roi_pkl_info of a Power Guided ROI, or None (any other
        2D ROI: the page uses the segmentation's bounding A-lines / depths)."""
        from src.qus.window_roi.views.spectral_estimation_page import SpectralEstimationPage
        if self._power_config_widget is None:
            self._power_config_widget = SpectralEstimationPage(self._image_data, roi_info, seg_data=self._seg_data)
            self._power_config_widget.params_accepted.connect(self._on_custom_params_configured)
            self._power_config_widget.back_requested.connect(self.show_config_type_selection)
            self.addWidget(self._power_config_widget)
        self.setCurrentWidget(self._power_config_widget)

    def shutdown(self) -> None:
        """Stop any running Power and Correlation Guided computation (it finishes in the background)."""
        if self._power_config_widget is not None:
            self._power_config_widget.shutdown()

    def set_frequency_prefill(self, analysis_band=None, center_frequency=None, sampling_frequency=None) -> None:
        """Frequencies (Hz) to pre-fill the Custom config form with."""
        self._frequency_prefill = {k: v for k, v in dict(analysis_band=analysis_band, center_frequency=center_frequency,
                                                          sampling_frequency=sampling_frequency).items() if v is not None}

    def show_config_preview(self, config_data: RfAnalysisConfig) -> None:
        """
        Show the config preview widget.
        
        Args:
            config_data: Analysis configuration data to preview
        """
        if self._config_preview_widget is None:
            self._config_preview_widget = ConfigPreviewWidget(self._image_data, self._seg_data)
            self._config_preview_widget.config_confirmed.connect(self._on_config_confirmed)
            self._config_preview_widget.back_requested.connect(self._on_preview_back)
            self.addWidget(self._config_preview_widget)
        
        self._config_preview_widget.set_config_data(config_data)
        self.setCurrentWidget(self._config_preview_widget)
        self._config_preview_widget.clear_error()

    # ============================================================================
    # EVENT HANDLERS - Handle events from child widgets
    # ============================================================================

    def _on_config_type_selected(self, config_type_name: str) -> None:
        """
        Handle config type selection.
        
        Args:
            config_type_name: Selected config type name
        """
        self._selected_config_type = config_type_name
        self._emit_user_action("config_type_selected", config_type_name)

    def _on_file_selected(self, file_data: dict) -> None:
        """
        Handle file selection.
        
        Args:
            file_data: Dictionary containing file path and metadata
        """
        self._emit_user_action("config_file_selected", file_data)

    def _on_custom_params_configured(self, params: dict) -> None:
        """
        Handle custom parameters configuration.
        
        Args:
            params: Dictionary containing custom parameters
        """
        self._emit_user_action("custom_params_configured", params)

    def _on_config_confirmed(self, config_data: RfAnalysisConfig) -> None:
        """
        Handle config confirmation.
        
        Args:
            config_data: Confirmed analysis configuration
        """
        self._config_data = config_data
        self._emit_user_action("config_confirmed", config_data)

    def _on_file_selection_back(self) -> None:
        """Handle back navigation from file selection."""
        self.show_config_type_selection()

    def _on_custom_params_back(self) -> None:
        """Handle back navigation from custom params."""
        # Go back to config type selection (analysis configuration loading menu)
        self.show_config_type_selection()

    def _on_preview_back(self) -> None:
        """Handle back navigation from preview."""
        # Go back to config type selection (analysis configuration loading menu)
        if self._selected_config_type == "Custom":
            self.show_custom_params()
        elif self._selected_config_type == POWER_CORRELATION_GUIDED and self._power_config_widget is not None:
            self.setCurrentWidget(self._power_config_widget)
        else:
            if len(self._input_file_extensions):
                self.show_file_selection(self._input_file_extensions, self._loading_options, self._default_option_vals)
            else:
                self.show_config_type_selection()

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _emit_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Emit user action signal.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        self.user_action.emit(action_name, action_data) 