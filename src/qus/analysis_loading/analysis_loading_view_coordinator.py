"""
Analysis Loading View Coordinator for MVC architecture

This coordinator manages the workflow between analysis type selection, function selection,
parameter configuration, and execution widgets, providing a unified interface for the controller.
It manages widgets that are designed to be embedded in the main application widget stack.
"""

from typing import Any, Optional, List, Dict
from PyQt6.QtWidgets import QWidget, QStackedWidget
from PyQt6.QtCore import pyqtSignal

from src.qus.mvc.base_view import BaseViewMixin
from .views.analysis_function_selection_widget import AnalysisFunctionSelectionWidget
from .views.analysis_params_widget import AnalysisParamsWidget
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig
from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis


class AnalysisLoadingViewCoordinator(QStackedWidget, BaseViewMixin):
    """
    Coordinator for analysis loading widgets.
    
    Manages the workflow between analysis type selection, function selection,
    parameter configuration, and execution widgets using a QStackedWidget. 
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
    
    
    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, config_data: RfAnalysisConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.__init_base_view__(parent)
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        
        # Widget instances
        self._function_selection_widget: Optional[AnalysisFunctionSelectionWidget] = None
        self._params_widget: Optional[AnalysisParamsWidget] = None

        # Note: Analysis type selection is now skipped - Paramap is automatically selected
        # The controller will call show_function_selection directly

    # ============================================================================
    # CONTROLLER INPUT ROUTING - Route inputs from controller to appropriate widget
    # ============================================================================
    
    def set_analysis_options(self, analysis_types: Dict, analysis_functions: Dict) -> None:
        """
        Set available analysis types and functions.
        
        Args:
            analysis_types: Dictionary of available analysis types
            analysis_functions: Dictionary of available functions for each type
        """
        # Note: This method is kept for compatibility but is no longer needed
        # since we automatically select paramap and skip type selection
        pass

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

    def show_function_selection(self, available_functions: Dict) -> None:
        """
        Show the analysis function selection widget.
        
        Args:
            available_functions: Dictionary of available functions for the selected analysis type
        """
        if self._function_selection_widget is None:
            self._function_selection_widget = AnalysisFunctionSelectionWidget(self._image_data, self._seg_data, 
                                                                              self._config_data, available_functions)
            self._function_selection_widget.functions_selected.connect(self._on_functions_selected)
            self._function_selection_widget.back_requested.connect(self._on_function_selection_back)
            self._function_selection_widget.close_requested.connect(self.close_requested.emit)
            self.addWidget(self._function_selection_widget)

        self.setCurrentWidget(self._function_selection_widget)
        self._function_selection_widget.clear_error()

    def show_params_configuration(self, required_params: Dict[str, Dict[str, str]], selected_functions: List[str]) -> None:
        """
        Show the analysis parameters configuration widget.
        
        Args:
            required_params: Dictionary of required parameter names and their default values organized by function name
            selected_functions: List of selected function names
        """
        if self._params_widget is None:
            self._params_widget = AnalysisParamsWidget(self._image_data, self._seg_data, self._config_data)
            self._params_widget.params_configured.connect(self._on_params_configured)
            self._params_widget.back_requested.connect(self._on_params_back)
            self.addWidget(self._params_widget)

        self._params_widget.set_required_params(required_params, selected_functions)
        self.setCurrentWidget(self._params_widget)
        self._params_widget.clear_error()

    # ============================================================================
    # EVENT HANDLERS - Handle events from child widgets
    # ============================================================================

    def _on_functions_selected(self, selected_func_names: List[str]) -> None:
        """
        Handle analysis functions selection.
        
        Args:
            selected_func_names: List of selected function names
        """
        self._emit_user_action("analysis_functions_selected", selected_func_names)

    def _on_params_configured(self, params: dict) -> None:
        """
        Handle analysis parameters configuration.
        
        Args:
            params: Dictionary containing analysis parameters
        """
        self._emit_user_action("analysis_execution_started", params)

    def _on_function_selection_back(self) -> None:
        """Handle back navigation from function selection."""
        # Since we skip analysis type selection, go back to the main application flow
        self.back_requested.emit()

    def _on_params_back(self) -> None:
        """Handle back navigation from parameters configuration."""
        # Go back to function selection
        if self._function_selection_widget:
            self.setCurrentWidget(self._function_selection_widget)

    def _on_execution_back(self) -> None:
        """Handle back navigation from execution."""
        # Go back to parameters configuration
        if self._params_widget:
            self.setCurrentWidget(self._params_widget)

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
