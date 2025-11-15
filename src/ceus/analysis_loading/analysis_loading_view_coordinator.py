"""
Analysis Loading View Coordinator for MVC architecture

This coordinator manages the workflow between analysis type selection, function selection,
parameter configuration, and execution widgets, providing a unified interface for the controller.
It manages widgets that are designed to be embedded in the main application widget stack.
"""

from typing import Any, Optional, List, Dict
from PyQt6.QtWidgets import QWidget, QStackedWidget
from PyQt6.QtCore import pyqtSignal

from quantus.gui.mvc.base_view import BaseViewMixin
from .views.analysis_function_selection_widget import AnalysisFunctionSelectionWidget
from quantus.gui.config_loading.views.analysis_params_widget import AnalysisParamsWidget
from .views.analysis_execution_widget import AnalysisExecutionWidget
from quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig
from quantus.analysis.paramap.framework import ParamapAnalysis


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
        
        print(f"DEBUG: AnalysisLoadingViewCoordinator - image_data = {image_data is not None}")
        if image_data is not None:
            print(f"DEBUG: AnalysisLoadingViewCoordinator - scan_name = {image_data.scan_name}")
            print(f"DEBUG: AnalysisLoadingViewCoordinator - phantom_name = {image_data.phantom_name}")
        else:
            print(f"DEBUG: AnalysisLoadingViewCoordinator - image_data is None!")
        
        # Widget instances
        self._function_selection_widget: Optional[AnalysisFunctionSelectionWidget] = None
        self._params_widget: Optional[AnalysisParamsWidget] = None
        self._execution_widget: Optional[AnalysisExecutionWidget] = None
        
        # Current state
        self._selected_analysis_type: Optional[str] = None
        self._selected_functions: List[str] = []
        self._analysis_params: dict = {}
        self._analysis_data: Optional[ParamapAnalysis] = None

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

    def reset_to_analysis_type_selection(self) -> None:
        """Reset to function selection and clear all state."""
        self._selected_analysis_type = None
        self._selected_functions = []
        self._analysis_params = {}
        self._analysis_data = None
        # Note: This method name is kept for compatibility, but it now resets to function selection
        # The controller will handle showing the appropriate widget

    def show_function_selection(self, available_functions: Dict) -> None:
        """
        Show the analysis function selection widget.
        
        Args:
            available_functions: Dictionary of available functions for the selected analysis type
        """
        if self._function_selection_widget is None:
            self._function_selection_widget = AnalysisFunctionSelectionWidget(self._image_data, self._seg_data, self._config_data)
            self._function_selection_widget.setup_ui()
            self._function_selection_widget.connect_signals()
            self._function_selection_widget.functions_selected.connect(self._on_functions_selected)
            self._function_selection_widget.back_requested.connect(self._on_function_selection_back)
            self._function_selection_widget.close_requested.connect(self.close_requested.emit)
            self.addWidget(self._function_selection_widget)
        
        self._function_selection_widget.set_available_functions(available_functions)
        self.setCurrentWidget(self._function_selection_widget)
        self._function_selection_widget.clear_error()

    def show_params_configuration(self, required_params: List[str], selected_functions: List[str]) -> None:
        """
        Show the analysis parameters configuration widget.
        
        Args:
            required_params: List of required parameter names
            selected_functions: List of selected function names
        """
        print(f"DEBUG: show_params_configuration called with required_params = {required_params}, selected_functions = {selected_functions}")
        
        if self._params_widget is None:
            print(f"DEBUG: Creating AnalysisParamsWidget with image_data = {self._image_data is not None}")
            if self._image_data is not None:
                print(f"DEBUG: Passing scan_name = {self._image_data.scan_name}")
                print(f"DEBUG: Passing phantom_name = {self._image_data.phantom_name}")
            self._params_widget = AnalysisParamsWidget(self._image_data, self._seg_data, self._config_data)
            self._params_widget.setup_ui()
            self._params_widget.connect_signals()
            self._params_widget.params_configured.connect(self._on_params_configured)
            self._params_widget.back_requested.connect(self._on_params_back)
            self.addWidget(self._params_widget)
        
        print(f"DEBUG: Calling set_required_params...")
        self._params_widget.set_required_params(required_params, selected_functions)
        print(f"DEBUG: Setting current widget to params widget...")
        self.setCurrentWidget(self._params_widget)
        print(f"DEBUG: Clearing error...")
        self._params_widget.clear_error()
        print(f"DEBUG: show_params_configuration completed")

    def show_analysis_execution(self, execution_summary: Dict) -> None:
        """
        Show the analysis execution widget.
        
        Args:
            execution_summary: Dictionary containing execution summary data
        """
        print(f"DEBUG: show_analysis_execution called with execution_summary = {execution_summary}")
        print(f"DEBUG: Transitioning from analysis params to execution screen...")
        
        if self._execution_widget is None:
            print(f"DEBUG: Creating new AnalysisExecutionWidget...")
            self._execution_widget = AnalysisExecutionWidget(self._image_data, self._seg_data, self._config_data)
            self._execution_widget.setup_ui()
            self._execution_widget.connect_signals()
            self._execution_widget.execution_started.connect(self._on_execution_started)
            self._execution_widget.analysis_confirmed.connect(self._on_analysis_confirmed)
            self._execution_widget.back_requested.connect(self._on_execution_back)
            self.addWidget(self._execution_widget)
            print(f"DEBUG: AnalysisExecutionWidget created and added to stack")
        else:
            print(f"DEBUG: Using existing AnalysisExecutionWidget")
        
        print(f"DEBUG: Setting execution summary...")
        self._execution_widget.set_execution_summary(execution_summary)
        print(f"DEBUG: About to set current widget to execution widget...")
        self.setCurrentWidget(self._execution_widget)
        print(f"DEBUG: Current widget set to execution widget - screen should now show execution")
        print(f"DEBUG: Clearing error...")
        self._execution_widget.clear_error()
        print(f"DEBUG: show_analysis_execution completed - execution screen should be visible")

    def show_analysis_results(self, analysis_data: ParamapAnalysis) -> None:
        """
        Show analysis results in the execution widget.
        
        Args:
            analysis_data: Completed analysis data
        """
        self._analysis_data = analysis_data
        if self._execution_widget:
            self._execution_widget.show_results(analysis_data)

    # ============================================================================
    # EVENT HANDLERS - Handle events from child widgets
    # ============================================================================

    def _on_functions_selected(self, function_data: dict) -> None:
        """
        Handle analysis functions selection.
        
        Args:
            function_data: Dictionary containing selected functions and metadata
        """
        self._selected_functions = function_data['selected_functions']
        self._emit_user_action("analysis_functions_selected", function_data)

    def _on_params_configured(self, params: dict) -> None:
        """
        Handle analysis parameters configuration.
        
        Args:
            params: Dictionary containing analysis parameters
        """
        print(f"DEBUG: View coordinator _on_params_configured called")
        print(f"DEBUG: params = {params}")
        self._analysis_params = params
        print(f"DEBUG: About to emit user_action with analysis_params_configured")
        self._emit_user_action("analysis_params_configured", params)
        print(f"DEBUG: user_action signal emitted - controller should receive this")

    def _on_execution_started(self, execution_data: dict) -> None:
        """
        Handle analysis execution start.
        
        Args:
            execution_data: Dictionary containing execution parameters
        """
        print(f"DEBUG: View coordinator received execution_started signal")
        print(f"DEBUG: execution_data = {execution_data}")
        print(f"DEBUG: About to emit user_action signal")
        self._emit_user_action("analysis_execution_started", execution_data)
        print(f"DEBUG: user_action signal emitted")

    def _on_analysis_confirmed(self, analysis_data: ParamapAnalysis) -> None:
        """
        Handle analysis completion confirmation.
        
        Args:
            analysis_data: Completed analysis data
        """
        self._analysis_data = analysis_data
        # User has clicked "Continue to Visualization" after reviewing results
        # Now emit the action to proceed to visualization
        print(f"DEBUG: User confirmed analysis completion, proceeding to visualization")
        print(f"DEBUG: analysis_data type = {type(analysis_data)}")
        print(f"DEBUG: analysis_data = {analysis_data}")
        print(f"DEBUG: About to emit analysis_loading_completed action")
        self._emit_user_action("analysis_loading_completed", analysis_data)
        print(f"DEBUG: analysis_loading_completed action emitted successfully")

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
