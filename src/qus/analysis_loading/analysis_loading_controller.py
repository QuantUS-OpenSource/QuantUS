"""
Analysis Loading Controller for MVC architecture

This controller manages the analysis loading workflow and communicates with the application model.
It handles the selection of analysis types, functions, parameter configuration, and analysis execution.
"""

from typing import Optional, Any, List
from PyQt6.QtCore import QObject, pyqtSignal

from src.qus.mvc.base_controller import BaseController
from src.qus.analysis_loading.analysis_loading_view_coordinator import AnalysisLoadingViewCoordinator
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig
from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis


class AnalysisLoadingController(BaseController):
    """
    Controller for analysis loading workflow.
    
    Manages the interaction between the analysis loading view coordinator and the application model.
    Handles analysis type selection, function selection, parameter configuration, and execution.
    """
    
    # ============================================================================
    # SIGNALS - Communication with application controller
    # ============================================================================
    
    user_action = pyqtSignal(str, object)  # action_name, action_data
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    
    def __init__(self, model, image_data: UltrasoundRfImage, seg_data: BmodeSeg, config_data: RfAnalysisConfig, parent=None):
        # Initialize view coordinator first
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        self._view_coordinator = AnalysisLoadingViewCoordinator(image_data, seg_data, config_data)
        
        # Initialize base controller with the view coordinator
        super().__init__(model, self._view_coordinator)
        
        # Current state
        self._analysis_data: Optional[ParamapAnalysis] = None
        self._selected_analysis_type: Optional[str] = None
        self._selected_functions: List[str] = []
        self._analysis_params: dict = {}
        self._analysis_running: bool = False
        self._analysis_completion_emitted: bool = False
        
        # Connect signals
        self._connect_signals()
        
        # Setup available analysis types and functions
        self._setup_analysis_options()
        
    def _connect_signals(self) -> None:
        """Connect signals between view coordinator and controller."""
        # Forward view coordinator signals
        self._view_coordinator.user_action.connect(self._on_user_action)
        self._view_coordinator.back_requested.connect(self._on_back_requested)
        self._view_coordinator.close_requested.connect(self._on_close_requested)
        
    def _setup_analysis_options(self) -> None:
        """Setup available analysis types and functions in the view."""
        analysis_types, analysis_functions = self._model.get_analysis_types()
        
        # Automatically select "Paramap" as the analysis type
        paramap_type = "paramap"
        if paramap_type in analysis_types:
            self._selected_analysis_type = paramap_type
            if self._model.set_analysis_type(paramap_type):
                # Get available functions for Paramap analysis
                available_functions = self._model.get_analysis_functions(
                    paramap_type, self._image_data.spatial_dims)
                # Skip analysis type selection and go directly to function selection
                self._view_coordinator.show_function_selection(available_functions)
            else:
                self._view_coordinator.show_error("Failed to set Paramap analysis type")
        else:
            self._view_coordinator.show_error("Paramap analysis type not available")
            
    def _on_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Handle user actions from the view coordinator.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        
        if action_name == "analysis_functions_selected":
            self._handle_analysis_functions_selection(action_data)
        elif action_name == "analysis_execution_started":
            self._handle_analysis_execution(action_data)
        else:
            # Forward unknown actions to application controller
            self.user_action.emit(action_name, action_data)
    
    def _on_analysis_completed(self, analysis_data: ParamapAnalysis) -> None:
        """
        Handle analysis completion.
        
        Args:
            analysis_data: Completed analysis data
        """
        self._analysis_data = analysis_data
        self._analysis_running = False
        
        # Disconnect signals to avoid multiple connections
        try:
            self._model.analysis_completed.disconnect(self._on_analysis_completed)
        except (TypeError, RuntimeError):
            pass  # Signal was not connected
        try:
            self._model.error_occurred.disconnect(self._on_analysis_error)
        except (TypeError, RuntimeError):
            pass  # Signal was not connected

        # Prevent duplicate emissions
        if self._analysis_completion_emitted:
            print("DEBUG: Analysis completion already emitted, skipping duplicate")
            return
            
        # Store analysis data in model
        self._model.set_analysis_data(analysis_data)
        
        # Mark as emitted to prevent duplicates
        self._analysis_completion_emitted = True

        self.user_action.emit("analysis_loading_completed", analysis_data)
    
    def _on_analysis_error(self, error_message: str) -> None:
        """
        Handle analysis execution error.
        
        Args:
            error_message: Error message
        """
        # Disconnect signals to avoid multiple connections
        try:
            self._model.analysis_completed.disconnect(self._on_analysis_completed)
        except (TypeError, RuntimeError):
            pass  # Signal was not connected
        try:
            self._model.error_occurred.disconnect(self._on_analysis_error)
        except (TypeError, RuntimeError):
            pass  # Signal was not connected
        
        self._view_coordinator.show_error(error_message)
        self._analysis_running = False

    def _handle_analysis_functions_selection(self, selected_func_names: List[str]) -> None:
        """
        Handle analysis functions selection.
        
        Args:
            selected_func_names: List of selected function names
        """
        self._selected_functions = selected_func_names
        # Get required parameters for selected functions
        required_params = self._model.get_required_params(self._selected_analysis_type, selected_func_names)
        self._view_coordinator.show_params_configuration(required_params, selected_func_names)
            
    def _handle_analysis_execution(self, params: dict) -> None:
        """
        Handle analysis execution.

        Args:
            params: Dictionary containing analysis parameters
        """
        self._analysis_params = params
        
        # Handle analysis execution start
        if self._analysis_running:
            return
        
        # Disconnect any existing signal connections to avoid duplicates
        try:
            self._model.analysis_completed.disconnect(self._on_analysis_completed)
        except (TypeError, RuntimeError):
            pass  # Signal was not connected
        try:
            self._model.error_occurred.disconnect(self._on_analysis_error)
        except (TypeError, RuntimeError):
            pass  # Signal was not connected
        
        # Start analysis using the model
        self._analysis_running = True
        self._analysis_completion_emitted = False  # Reset completion flag
        self._model.run_analysis(
            self._selected_analysis_type, 
            self._image_data, 
            self._config_data, 
            self._seg_data, 
            self._selected_functions, 
            **self._analysis_params
        )
        
        # Connect to model signals for this operation
        self._model.analysis_completed.connect(self._on_analysis_completed)
        self._model.error_occurred.connect(self._on_analysis_error)
        
    def _on_back_requested(self) -> None:
        """Handle back navigation request."""
        self.back_requested.emit()
        
    def _on_close_requested(self) -> None:
        """Handle close request."""
        self.close_requested.emit()
        
    # ============================================================================
    # PUBLIC INTERFACE - Methods called by application controller
    # ============================================================================
    
    def get_widget(self) -> AnalysisLoadingViewCoordinator:
        """Get the view coordinator widget."""
        return self._view_coordinator
        
    def show_loading(self) -> None:
        """Show loading state."""
        if self._view_coordinator:
            self._view_coordinator.show_loading()
            
    def hide_loading(self) -> None:
        """Hide loading state."""
        if self._view_coordinator:
            self._view_coordinator.hide_loading()
            
    def show_error(self, error_message: str) -> None:
        """Show error message."""
        if self._view_coordinator:
            self._view_coordinator.show_error(error_message)
            
    def clear_error(self) -> None:
        """Clear error message."""
        if self._view_coordinator:
            self._view_coordinator.clear_error()
            
    @property
    def analysis_data(self) -> Optional[ParamapAnalysis]:
        """Get the analysis results data."""
        return self._analysis_data
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model.cleanup()
