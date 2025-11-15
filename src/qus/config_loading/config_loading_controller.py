"""
Config Loading Controller for MVC architecture

This controller manages the config loading workflow and communicates with the application model.
It handles the loading of analysis configuration files and custom parameter setup.
"""

from typing import Optional, Any
from PyQt6.QtCore import QObject, pyqtSignal

from src.qus.mvc.base_controller import BaseController
from src.qus.config_loading.config_loading_view_coordinator import ConfigLoadingViewCoordinator
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig



class ConfigLoadingController(BaseController):
    """
    Controller for config loading workflow.
    
    Manages the interaction between the config loading view coordinator and the application model.
    Handles config file loading, validation, and parameter setup.
    """
    
    # ============================================================================
    # SIGNALS - Communication with application controller
    # ============================================================================
    
    user_action = pyqtSignal(str, object)  # action_name, action_data
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    
    def __init__(self, model, image_data: UltrasoundRfImage, seg_data: BmodeSeg, parent=None):
        # Initialize view coordinator first
        self._image_data = image_data
        self._seg_data = seg_data
        self._view_coordinator = ConfigLoadingViewCoordinator(image_data, seg_data)
        
        # Initialize base controller with the view coordinator
        super().__init__(model, self._view_coordinator)
        
        # Current state
        self._config_data: Optional[RfAnalysisConfig] = None
        self._selected_config_type: Optional[str] = None
        
        # Connect signals
        self._connect_signals()
        
        # Setup available config loaders
        self._setup_config_loaders()
        
    def _connect_signals(self) -> None:
        """Connect signals between view coordinator and controller."""
        # Forward view coordinator signals
        self._view_coordinator.user_action.connect(self._on_user_action)
        self._view_coordinator.back_requested.connect(self._on_back_requested)
        self._view_coordinator.close_requested.connect(self._on_close_requested)
        
    def _setup_config_loaders(self) -> None:
        """Setup available config loaders in the view."""
        loader_names = self._model.config_loader_names
        
        if self._view_coordinator:
            self._view_coordinator.set_config_loaders(loader_names)
            
    def _on_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Handle user actions from the view coordinator.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        if action_name == "config_type_selected":
            self._handle_config_type_selection(action_data)
        elif action_name == "config_file_selected":
            self._handle_config_file_selection(action_data)
        elif action_name == "custom_params_configured":
            self._handle_custom_params_configuration(action_data)
        elif action_name == "config_confirmed":
            self._handle_config_confirmation(action_data)
        else:
            # Forward unknown actions to application controller
            self.user_action.emit(action_name, action_data)
    
    def _on_config_loaded(self, config_data: RfAnalysisConfig) -> None:
        """
        Handle config loading completion.
        
        Args:
            config_data: Loaded configuration data
        """
        # Disconnect signals to avoid multiple connections
        self._model.config_loaded.disconnect(self._on_config_loaded)
        self._model.error_occurred.disconnect(self._on_config_error)
        
        self._config_data = config_data
        self._view_coordinator.show_config_preview(config_data)
    
    def _on_config_error(self, error_message: str) -> None:
        """
        Handle config loading error.
        
        Args:
            error_message: Error message
        """
        # Disconnect signals to avoid multiple connections
        self._model.config_loaded.disconnect(self._on_config_loaded)
        self._model.error_occurred.disconnect(self._on_config_error)
        
        self._view_coordinator.show_error(error_message)
            
    def _handle_config_type_selection(self, config_type: str) -> None:
        """
        Handle config type selection.
        
        Args:
            config_type: Selected config type
        """
        # Set config type in model
        if self._model.set_config_type(config_type):
            self._selected_config_type = config_type
            
            # Determine navigation based on config type
            if config_type == "Custom":
                # For custom option: go to custom analysis menu
                self._view_coordinator.show_custom_params()
            else:
                file_extensions = self._model.get_config_file_extensions()
                loading_options, default_vals = self._model.get_config_loading_options()

                if not len(file_extensions):
                    # For config types without file extensions, no inputs are required.
                    # Load the config immediately and show preview
                    self._load_config_for_preview()
                else:
                    # For config types with file extensions, show file selection as inputs as required.
                    self._view_coordinator.show_file_selection(file_extensions, loading_options, default_vals)
        else:
            self._view_coordinator.show_error("Failed to set config type")

    def _handle_config_file_selection(self, file_data: dict) -> None:
        """
        Handle config file selection.
        
        Args:
            file_data: Dictionary containing file path and metadata
        """
        config_path = file_data['file_path']
        config_kwargs = file_data['config_loader_kwargs']

        # Load config using the model
        self._model.load_config(config_path, config_kwargs=config_kwargs)
        
        # Connect to model signals for this operation
        self._model.config_loaded.connect(self._on_config_loaded)
        self._model.error_occurred.connect(self._on_config_error)
            
    def _handle_custom_params_configuration(self, params: dict) -> None:
        """
        Handle custom parameters configuration.
        
        Args:
            params: Dictionary containing custom parameters
        """
        # Load config using custom parameters via model
        self._model.load_config("", params)
        
        # Connect to model signals for this operation
        self._model.config_loaded.connect(self._on_config_loaded)
        self._model.error_occurred.connect(self._on_config_error)
            
    def _load_config_for_preview(self) -> None:
        """
        Load configuration for Philips and Clarius options and show preview directly.
        """
        # Load config using empty path (these configs don't require files)
        self._model.load_config("")
        
        # Connect to model signals for this operation
        self._model.config_loaded.connect(self._on_config_loaded)
        self._model.error_occurred.connect(self._on_config_error)
        
    def _handle_config_confirmation(self, config_data: RfAnalysisConfig) -> None:
        """
        Handle config confirmation.
        
        Args:
            config_data: Confirmed analysis configuration
        """
        # Store config data in model
        self._model.set_config_data(config_data)
        
        # Emit action to move to next step
        self.user_action.emit("config_loading_completed", config_data)
        
    def _on_back_requested(self) -> None:
        """Handle back navigation request."""
        self.back_requested.emit()
        
    def _on_close_requested(self) -> None:
        """Handle close request."""
        self.close_requested.emit()
        
    # ============================================================================
    # PUBLIC INTERFACE - Methods called by application controller
    # ============================================================================
    
    def get_widget(self) -> ConfigLoadingViewCoordinator:
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
            
    def reset(self) -> None:
        """Reset the controller to initial state."""
        if self._view_coordinator:
            self._view_coordinator.reset_to_config_type_selection()
            
    @property
    def config_data(self) -> Optional[RfAnalysisConfig]:
        """Get the loaded config data."""
        return self._config_data
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model.cleanup() 