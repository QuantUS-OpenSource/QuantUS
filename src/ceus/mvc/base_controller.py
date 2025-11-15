"""
Base Controller class for MVC architecture
"""

from typing import Any, Optional
from PyQt6.QtCore import QObject, pyqtSlot

from .base_model import BaseModel
from .base_view import BaseViewMixin


class BaseController(QObject):
    """
    Base class for all controllers in the MVC architecture.
    
    Controllers coordinate between models and views, handling user actions
    and updating the view based on model changes.
    """
    
    def __init__(self, model: BaseModel, view: BaseViewMixin):
        super().__init__()
        self._model = model
        self._view = view
        
        # Connect model signals to controller slots
        self._model.error_occurred.connect(self._on_model_error)
        self._model.loading_started.connect(self._on_loading_started)
        self._model.loading_finished.connect(self._on_loading_finished)
        
        # Connect view signals to controller slots
        self._view.user_action.connect(self._on_user_action)
        self._view.close_requested.connect(self._on_close_requested)

    @property
    def model(self) -> BaseModel:
        """Get the model instance."""
        return self._model
    
    @property
    def view(self):
        """Get the view instance."""
        return self._view
    
    def handle_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Handle user actions from the view.
        
        Args:
            action_name: Name of the action performed
            action_data: Data associated with the action
        """
        raise NotImplementedError("Subclasses must implement handle_user_action()")
    
    @pyqtSlot(str)
    def _on_model_error(self, error_message: str) -> None:
        """
        Handle model errors.
        
        Args:
            error_message: Error message from the model
        """
        self._view.show_error(error_message)

    @pyqtSlot()
    def _on_loading_started(self) -> None:
        """Handle start of loading operation."""
        self._view.show_loading()
    
    @pyqtSlot()
    def _on_loading_finished(self) -> None:
        """Handle completion of loading operation."""
        self._view.hide_loading()
    
    @pyqtSlot(str, object)
    def _on_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Handle user actions from the view.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        self.handle_user_action(action_name, action_data)
    
    @pyqtSlot()
    def _on_close_requested(self) -> None:
        """Handle close request from view."""
        self._view.close()
    
    def show_view(self) -> None:
        """Show the view."""
        self._view.show()
    
    def hide_view(self) -> None:
        """Hide the view."""
        self._view.hide()
    
    def close_view(self) -> None:
        """Close the view."""
        self._view.close()
