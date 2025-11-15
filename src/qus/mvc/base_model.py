"""
Base Model class for MVC architecture
"""

from typing import Any, Dict, Optional
from PyQt6.QtCore import QObject, pyqtSignal


class BaseModel(QObject):
    """
    Base class for all models in the MVC architecture.
    
    Models handle data management and business logic, communicating
    directly with the backend services and data objects.
    """
    
    # Signals for notifying views of data changes
    error_occurred = pyqtSignal(str)   # Emitted when an error occurs
    loading_started = pyqtSignal()     # Emitted when async operation starts
    loading_finished = pyqtSignal()    # Emitted when async operation completes
    
    def __init__(self):
        super().__init__()
        self._is_loading: bool = False
    
    @property
    def is_loading(self) -> bool:
        """Check if model is currently loading data."""
        return self._is_loading
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before processing.
        
        Args:
            input_data: Dictionary of input parameters to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_input()")
    
    def _set_loading(self, loading: bool) -> None:
        """
        Set loading state and emit appropriate signals.
        
        Args:
            loading: Whether model is currently loading
        """
        self._is_loading = loading
        if loading:
            self.loading_started.emit()
        else:
            self.loading_finished.emit()
    
    def _emit_error(self, error_message: str) -> None:
        """
        Emit error signal with message.
        
        Args:
            error_message: Description of the error
        """
        self.error_occurred.emit(error_message)
