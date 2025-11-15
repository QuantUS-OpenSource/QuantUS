"""
Base View class for MVC architecture
"""

from typing import Any, Optional
from PyQt6.QtWidgets import QWidget


class BaseViewMixin:
    """
    Mixin class for all views in the MVC architecture.
    
    Views handle the UI presentation and user interactions,
    following the current QuantUS GUI style with single menu per .ui file.
    """
    
    def __init_base_view__(self, parent: Optional[QWidget] = None):
        """Initialize base view properties."""
        self._is_loading = False  # Track loading state
        self._setup_base_style()  # Setup base UI styling to match existing QuantUS style

    def _setup_base_style(self) -> None:
        """Apply base styling consistent with QuantUS GUI."""
        self.setStyleSheet("""
            QWidget {
                background: rgb(42, 42, 42);
                color: white;
            }
            QPushButton {
                background-color: rgb(60, 60, 60);
                border: 1px solid rgb(80, 80, 80);
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgb(70, 70, 70);
            }
            QPushButton:pressed {
                background-color: rgb(50, 50, 50);
            }
            QComboBox {
                background-color: rgb(60, 60, 60);
                border: 1px solid rgb(80, 80, 80);
                padding: 4px;
                border-radius: 4px;
            }
            QLineEdit {
                background-color: rgb(60, 60, 60);
                border: 1px solid rgb(80, 80, 80);
                padding: 4px;
                border-radius: 4px;
            }
            QLabel {
                color: white;
            }
        """)
    
    def _setup_ui(self) -> None:
        """
        Setup the user interface.
        Should call setupUi() from the compiled .ui file.
        """
        raise NotImplementedError("Subclasses must implement _setup_ui()")
    
    def _connect_signals(self) -> None:
        """
        Connect UI signals to appropriate handlers.
        """
        raise NotImplementedError("Subclasses must implement _connect_signals()")
    
    def clear_error(self) -> None:
        """
        Clear any error messages displayed in the view.
        """
        # This can be overridden by subclasses for custom error handling
        pass
    
    def show_error(self, error_message: str) -> None:
        """
        Display error message to user.
        
        Args:
            error_message: Error message to display
        """
        # This can be overridden by subclasses for custom error display
        print(f"Error: {error_message}")

    def hide_loading(self) -> None:
        """Hide any loading indicators in the view."""
        # This can be overridden by subclasses to hide loading indicators
        self._is_loading = False

    def show_loading(self) -> None:
        """Show loading indicators in the view."""
        # This can be overridden by subclasses to show loading indicators
        self._is_loading = True
