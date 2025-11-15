"""
Image Loading View Coordinator for MVC architecture

This coordinator manages the workflow between scan type selection and file selection
widgets, providing a unified interface for the controller. It manages widgets that
are designed to be embedded in the main application widget stack.
"""

from typing import Any, Optional
from PyQt6.QtWidgets import QWidget, QStackedWidget
from PyQt6.QtCore import pyqtSignal, QObject

from ..mvc.base_view import BaseViewMixin
from .views.scan_type_widget import ScanTypeSelectionWidget
from .views.file_selection_widget import FileSelectionWidget
from engines.ceus.src.data_objs import UltrasoundImage


class ImageLoadingViewCoordinator(QStackedWidget):
    """
    Coordinator for image loading widgets.
    
    Manages the workflow between scan type selection and file selection
    widgets using a QStackedWidget. This allows embedding in the main
    application widget stack for a seamless navigation experience.
    """
    
    # ============================================================================
    # SIGNALS - Communication with controller
    # ============================================================================
    
    user_action = pyqtSignal(str, object)  # action_name, action_data
    close_requested = pyqtSignal()
    
    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Widget instances
        self._scan_type_widget: Optional[ScanTypeSelectionWidget] = None
        self._file_selection_widget: Optional[FileSelectionWidget] = None
        
        # Current state
        self._selected_scan_type: Optional[str] = None

        # Start with scan type selection
        self.show_scan_type_selection()

    # ============================================================================
    # CONTROLLER INPUT ROUTING - Route inputs from controller to appropriate widget
    # ============================================================================

    def set_scan_loaders(self, loader_names: list) -> None:
        """
        Set available scan loaders in the dropdown.
        
        Args:
            loader_names: List of formatted scan loader names
        """
        if self._scan_type_widget:
            self._scan_type_widget.set_scan_loaders(loader_names)
    
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
        """Clear any displayed error message in the current widget."""
        current_widget: BaseViewMixin = self.currentWidget()
        current_widget.clear_error()

    # ============================================================================
    # WIDGET NAVIGATION MANAGEMENT - Control workflow between widgets
    # ============================================================================
    
    def reset_to_scan_type_selection(self) -> None:
        """Reset to the scan type selection widget."""
        # Remove file selection widget if it exists
        if self._file_selection_widget:
            self.removeWidget(self._file_selection_widget)
            self._file_selection_widget.deleteLater()
            self._file_selection_widget = None

        # Reset widget references
        self._file_selection_widget = None
        
        # Clear state
        self._selected_scan_type = None
        
        # Return to scan type widget
        if self._scan_type_widget:
            self._scan_type_widget.clear_error()
            self.setCurrentWidget(self._scan_type_widget)
        else:
            raise RuntimeError("Scan type widget not initialized")

    def show_scan_type_selection(self) -> None:
        """Show the scan type selection widget."""
        # Create and setup scan type selection widget
        self._scan_type_widget = ScanTypeSelectionWidget()

        # Connect signals to handle user actions
        self._scan_type_widget.scan_type_selected.connect(self._on_scan_type_selected)
        self._scan_type_widget.close_requested.connect(self.close_requested.emit)
        
        # Add to stack and show
        self.addWidget(self._scan_type_widget)
        self.setCurrentWidget(self._scan_type_widget)

    def show_file_selection(self, file_extensions: list, loading_options: list) -> None:
        """
        Show the file selection widget after scan type is selected.
        Args:
            file_extensions: List of file extensions for the selected scan type
            loading_options: List of loading options for the selected scan type
        """
        if not self._selected_scan_type:
            self.show_error("No scan type selected")
            return
        
        # Create file selection widget
        self._file_selection_widget = FileSelectionWidget(
            self._selected_scan_type, 
            file_extensions,
            loading_options
        )
        
        # Connect signals
        self._file_selection_widget.files_selected.connect(self._on_files_selected)
        self._file_selection_widget.back_requested.connect(self.reset_to_scan_type_selection)
        self._file_selection_widget.close_requested.connect(self.close_requested.emit)
        
        # Add to stack and show
        self.addWidget(self._file_selection_widget)
        self.setCurrentWidget(self._file_selection_widget)
    
    # ============================================================================
    # USER ACTION HANDLING - Process user interactions and communicate with controller
    # ============================================================================

    def _on_scan_type_selected(self, scan_type_name: str) -> None:
        """
        Handle scan type selection from the scan type widget.
        
        Args:
            scan_type_name: Selected scan type name
        """
        self._selected_scan_type = scan_type_name
        
        # Emit signal to controller
        self._emit_user_action('scan_type_selected', scan_type_name)
    
    def _on_files_selected(self, file_data: dict) -> None:
        """
        Handle file selection from the file selection widget.
        
        Args:
            file_data: Dictionary with selected file paths and parameters
        """
        self._emit_user_action('load_image', file_data)
    
    def _emit_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Emit user action signal to controller.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        self.user_action.emit(action_name, action_data)
