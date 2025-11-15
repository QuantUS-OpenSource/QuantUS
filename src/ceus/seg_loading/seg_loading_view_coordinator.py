"""
Segmentation Loading View Coordinator for MVC architecture

This coordinator manages the workflow between segmentation type selection, file selection,
ROI drawing, and preview widgets, providing a unified interface for the controller.
It manages widgets that are designed to be embedded in the main application widget stack.
"""

from typing import Any, Optional
from PyQt6.QtWidgets import QWidget, QStackedWidget
from PyQt6.QtCore import pyqtSignal

from ..mvc.base_view import BaseViewMixin
from .views.seg_type_selection_widget import SegTypeSelectionWidget
from .views.seg_file_selection_widget import SegFileSelectionWidget
from .views.draw_roi_widget import DrawROIWidget
from .views.draw_voi_widget import DrawVOIWidget
from engines.ceus.src.data_objs import UltrasoundImage, CeusSeg


class SegLoadingViewCoordinator(QStackedWidget):
    """
    Coordinator for segmentation loading widgets.
    
    Manages the workflow between segmentation type selection, file selection,
    ROI drawing, and preview widgets using a QStackedWidget. This allows 
    embedding in the main application widget stack for a seamless navigation experience.
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


    def __init__(self, image_data: UltrasoundImage, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_data = image_data
        
        # Widget instances
        self._seg_type_widget: Optional[SegTypeSelectionWidget] = None
        self._seg_file_widget: Optional[SegFileSelectionWidget] = None
        self._voi_drawing_widget: Optional[DrawVOIWidget] = None
        
        # Current state
        self._selected_seg_type: Optional[str] = None
        self._seg_data: Optional[CeusSeg] = None

        # Start with segmentation type selection
        self.show_seg_type_selection()

    # ============================================================================
    # CONTROLLER INPUT ROUTING - Route inputs from controller to appropriate widget
    # ============================================================================
    
    def set_seg_loaders(self, loader_names: list) -> None:
        """
        Set available segmentation loaders in the dropdown.
        
        Args:
            loader_names: List of formatted segmentation loader names
        """
        if self._seg_type_widget:
            self._seg_type_widget.set_seg_loaders(loader_names)

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
    
    def reset_to_seg_type_selection(self) -> None:
        """Reset to the segmentation type selection widget."""
        # Remove other widgets if they exist
        widgets_to_remove = [
            self._seg_file_widget, 
            self._voi_drawing_widget,
        ]
        
        for widget in widgets_to_remove:
            if widget:
                self.removeWidget(widget)
                widget.deleteLater()
        
        # Reset widget references
        self._seg_file_widget = None
        self._roi_drawing_widget = None
        self._voi_drawing_widget = None
        self._seg_preview_widget = None
        
        # Clear state
        self._selected_seg_type = None
        self._seg_data = None
        
        # Return to seg type widget
        if self._seg_type_widget:
            self._seg_type_widget.clear_error()
            self.setCurrentWidget(self._seg_type_widget)
        else:
            raise RuntimeError("Segmentation type widget not initialized")
        
    def show_seg_type_selection(self) -> None:
        """Show the segmentation type selection widget."""
        # Create and setup segmentation type selection widget
        self._seg_type_widget = SegTypeSelectionWidget(self._image_data)
        self._seg_type_widget.setup_ui()
        self._seg_type_widget.connect_signals()

        # Connect signals to handle user actions
        self._seg_type_widget.seg_type_selected.connect(self._on_seg_type_selected)
        self._seg_type_widget.close_requested.connect(self.close_requested.emit)
        self._seg_type_widget.back_requested.connect(self.back_requested.emit)

        # Add to stack and show
        self.addWidget(self._seg_type_widget)
        self.setCurrentWidget(self._seg_type_widget)

    def show_file_selection(self, file_extensions: list) -> None:
        """
        Show the file selection widget after seg type is selected.
        
        Args:
            file_extensions: List of file extensions for the selected segmentation type
        """
        if not self._selected_seg_type:
            self.show_error("No segmentation type selected")
            return
        
        # Create file selection widget
        self._seg_file_widget = SegFileSelectionWidget(
            self._selected_seg_type, 
            self._image_data,
            file_extensions
        )
        
        # Connect signals
        self._seg_file_widget.file_selected.connect(self._on_file_selected)
        self._seg_file_widget.back_requested.connect(self.reset_to_seg_type_selection)
        self._seg_file_widget.close_requested.connect(self.close_requested.emit)
        
        # Add to stack and show
        self.addWidget(self._seg_file_widget)
        self.setCurrentWidget(self._seg_file_widget)

    def show_voi_drawing(self) -> None:
        """Show the VOI drawing widget."""
        self._voi_drawing_widget = DrawVOIWidget(self._image_data)

        # Connect signals to handle user actions
        self._voi_drawing_widget.back_requested.connect(self.reset_to_seg_type_selection)
        self._voi_drawing_widget.close_requested.connect(self.close_requested.emit)

        # Add to stack and show
        self.addWidget(self._voi_drawing_widget)
        self.setCurrentWidget(self._voi_drawing_widget)

    def show_roi_drawing(self) -> None:
        """Show the ROI drawing widget."""
        self._roi_drawing_widget = DrawROIWidget(self._image_data)

        # Connect signals to handle user actions
        self._roi_drawing_widget.back_requested.connect(self.reset_to_seg_type_selection)
        self._roi_drawing_widget.close_requested.connect(self.close_requested.emit)

        # Add to stack and show
        self.addWidget(self._roi_drawing_widget)
        self.setCurrentWidget(self._roi_drawing_widget)

    # ============================================================================
    # USER ACTION HANDLING - Process user interactions and communicate with controller
    # ============================================================================

    def _on_seg_type_selected(self, seg_type_name: str) -> None:
        """
        Handle segmentation type selection from the seg type widget.
        
        Args:
            seg_type_name: Selected segmentation type name
        """
        self._selected_seg_type = seg_type_name
        
        # Emit signal to controller
        self._emit_user_action('seg_type_selected', seg_type_name)
    
    def _on_file_selected(self, file_data: dict) -> None:
        """
        Handle file selection from the file selection widget.
        
        Args:
            file_data: Dictionary with selected file path and seg type
        """
        self._emit_user_action('load_segmentation', file_data)
    
    def _emit_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Emit user action signal to controller.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        self.user_action.emit(action_name, action_data)
