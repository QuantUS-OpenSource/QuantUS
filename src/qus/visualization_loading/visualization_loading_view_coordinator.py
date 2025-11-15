"""
Visualization Loading View Coordinator for QuantUS GUI

This coordinator manages the visualization loading views and their interactions
with the controller and model.
"""

from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal

from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis
from engines.qus.quantus.data_objs.analysis_config import RfAnalysisConfig
from engines.qus.quantus.data_objs.image import UltrasoundRfImage
from engines.qus.quantus.data_objs.seg import BmodeSeg
from ..mvc.base_view import BaseViewMixin
from .views.visualization_function_selection_widget import VisualizationFunctionSelectionWidget
from .views.visualization_preview_2d_widget import VisualizationPreview2DWidget


class VisualizationLoadingViewCoordinator(QStackedWidget, BaseViewMixin):
    """
    Coordinator for visualization loading widgets.

    Manages the workflow between visualization type selection, function selection,
    parameter configuration, and preview widgets using a stacked layout. This allows
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

    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, config_data: RfAnalysisConfig, 
                 analysis_data: ParamapAnalysis, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        self._analysis_data = analysis_data

        # Widget instances
        self._visualization_function_widget: Optional[VisualizationFunctionSelectionWidget] = None
        self._visualization_preview_widget: Optional[VisualizationPreview2DWidget] = None

        # Note: Visualization type selection is now skipped - Paramap is automatically selected
        # The controller will call show_function_selection directly
    
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

    def show_function_selection(self, available_functions: List[str]) -> None:
        """
        Show the analysis function selection widget.
        
        Args:
            available_functions: List of available functions for the selected analysis type
        """
        if self._visualization_function_widget is None:
            self._visualization_function_widget = VisualizationFunctionSelectionWidget(self._image_data, self._seg_data,
                                                                                     self._config_data, self._analysis_data)
            self._visualization_function_widget.visualization_info.connect(self._on_functions_selected)
            self._visualization_function_widget.back_requested.connect(self._on_function_selection_back)
            self._visualization_function_widget.close_requested.connect(self.close_requested.emit)
            self.addWidget(self._visualization_function_widget)

        self._visualization_function_widget.update_available_functions(available_functions)
        self.setCurrentWidget(self._visualization_function_widget)
        self._visualization_function_widget.clear_error()

    def show_visualization_previews(self, visualization_folder: str) -> None:
        """
        Show the visualization previews widget.
        Args:
            visualization_folder: Folder where visualization results are stored
        """
        if self._visualization_preview_widget is None:
            self._visualization_preview_widget = VisualizationPreview2DWidget(self._image_data)
            self._visualization_preview_widget.back_requested.connect(self._on_visualization_preview_back)
            self.addWidget(self._visualization_preview_widget)

        self.setCurrentWidget(self._visualization_preview_widget)
        self._visualization_preview_widget.clear_error()

        self._visualization_preview_widget.set_visualization_folder(visualization_folder)
        self.setCurrentWidget(self._visualization_preview_widget)
        self._visualization_preview_widget.clear_error()

    # ============================================================================
    # EVENT HANDLERS - Handle events from child widgets
    # ============================================================================

    def _on_visualization_preview_back(self) -> None:
        """Handle back navigation from visualization preview."""
        self.show_function_selection(self._visualization_function_widget._available_functions if self._visualization_function_widget else [])

    def _on_functions_selected(self, selected_func_names: Dict[str, Any]) -> None:
        """
        Handle visualization functions selection.

        Args:
            selected_func_names: Dictionary containing selected function names and the destination folder.
        """
        self._emit_user_action("visualization_functions_selected", selected_func_names)

    def _on_params_configured(self, params: dict) -> None:
        """
        Handle visualization parameters configuration.

        Args:
            params: Dictionary containing visualization parameters
        """
        self._emit_user_action("visualization_execution_started", params)

    def _on_function_selection_back(self) -> None:
        """Handle back navigation from function selection."""
        # Since we skip analysis type selection, go back to the main application flow
        self.back_requested.emit()

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
