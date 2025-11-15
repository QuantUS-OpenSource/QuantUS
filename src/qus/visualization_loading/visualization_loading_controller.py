"""
Visualization Loading Controller for QuantUS GUI

This controller manages the visualization loading process, including selecting
visualization types and functions based on the analysis results.
"""

from pathlib import Path 
from typing import Dict, List, Optional, Any

from PyQt6.QtCore import QObject, pyqtSignal
from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis
from engines.qus.quantus.data_objs.visualizations import ParamapDrawingBase
from .visualization_loading_view_coordinator import VisualizationLoadingViewCoordinator


class VisualizationLoadingController(QObject):
    """Controller for visualization loading functionality."""
    
    # Signals
    user_action = pyqtSignal(str, object)  # action_name, action_data
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    
    def __init__(self, model, analysis_data: ParamapAnalysis, parent=None):
        """Initialize the visualization loading controller.
        
        Args:
            model: The application model containing analysis data.
            analysis_data: The analysis data to visualize.
            parent: Parent object.
        """
        super().__init__(parent)
        self._model = model
        self._analysis_data = analysis_data
        self._view_coordinator = None
        
        # Current state
        self._selected_visualization_type = None # default type
        self._selected_visualization_funcs = []
        self._visualization_kwargs = {}
        self._visualization_folder: str = None
        self._visualization_obj: Optional[ParamapDrawingBase] = None

        # Setup available visualization options based on analysis results
        self._setup_visualization_options()

        # # Directory for visualization assets (under repo root for easy access)
        # self._preview_output_dir = self._compute_preview_dir()

        # Connect view signals once view is created
        # We defer creation to get_widget; connections are established there

    def _setup_visualization_options(self) -> None:
        """Setup available visualization functions based on analysis results."""
        # "paramap" is the only type currently supported
        self._selected_visualization_type = "paramap"
        self.get_widget()  # Ensure view coordinator is created

        available_func_names = self._model.get_compatible_visualization_funcs(self._selected_visualization_type)
        self._view_coordinator.show_function_selection(available_func_names)
    
    def get_widget(self):
        """Get the main widget for this controller.
        
        Returns:
            The view coordinator widget.
        """
        if self._view_coordinator is None:
            self._view_coordinator = VisualizationLoadingViewCoordinator(self._analysis_data.image_data,
                                                self._analysis_data.seg_data, self._analysis_data.config, self._analysis_data)
            # Forward coordinator signals to application controller
            self._view_coordinator.user_action.connect(self._on_user_action)
            self._view_coordinator.back_requested.connect(self._on_back_requested)
            self._view_coordinator.close_requested.connect(self._on_close_requested)
        return self._view_coordinator
    
    def cleanup(self):
        """Clean up resources."""
        if self._view_coordinator:
            self._view_coordinator.deleteLater()
            self._view_coordinator = None

    # ---------------------------------------------------------------------
    # Signal forwarding to app controller
    # ---------------------------------------------------------------------
    def _on_user_action(self, action_name: str, action_data: Any) -> None:
        """Handle user actions from the view coordinator.
        
        Args:
            action_name: Name of the action.
            action_data: Data associated with the action.
        """
        if action_name == "visualization_functions_selected":
            self._handle_analysis_functions_selection(action_data)
        
        self.user_action.emit(action_name, action_data)

    def _on_back_requested(self) -> None:
        self.back_requested.emit()

    def _on_close_requested(self) -> None:
        self.close_requested.emit()

    def _handle_analysis_functions_selection(self, action_data: Dict[str, Any]) -> None:
        """Handle visualization functions selection.
        
        Args:
            action_data: Dictionary containing selected function names and the destination folder.
        """
        self._selected_visualization_funcs = action_data["functions"]
        self._visualization_folder = action_data["dest_folder"]
        self._visualization_kwargs['paramap_folder_path'] = self._visualization_folder
        self._visualization_kwargs['hide_all_visualizations'] = False # always output, even if just to internal folder

        self._visualization_obj = self._model.execute_visualization(
            self._selected_visualization_type,
            self._analysis_data,
            self._selected_visualization_funcs,
            **self._visualization_kwargs
        )
        
        self._view_coordinator.show_visualization_previews(self._visualization_folder)
