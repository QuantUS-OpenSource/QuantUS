"""
Export Loading Controller for QuantUS GUI

This controller manages the export step after visualization, allowing the user
to choose export type (e.g., CSV), select export functions, configure kwargs,
and specify output path, then perform the export.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal


class ExportLoadingController(QObject):
    """Controller for export loading functionality."""

    # Signals
    user_action = pyqtSignal(str, object)  # action_name, action_data
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, model, visualization_controller, parent=None):
        """Initialize the export loading controller.

        Args:
            model: The unified application model.
            visualization_controller: The visualization controller to fetch last visualization object.
            parent: Parent object.
        """
        super().__init__(parent)
        self._model = model
        self._viz_controller = visualization_controller
        self._view_coordinator = None

        # Current state
        self._selected_export_type: Optional[str] = None
        self._selected_export_funcs: List[str] = []
        self._export_kwargs: Dict[str, Any] = {}
        self._export_path: Optional[str] = None

        # Defaults - Always use CSV export type
        export_types, _ = self._model.get_data_export_types()
        if export_types:
            # Always pick 'csv' if available
            if 'csv' in export_types:
                self._selected_export_type = 'csv'
            else:
                # Fallback to first available type if CSV is not available
                self._selected_export_type = list(export_types.keys())[0]

        # Pre-select export functions based on chosen visualization functions
        try:
            viz_funcs = visualization_controller.get_selected_visualization_functions()
            self._selected_export_funcs = self._recommend_export_functions(viz_funcs)
        except Exception:
            self._selected_export_funcs = []

    # ----------------------------- Query methods -----------------------------
    def get_available_export_types(self) -> Dict[str, Any]:
        return self._model.get_data_export_types()[0]

    def get_available_export_functions(self, export_type: str) -> Dict[str, Any]:
        return self._model.get_data_export_types()[1].get(export_type, {})

    def get_selected_export_type(self) -> Optional[str]:
        return self._selected_export_type

    def get_selected_export_functions(self) -> List[str]:
        return self._selected_export_funcs

    def get_export_kwargs(self) -> Dict[str, Any]:
        return self._export_kwargs

    def get_export_path(self) -> Optional[str]:
        return self._export_path

    # ----------------------------- Mutators ----------------------------------
    def set_export_type(self, export_type: str) -> None:
        # Always force CSV export type - ignore any attempts to change it
        if export_type != 'csv':
            print("Warning: Export type is locked to 'csv'. Attempted change to '{}' ignored.".format(export_type))
        
        # Ensure CSV is selected
        self._selected_export_type = 'csv'
        
        # Recompute recommendations when type changes
        try:
            viz_funcs = self._viz_controller.get_selected_visualization_functions()
            self._selected_export_funcs = self._recommend_export_functions(viz_funcs)
        except Exception:
            self._selected_export_funcs = []
        if self._view_coordinator:
            self._view_coordinator.update_views()

    def set_export_functions(self, export_funcs: List[str]) -> None:
        self._selected_export_funcs = export_funcs
        if self._view_coordinator:
            self._view_coordinator.update_views()

    def set_export_kwargs(self, kwargs: Dict[str, Any]) -> None:
        self._export_kwargs = kwargs
        if self._view_coordinator:
            self._view_coordinator.update_views()

    def set_export_path(self, path: str) -> None:
        self._export_path = path
        if self._view_coordinator:
            self._view_coordinator.update_views()

    # ----------------------------- Validation --------------------------------
    def validate_export_selection(self) -> bool:
        if not self._selected_export_type:
            return False
        if not self._selected_export_funcs:
            return False
        if not self._export_path:
            return False
        # Basic check: ensure funcs exist
        available_funcs = self.get_available_export_functions(self._selected_export_type)
        for func in self._selected_export_funcs:
            if func not in available_funcs:
                return False
        # CSV path must end with .csv if exporting as single CSV
        if self._selected_export_type == 'csv' and self._export_path and not self._export_path.lower().endswith('.csv'):
            # allow functions that require a folder to not end with csv; descriptive validation is complex, so we don't block here
            pass
        return True

    # ----------------------------- Execution ---------------------------------
    def perform_export(self) -> Any:
        if not self.validate_export_selection():
            raise ValueError("Invalid export selection")

        viz_obj = None
        try:
            viz_obj = self._viz_controller.get_last_visualization_obj()
        except Exception:
            viz_obj = None
        if viz_obj is None:
            # Attempt to regenerate via controller
            try:
                viz_obj = self._viz_controller.execute_visualization()
            except Exception as e:
                raise RuntimeError(f"Could not obtain visualization object for export: {e}")

        # Auto-fill required kwargs if possible (e.g., output_folder)
        try:
            funcs_map = self.get_available_export_functions(self._selected_export_type)
            for name in self._selected_export_funcs:
                fn = funcs_map.get(name)
                if fn is None:
                    continue
                if hasattr(fn, 'required_kwargs'):
                    for kw in getattr(fn, 'required_kwargs', []):
                        if kw not in self._export_kwargs:
                            if kw == 'output_folder' and self._export_path:
                                self._export_kwargs['output_folder'] = str(Path(self._export_path).parent)
        except Exception:
            pass

        export_obj = self._model.execute_data_export(
            self._selected_export_type,
            viz_obj,
            self._export_path,
            self._selected_export_funcs,
            **self._export_kwargs,
        )
        return export_obj

    # ----------------------------- Recommendations ---------------------------
    def _recommend_export_functions(self, visualization_funcs: list[str]) -> list[str]:
        """Map visualization choices to recommended export functions.
        
        Always use descr_vals for all analysis functions.
        Add hscan_stats for hscan function.
        """
        # Get available export functions
        funcs_map = self.get_available_export_functions(self._selected_export_type or '')
        
        # Always include descr_vals if available
        recommended = ['descr_vals'] if 'descr_vals' in funcs_map else []
        
        # Add hscan_stats only if hscan visualization is present
        if 'plot_hscan_result' in (visualization_funcs or []) and 'hscan_stats' in funcs_map:
            recommended.append('hscan_stats')
        
        return recommended

    # ----------------------------- View wiring --------------------------------
    def get_widget(self):
        if self._view_coordinator is None:
            from .export_loading_view_coordinator import ExportLoadingViewCoordinator
            self._view_coordinator = ExportLoadingViewCoordinator(self)
            self._view_coordinator.user_action.connect(self._on_user_action)
            self._view_coordinator.back_requested.connect(self._on_back_requested)
            self._view_coordinator.close_requested.connect(self._on_close_requested)
        return self._view_coordinator

    def cleanup(self):
        if self._view_coordinator:
            self._view_coordinator.deleteLater()
            self._view_coordinator = None

    # Signal forwarding
    def _on_user_action(self, action_name: str, action_data: Any) -> None:
        self.user_action.emit(action_name, action_data)

    def _on_back_requested(self) -> None:
        self.back_requested.emit()

    def _on_close_requested(self) -> None:
        self.close_requested.emit()


