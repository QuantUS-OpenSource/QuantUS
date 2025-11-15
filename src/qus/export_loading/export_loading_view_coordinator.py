"""
Export Loading View Coordinator for QuantUS GUI

This coordinator manages the export loading views and their interactions
with the controller and model, using a modular stacked widget approach.
"""

from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from .ui.export_loading_ui import Ui_exportLoading
from .views.export_config_widget import ExportConfigWidget
from .views.export_functions_widget import ExportFunctionsWidget


class ExportLoadingViewCoordinator(QWidget):
    """View coordinator for export loading functionality."""
    
    # Signals
    export_completed = pyqtSignal(object)  # Emitted when export is completed
    export_failed = pyqtSignal(str)  # Emitted when export fails
    user_action = pyqtSignal(str, object)  # action_name, action_data
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    
    def __init__(self, controller, parent=None):
        """Initialize the export loading view coordinator.
        
        Args:
            controller: The export loading controller.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.controller = controller
        self._ui = Ui_exportLoading()
        
        # Initialize views
        self._export_config_widget = None
        self._export_functions_widget = None
        self._current_step_index = 0  # 0=config, 1=functions
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        self.update_views()
    
    def _setup_ui(self):
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for export loading - use the main layout
        self.setLayout(self._ui.main_layout)
        
        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.export_loading_layout, 10)
        
        # Ensure the layout fills the entire widget
        self._ui.main_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.main_layout.setSpacing(0)
        self._ui.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.full_screen_layout.setSpacing(0)
        
        # Create view widgets
        self._export_config_widget = ExportConfigWidget(self.controller)
        self._export_functions_widget = ExportFunctionsWidget(self.controller)
        
        # Add widgets to stacked widget
        self._ui.export_stack.addWidget(self._export_config_widget)
        self._ui.export_stack.addWidget(self._export_functions_widget)

        # Constrain button sizes and styles to match visualization menu
        try:
            self._ui.execute_button.setFixedHeight(41)
            self._ui.execute_button.setFixedWidth(250)
        except Exception:
            pass
    
    def _connect_signals(self):
        """Connect signals between widgets and controller."""
        # Connect widget signals
        if self._export_config_widget:
            self._export_config_widget.export_path_changed.connect(
                self.controller.set_export_path
            )
        
        if self._export_functions_widget:
            self._export_functions_widget.export_functions_selected.connect(
                self.controller.set_export_functions
            )
        
        # Connect button signals
        self._ui.execute_button.clicked.connect(self._on_execute_clicked)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
    
    def update_views(self):
        """Update all views with current data."""
        # Highlight the current section in the sidebar
        self._highlight_sidebar()
        
        # Update image and phantom names in the sidebar
        self._update_image_phantom_names()
        
        # Update export config widget
        if self._export_config_widget:
            current_path = self.controller.get_export_path()
            if current_path:
                self._export_config_widget.set_export_path(current_path)
        
        # Update export functions widget
        if self._export_functions_widget:
            current_type = self.controller.get_selected_export_type()
            if current_type:
                available_funcs = self.controller.get_available_export_functions(current_type)
                self._export_functions_widget.update_available_functions(available_funcs)
                
                # Get recommended functions based on visualization results
                try:
                    viz_funcs = self.controller._viz_controller.get_selected_visualization_functions()
                    recommended = self.controller._recommend_export_functions(viz_funcs)
                    self._export_functions_widget.set_recommended_functions(recommended)
                except Exception:
                    self._export_functions_widget.set_recommended_functions([])
                
                # Set selected functions
                selected_funcs = self.controller.get_selected_export_functions()
                self._export_functions_widget.set_selected_functions(selected_funcs)
        
        # Enable execute button based on validation
        self._ui.execute_button.setEnabled(self.controller.validate_export_selection())
    
    def _highlight_sidebar(self):
        """Highlight the current section in the sidebar to match visualization menu."""
        purple = "rgb(99, 0, 174)"
        dark = "rgb(49, 0, 124)"

        # Set all sections to purple by default
        for sidebar in [
            self._ui.imageSelectionSidebar,
            self._ui.segmentationSidebar,
            self._ui.analysisParamsSidebar,
            self._ui.rfAnalysisSidebar,
            self._ui.exportResultsSidebar,
        ]:
            sidebar.setStyleSheet(f"""
                QFrame {{
                    background-color: {purple};
                    border: 1px solid black;
                }}
            """)

        # Highlight current section (Data Export) - use lighter purple like Radio Frequency Analysis
        self._ui.exportResultsSidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {purple};
                border: 1px solid black;
            }}
        """)

    def _update_image_phantom_names(self):
        """Update the image and phantom names in the sidebar from visualization controller."""
        try:
            # Get analysis data from visualization controller
            viz_controller = self.controller._viz_controller
            if viz_controller and hasattr(viz_controller, '_analysis_data'):
                analysis_data = viz_controller._analysis_data
                if analysis_data and hasattr(analysis_data, 'image_data'):
                    image_data = analysis_data.image_data
                    
                    # Update image name
                    if hasattr(image_data, 'scan_name') and image_data.scan_name:
                        self._ui.image_path_input.setText(image_data.scan_name)
                    else:
                        self._ui.image_path_input.setText("No image loaded")
                    
                    # Update phantom name
                    if hasattr(image_data, 'phantom_name') and image_data.phantom_name:
                        self._ui.phantom_path_input.setText(image_data.phantom_name)
                    else:
                        self._ui.phantom_path_input.setText("No phantom loaded")
                        
                    print(f"DEBUG: Updated export sidebar - scan_name: {getattr(image_data, 'scan_name', 'None')}, phantom_name: {getattr(image_data, 'phantom_name', 'None')}")
                else:
                    # Fallback to default values if no analysis data
                    self._ui.image_path_input.setText("Sample filename")
                    self._ui.phantom_path_input.setText("Sample filename")
                    print("DEBUG: No analysis data available for export sidebar")
            else:
                # Fallback to default values if no visualization controller
                self._ui.image_path_input.setText("Sample filename")
                self._ui.phantom_path_input.setText("Sample filename")
                print("DEBUG: No visualization controller available for export sidebar")
        except Exception as e:
            print(f"DEBUG: Error updating image/phantom names in export sidebar: {e}")
            # Fallback to default values on error
            self._ui.image_path_input.setText("Sample filename")
            self._ui.phantom_path_input.setText("Sample filename")
    
    def _on_execute_clicked(self):
        """Handle execute export button click."""
        try:
            export_obj = self.controller.perform_export()
            self.user_action.emit('export_loading_completed', {
                'export_type': self.controller.get_selected_export_type(),
                'export_functions': self.controller.get_selected_export_functions(),
                'export_path': self.controller.get_export_path(),
            })
        except Exception as e:
            # Simple stderr feedback; can be replaced with QMessageBox
            print(f"ERROR: Export failed: {e}")
            self.export_failed.emit(str(e))
    

    
    def _on_back_clicked(self):
        """Handle back button click."""
        self.back_requested.emit()


