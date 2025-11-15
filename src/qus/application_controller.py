"""
Main Application Controller for QuantUS GUI MVC architecture
"""

import sys
from typing import Optional
import qdarktheme
from PyQt6.QtWidgets import QApplication, QStackedWidget
from PyQt6.QtCore import QObject, pyqtSignal, QEvent

from .application_model import ApplicationModel
from src.qus.image_loading.image_loading_view_coordinator import ImageLoadingViewCoordinator
from src.qus.image_loading.image_loading_controller import ImageLoadingController
from src.qus.seg_loading.seg_loading_controller import SegmentationLoadingController
from src.qus.config_loading.config_loading_controller import ConfigLoadingController
from src.qus.analysis_loading.analysis_loading_controller import AnalysisLoadingController
from src.qus.visualization_loading.visualization_loading_controller import VisualizationLoadingController
from src.qus.export_loading.export_loading_controller import ExportLoadingController
from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig


class ApplicationController(QObject):
    """
    Main application controller that manages navigation between different screens
    and coordinates the overall application workflow using a unified application model.
    
    Follows MVC architecture with a single application-wide model.
    """
    
    # Signals for application-level events
    application_exit = pyqtSignal()
    
    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app
        self._widget_stack = QStackedWidget()
        self._widget_stack.setStyleSheet("QWidget {\n"
        "    background: rgb(42, 42, 42);\n"
        "}")
        
        # Unified application model
        self._model = ApplicationModel()
        
        # Controllers for different screens (using the same model)
        self._image_loading_controller: Optional[ImageLoadingController] = None
        self._segmentation_controller: Optional[SegmentationLoadingController] = None
        self._config_loading_controller: Optional[ConfigLoadingController] = None
        self._analysis_loading_controller: Optional[AnalysisLoadingController] = None
        self._visualization_loading_controller: Optional[VisualizationLoadingController] = None
        self._export_loading_controller: Optional[ExportLoadingController] = None
        
        # Data storage
        self._image_data: Optional[UltrasoundRfImage] = None
        self._seg_data: Optional[BmodeSeg] = None
        self._config_data: Optional[RfAnalysisConfig] = None
        
        # Setup main widget
        self._setup_main_widget()
        
        # Install event filter for window management
        self._widget_stack.installEventFilter(self)
        
        # Connect model signals
        self._connect_model_signals()
        
        # Initialize first screen
        self._initialize_image_loading()
        
    def _setup_main_widget(self) -> None:
        """Setup the main stacked widget for screen navigation."""
        # Set minimum size constraints
        self._widget_stack.setMinimumWidth(1400)
        self._widget_stack.setMinimumHeight(662)
        
        # Set window title
        self._widget_stack.setWindowTitle("QuantUS - Ultrasound Analysis")
        
        # Center the window on screen and set reasonable initial size
        self._center_window_on_screen()
        
    def _center_window_on_screen(self) -> None:
        """Center the window on the primary screen with appropriate size."""
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QRect
        
        # Get the primary screen geometry (PyQt6 compatible)
        app = QApplication.instance()
        if app is None:
            # Fallback to default size if no application instance
            self._widget_stack.setGeometry(100, 100, 1600, 900)
            return
            
        primary_screen = app.primaryScreen()
        if primary_screen is None:
            # Fallback to default size if no primary screen
            self._widget_stack.setGeometry(100, 100, 1600, 900)
            return
            
        screen_geometry = primary_screen.geometry()
        
        # Calculate window size (60% of screen size, but respect minimums)
        window_width = max(1400, int(screen_geometry.width() * 0.6))
        window_height = max(662, int(screen_geometry.height() * 0.6))
        
        # Ensure window doesn't exceed screen bounds
        window_width = min(window_width, screen_geometry.width() - 100)
        window_height = min(window_height, screen_geometry.height() - 100)
        
        # Calculate center position
        x = (screen_geometry.width() - window_width) // 2
        y = (screen_geometry.height() - window_height) // 2
        
        # Set the geometry
        self._widget_stack.setGeometry(x, y, window_width, window_height)
        
    def eventFilter(self, obj, event):
        """
        Event filter to handle window resize events and ensure proper geometry.
        
        Args:
            obj: The object that generated the event
            event: The event that occurred
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        if obj == self._widget_stack and event.type() == QEvent.Type.Resize:
            # Ensure minimum size constraints are maintained
            current_size = self._widget_stack.size()
            if current_size.width() < 1400 or current_size.height() < 662:
                # Resize to minimum if needed
                new_width = max(1400, current_size.width())
                new_height = max(662, current_size.height())
                self._widget_stack.resize(new_width, new_height)
                return True
        return super().eventFilter(obj, event)
        
    def _connect_model_signals(self) -> None:
        """Connect unified model signals to application controller."""
        self._model.image_loaded.connect(self._initialize_segmentation_loading)
        self._model.error_occurred.connect(self._on_model_error)
        
    def _initialize_image_loading(self) -> None:
        """Initialize the image loading screen."""
        if self._image_loading_controller:
            self._cleanup_image_loading()
            
        # Create controller with unified model
        self._image_loading_controller = ImageLoadingController(self._model)
        
        # Connect to handle image loading completion
        self._image_loading_controller.view.user_action.connect(self._on_image_action)
        
        # Add the coordinator widget to the main stack
        self._widget_stack.addWidget(self._image_loading_controller.view)
        self._widget_stack.setCurrentWidget(self._image_loading_controller.view)
        
    def _initialize_segmentation_loading(self, image_data: UltrasoundRfImage) -> None:
        """
        Initialize the segmentation loading screen.
        
        Args:
            image_data: Loaded image data from previous screen
        """
        if self._segmentation_controller:
            self._cleanup_segmentation_loading()
        
        # Create controller with the unified model (automatically creates modular coordinator)
        self._segmentation_controller = SegmentationLoadingController(self._model)
        
        # Connect to handle segmentation actions
        self._segmentation_controller.view.user_action.connect(self._on_segmentation_action)
        self._segmentation_controller.view.back_requested.connect(self._navigate_to_image_loading)
        
        # Add to stack and show
        self._widget_stack.addWidget(self._segmentation_controller.view)
        self._widget_stack.setCurrentWidget(self._segmentation_controller.view)

    def _initialize_config_loading(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg) -> None:
        """
        Initialize the config loading screen.
        
        Args:
            image_data: Loaded image data from previous screen
            seg_data: Loaded segmentation data from previous screen
        """
        if self._config_loading_controller:
            self._cleanup_config_loading()
        
        # Create controller with the unified model (automatically creates modular coordinator)
        self._config_loading_controller = ConfigLoadingController(self._model, image_data, seg_data)
        
        # Connect to handle config actions
        self._config_loading_controller.user_action.connect(self._on_config_action)
        self._config_loading_controller.back_requested.connect(self._navigate_to_segmentation_loading)
        
        # Add to stack and show
        self._widget_stack.addWidget(self._config_loading_controller.get_widget())
        self._widget_stack.setCurrentWidget(self._config_loading_controller.get_widget())

    def _initialize_analysis_loading(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, config_data: RfAnalysisConfig) -> None:
        """
        Initialize the analysis loading screen.
        
        Args:
            image_data: Loaded image data from previous screen
            seg_data: Loaded segmentation data from previous screen
            config_data: Loaded config data from previous screen
        """
        if self._analysis_loading_controller:
            self._cleanup_analysis_loading()
        
        # Create controller with the unified model
        self._analysis_loading_controller = AnalysisLoadingController(self._model, image_data, seg_data, config_data)
        
        # Connect to handle analysis actions
        self._analysis_loading_controller.user_action.connect(self._on_analysis_action)
        self._analysis_loading_controller.back_requested.connect(self._navigate_to_config_loading)
        
        # Add to stack and show
        self._widget_stack.addWidget(self._analysis_loading_controller.get_widget())
        self._widget_stack.setCurrentWidget(self._analysis_loading_controller.get_widget())

    def _on_model_error(self, error_message: str) -> None:
        """
        Handle errors from unified model.
        
        Args:
            error_message: Error message from model
        """
        print(f"DEBUG: Application model error: {error_message}")
        # The individual view controllers will handle displaying the error to the user
        
    def _on_image_action(self, action_name: str, action_data) -> None:
        """
        Handle actions from the image loading screen.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        if action_name == 'image_loaded':
            self._image_data = action_data
            self._initialize_segmentation_loading(self._image_data)
            
    def _on_segmentation_action(self, action_name: str, action_data) -> None:
        """
        Handle actions from the segmentation loading screen.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        if action_name == 'segmentation_confirmed':
            self._seg_data = self._segmentation_controller.get_loaded_segmentation()
            model_image_data = self._model.image_data
            
            # Use the model data instead of potentially stale controller data
            image_data_to_use = model_image_data if model_image_data is not None else self._image_data

            self._initialize_config_loading(image_data_to_use, self._seg_data)
            
    def _on_config_action(self, action_name: str, action_data) -> None:
        """
        Handle actions from the config loading screen.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        if action_name == 'config_loading_completed':
            self._config_data = action_data
            
            # Prevent multiple initializations if analysis loading is already active
            if hasattr(self, '_analysis_loading_controller') and self._analysis_loading_controller is not None:
                return
            
            # Fix: Get the actual image and segmentation data from the model
            model_image_data = self._model.image_data
            model_seg_data = self._model.seg_data
            
            # Use model data if controller data is missing
            image_data = model_image_data if model_image_data else self._image_data
            seg_data = model_seg_data if model_seg_data else self._seg_data
            
            self._initialize_analysis_loading(image_data, seg_data, self._config_data)
            
    def _on_analysis_action(self, action_name: str, action_data) -> None:
        """
        Handle actions from the analysis loading screen.
        
        Args:
            action_name: Name of the action
            action_data: Data associated with the action
        """
        
        if action_name == 'analysis_loading_completed':
            # Prevent duplicate processing
            if hasattr(self, '_visualization_loading_controller') and self._visualization_loading_controller is not None:
                return
                
            # Analysis has been completed successfully; navigate to visualization loader
            self._initialize_visualization_loading(action_data)

    def _initialize_visualization_loading(self, analysis_data: ParamapAnalysis) -> None:
        """
        Initialize the visualization loading screen after analysis completes.
        
        Args:
            analysis_data: Completed analysis results passed from analysis stage
        """
        try:
            # Clean up any existing visualization controller
            if self._visualization_loading_controller:
                self._cleanup_visualization_loading()
            
            # Create visualization controller using the same unified model
            self._visualization_loading_controller = VisualizationLoadingController(self._model, analysis_data)
            
            # Get the widget and check if it's valid
            widget = self._visualization_loading_controller.get_widget()
            if widget is None:
                print("ERROR: Failed to create visualization loading widget")
                return
            
            # Connect to handle visualization actions
            self._visualization_loading_controller.user_action.connect(self._on_visualization_action)
            self._visualization_loading_controller.back_requested.connect(self._navigate_to_analysis_loading)
            
            # Add to stack and show
            self._widget_stack.addWidget(widget)
            self._widget_stack.setCurrentWidget(widget)
            
        except Exception as e:
            print(f"ERROR: Failed to initialize visualization loading: {e}")
            # Clean up on error
            if self._visualization_loading_controller:
                self._visualization_loading_controller = None

    def _on_visualization_action(self, action_name: str, action_data) -> None:
        """
        Handle actions from the visualization loading screen.
        """
        if action_name == 'visualization_loading_completed':
            print(f"Visualization completed successfully! Data: {type(action_data)}")
            # Stay on visualization; navigation happens only on explicit continue
        elif action_name == 'visualization_loading_continue':
            # User explicitly chose to continue to export
            self._initialize_export_loading()

    def _initialize_export_loading(self) -> None:
        """Initialize export loading screen after visualization completes."""
        # Clean existing
        if self._export_loading_controller:
            try:
                widget = self._export_loading_controller.get_widget()
                if widget is not None:
                    self._widget_stack.removeWidget(widget)
                    self._export_loading_controller.cleanup()
                    widget.deleteLater()
            except Exception:
                pass
            self._export_loading_controller = None

        # Create
        self._export_loading_controller = ExportLoadingController(self._model, self._visualization_loading_controller)
        widget = self._export_loading_controller.get_widget()
        self._export_loading_controller.user_action.connect(self._on_export_action)
        self._export_loading_controller.back_requested.connect(self._navigate_to_visualization_loading)
        self._widget_stack.addWidget(widget)
        self._widget_stack.setCurrentWidget(widget)

    def _on_export_action(self, action_name: str, action_data) -> None:
        if action_name == 'export_loading_completed':
            print(f"Export completed → {action_data.get('export_path')}")
                
    def _navigate_to_image_loading(self) -> None:
        """Navigate to image loading screen."""
        # Reset image loading controller to initial state
        if self._image_loading_controller:
            self._image_loading_controller.reset_view()
            
        # Clean up segmentation controller
        if self._segmentation_controller:
            self._cleanup_segmentation_loading()
            
        # Show image loading screen
        if self._image_loading_controller:
            self._widget_stack.setCurrentWidget(self._image_loading_controller.view)
        else:
            self._initialize_image_loading()
            
        # Reset current data
        self._image_data = None
        self._seg_data = None
        self._config_data = None
        
    def _navigate_to_segmentation_loading(self) -> None:
        """Navigate to segmentation loading screen."""
        # Clean up config controller
        if self._config_loading_controller:
            self._cleanup_config_loading()
            
        # Show segmentation loading screen
        if self._segmentation_controller:
            self._widget_stack.setCurrentWidget(self._segmentation_controller.view)
        else:
            self._initialize_segmentation_loading(self._image_data)
            
        # Reset current data
        self._config_data = None
    
    def _navigate_to_config_loading(self) -> None:
        """Navigate to config loading screen."""
        # Clean up analysis controller
        if self._analysis_loading_controller:
            self._cleanup_analysis_loading()
            
        # Show config loading screen
        if self._config_loading_controller:
            self._widget_stack.setCurrentWidget(self._config_loading_controller.get_widget())
        else:
            # Fix: Use model data instead of potentially stale controller data
            model_image_data = self._model.image_data
            model_seg_data = self._model.seg_data
            image_data_to_use = model_image_data if model_image_data is not None else self._image_data
            seg_data_to_use = model_seg_data if model_seg_data is not None else self._seg_data
            
            print(f"DEBUG: _navigate_to_config_loading using image_data = {image_data_to_use is not None}")
            self._initialize_config_loading(image_data_to_use, seg_data_to_use)
        
    def _cleanup_image_loading(self) -> None:
        """Clean up image loading controller resources."""
        try:
            if self._image_loading_controller:
                widget = self._image_loading_controller.get_widget()
                if widget is not None:
                    self._widget_stack.removeWidget(widget)
                    self._image_loading_controller.cleanup()
                    widget.deleteLater()
                else:
                    self._image_loading_controller.cleanup()
                self._image_loading_controller = None
        except Exception as e:
            print(f"Warning: Error during image loading cleanup: {e}")
            self._image_loading_controller = None

    def _cleanup_segmentation_loading(self) -> None:
        """Clean up segmentation loading controller resources."""
        try:
            if self._segmentation_controller:
                widget = self._segmentation_controller.get_widget()
                if widget is not None:
                    self._widget_stack.removeWidget(widget)
                    self._segmentation_controller.cleanup()
                    widget.deleteLater()
                else:
                    self._segmentation_controller.cleanup()
                self._segmentation_controller = None
        except Exception as e:
            print(f"Warning: Error during segmentation loading cleanup: {e}")
            self._segmentation_controller = None

    def _cleanup_config_loading(self) -> None:
        """Clean up config loading controller resources."""
        try:
            if self._config_loading_controller:
                widget = self._config_loading_controller.get_widget()
                if widget is not None:
                    self._widget_stack.removeWidget(widget)
                    self._config_loading_controller.cleanup()
                    widget.deleteLater()
                else:
                    self._config_loading_controller.cleanup()
                self._config_loading_controller = None
        except Exception as e:
            print(f"Warning: Error during config loading cleanup: {e}")
            self._config_loading_controller = None

    def _cleanup_analysis_loading(self) -> None:
        """Clean up analysis loading controller resources."""
        try:
            if self._analysis_loading_controller:
                widget = self._analysis_loading_controller.get_widget()
                if widget is not None:
                    self._widget_stack.removeWidget(widget)
                    self._analysis_loading_controller.cleanup()
                    widget.deleteLater()
                else:
                    self._analysis_loading_controller.cleanup()
                self._analysis_loading_controller = None
        except Exception as e:
            print(f"Warning: Error during analysis loading cleanup: {e}")
            self._analysis_loading_controller = None
    
    def _cleanup_visualization_loading(self) -> None:
        """Clean up visualization loading controller resources."""
        try:
            if self._visualization_loading_controller:
                # Get the widget first and check if it's valid
                widget = self._visualization_loading_controller.get_widget()
                if widget is not None:
                    # Remove from widget stack
                    self._widget_stack.removeWidget(widget)
                    # Clean up the controller
                    self._visualization_loading_controller.cleanup()
                    # Delete the widget
                    widget.deleteLater()
                else:
                    # Widget is None, just clean up the controller
                    self._visualization_loading_controller.cleanup()
                
                # Clear the reference
                self._visualization_loading_controller = None
        except Exception as e:
            print(f"Warning: Error during visualization loading cleanup: {e}")
            # Ensure the reference is cleared even if cleanup fails
            self._visualization_loading_controller = None

    def _navigate_to_analysis_loading(self) -> None:
        """Navigate back to the analysis loading screen from visualization."""
        try:
            # Clean up visualization controller
            if self._visualization_loading_controller:
                self._cleanup_visualization_loading()
            
            # Always reinitialize analysis loading to ensure clean state
            # Use the most up-to-date data from the unified model, with
            # fallbacks to the controller's stored references.
            model_image_data = getattr(self._model, 'image_data', None)
            model_seg_data = getattr(self._model, 'seg_data', None)
            model_config_data = getattr(self._model, 'config_data', None)

            image_data_to_use = model_image_data if model_image_data is not None else self._image_data
            seg_data_to_use = model_seg_data if model_seg_data is not None else self._seg_data
            config_data_to_use = model_config_data if model_config_data is not None else self._config_data

            self._initialize_analysis_loading(image_data_to_use, seg_data_to_use, config_data_to_use)
            
        except Exception as e:
            print(f"ERROR: Failed to navigate to analysis loading: {e}")
            # Try to reinitialize as fallback
            try:
                print("DEBUG: Attempting fallback reinitialization...")
                self._initialize_analysis_loading()
                print("DEBUG: Fallback reinitialization successful")
            except Exception as fallback_error:
                print(f"ERROR: Failed to reinitialize analysis loading: {fallback_error}")

    def _navigate_to_visualization_loading(self) -> None:
        """Navigate back to visualization loading screen from export."""
        if self._export_loading_controller:
            try:
                widget = self._export_loading_controller.get_widget()
                if widget is not None:
                    self._widget_stack.removeWidget(widget)
                    self._export_loading_controller.cleanup()
                    widget.deleteLater()
            except Exception:
                pass
            self._export_loading_controller = None
        if self._visualization_loading_controller:
            self._widget_stack.setCurrentWidget(self._visualization_loading_controller.get_widget())

    def show(self) -> None:
        """Show the main application window."""
        # Ensure window is properly sized before showing
        self._widget_stack.show()
        
        # Apply window state to ensure proper display
        self._widget_stack.raise_()
        self._widget_stack.activateWindow()
        
    def run(self) -> None:
        """
        Run the main application event loop.
        
        This replaces the original complex event loop logic with clean MVC navigation.
        """
        self.show()
        
        # Run Qt event loop
        try:
            sys.exit(self._app.exec())
        except SystemExit:
            # Clean shutdown
            self._cleanup()
            
    def _cleanup(self) -> None:
        """Clean up all resources before application exit."""
        # Remove event filter
        if self._widget_stack:
            self._widget_stack.removeEventFilter(self)
            
        self._cleanup_image_loading()
        self._cleanup_segmentation_loading()
        self._cleanup_config_loading()
        self._cleanup_analysis_loading()
        
    @property
    def image_data(self) -> Optional[UltrasoundRfImage]:
        """Get the currently loaded image data."""
        return self._image_data
        
    @property
    def seg_data(self) -> Optional[BmodeSeg]:
        """Get the currently loaded segmentation data."""
        return self._seg_data
        
    @property
    def config_data(self) -> Optional[RfAnalysisConfig]:
        """Get the loaded config data."""
        return self._config_data


def create_application():
    """
    Create and configure the QuantUS application with unified MVC architecture.
    
    Returns:
        ApplicationController: Configured application controller
    """
    import sys
    from PyQt6.QtWidgets import QApplication
    
    # Create QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    qdarktheme.setup_theme()
        
    # Set application properties for better window management
    app.setApplicationName("QuantUS")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("QuantUS")
    
    # Create and configure application controller
    app_controller = ApplicationController(app)
    
    return app_controller


def run_application():
    """
    Run the QuantUS application.
    
    Returns:
        int: Exit code
    """
    try:
        app_controller = create_application()
        app_controller.run()
        return 0
    except Exception as e:
        print(f"Error running QuantUS application: {e}")
        import traceback
        traceback.print_exc()
        return 1
