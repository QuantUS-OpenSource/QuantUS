"""
Main Application Controller for QuantUS GUI MVC architecture
"""

import sys
import qdarktheme
from typing import Optional
from PyQt6.QtWidgets import QApplication, QStackedWidget
from PyQt6.QtCore import QObject, pyqtSignal

from .application_model import ApplicationModel
from .image_loading.image_loading_view_coordinator import ImageLoadingViewCoordinator
from .image_loading.image_loading_controller import ImageLoadingController
from .seg_loading.seg_loading_controller import SegmentationLoadingController
from engines.ceus.src.data_objs import UltrasoundImage, CeusSeg


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
        
        # Setup main widget
        self._setup_main_widget()
        
        # Connect model signals
        self._connect_model_signals()
        
        # Initialize first screen
        self._initialize_image_loading()
        
    def _setup_main_widget(self) -> None:
        """Setup the main stacked widget for screen navigation."""
        self._widget_stack.setMinimumWidth(1400)
        self._widget_stack.setWindowTitle("QuantUS - Ultrasound Analysis")
        
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

    def _initialize_segmentation_loading(self, image_data: UltrasoundImage) -> None:
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
            # TODO: Navigate to analysis screen when implemented
            print("Analysis screen coming soon...")
            self._app.quit()
                
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
        
    def _cleanup_image_loading(self) -> None:
        """Clean up image loading controller resources."""
        if self._image_loading_controller:
            # Remove from widget stack and clean up
            self._widget_stack.removeWidget(self._image_loading_controller.view)
            self._image_loading_controller.cleanup()
            self._image_loading_controller.view.deleteLater()
            self._image_loading_controller = None
            
    def _cleanup_segmentation_loading(self) -> None:
        """Clean up segmentation loading controller resources."""
        if self._segmentation_controller:
            self._widget_stack.removeWidget(self._segmentation_controller.view)
            self._segmentation_controller.cleanup()
            self._segmentation_controller.view.deleteLater()
            self._segmentation_controller = None
            
    def show(self) -> None:
        """Show the main application window."""
        self._widget_stack.show()
        
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
        self._cleanup_image_loading()
        self._cleanup_segmentation_loading()
        
    @property
    def image_data(self) -> Optional[UltrasoundImage]:
        """Get the currently loaded image data."""
        return self._image_data
        
    @property
    def seg_data(self) -> Optional[CeusSeg]:
        """Get the currently loaded segmentation data."""
        return self._seg_data


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
