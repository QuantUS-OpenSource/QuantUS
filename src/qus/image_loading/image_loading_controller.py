"""
Image Loading Controller for MVC architecture
"""

from typing import Any, Optional

from src.qus.mvc.base_controller import BaseController
from src.qus.application_model import ApplicationModel
from .image_loading_view_coordinator import ImageLoadingViewCoordinator


class ImageLoadingController(BaseController):
    """
    Controller for image loading functionality.
    
    Coordinates between ApplicationModel and ImageLoadingViewCoordinator,
    handling user interactions and data flow through multiple dialogs.
    """
    
    def __init__(self, model: Optional[ApplicationModel] = None):
        if model is None:
            model = ApplicationModel()
        view = ImageLoadingViewCoordinator()
        super().__init__(model, view)
        
        # Initialize view with scan loaders
        self._initialize_view()
        
    def _initialize_view(self) -> None:
        """Initialize the view with data from the model."""
        scan_loader_names = self.model.scan_loader_names
        self.view.set_scan_loaders(scan_loader_names)
        
    def handle_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Handle user actions from the view.
        
        Args:
            action_name: Name of the action performed
            action_data: Data associated with the action
        """
        if action_name == 'scan_type_selected':
            self._handle_scan_type_selection(action_data)
        elif action_name == 'load_image':
            self._handle_image_loading(action_data)
            
    def _handle_scan_type_selection(self, scan_type_name: str) -> None:
        """
        Handle scan type selection.
        
        Args:
            scan_type_name: Display name of selected scan type
        """
        success = self.model.set_scan_type(scan_type_name)
        if success:
            # Show file selection dialog
            file_extensions = self.model.get_file_extensions()
            loading_options, default_vals = self.model.get_image_loading_options()
            self.view.show_file_selection(file_extensions, loading_options, default_vals)

    def _handle_image_loading(self, load_data: dict) -> None:
        """
        Handle image loading request.
        
        Args:
            load_data: Dictionary with loading parameters
        """
        try:
            self.model.load_image(
                image_path=load_data['image_path'],
                phantom_path=load_data['phantom_path'],
                scan_loader_kwargs=load_data['scan_loader_kwargs']
            )
            
        except Exception as e:
            print(f"DEBUG: Error in image loading: {e}")
            import traceback
            traceback.print_exc()
            self.view.show_error(f"Failed to start image loading: {e}")

    def reset_view(self) -> None:
        """Reset the view to initial state."""
        self.view.clear_error()
        self.view.reset_to_scan_type_selection()
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model.cleanup()
    
    def get_widget(self):
        """Get the view coordinator widget."""
        return self.view
