"""
Segmentation Loading Controller for MVC architecture
"""

from typing import Any, Optional

from ..mvc.base_controller import BaseController
from ..application_model import ApplicationModel
from .seg_loading_view_coordinator import SegLoadingViewCoordinator
from engines.ceus.src.data_objs import UltrasoundImage, CeusSeg


class SegmentationLoadingController(BaseController):
    """
    Controller for segmentation loading functionality.
    
    Coordinates between ApplicationModel and SegLoadingViewCoordinator,
    handling user interactions and data flow through multiple widgets for
    segmentation type selection, file loading, ROI drawing, and preview.
    """
    
    def __init__(self, model: Optional[ApplicationModel] = None, custom_view=None):
        if model is None:
            raise ValueError("ApplicationModel must be provided to SegmentationLoadingController")
        
        # Use custom view if provided, otherwise create coordinator with image data
        if custom_view:
            view = custom_view
        else:
            # Get current image from the unified model
            image_data = model.image_data
            if not image_data:
                raise ValueError("No image loaded in ApplicationModel")
            view = SegLoadingViewCoordinator(image_data)
            
        super().__init__(model, view)
        
        # # Connect to model signals for automatic view updates
        # self._connect_model_signals()
        
        # Initialize view with segmentation loaders
        self._initialize_view()
        
    # def _connect_model_signals(self) -> None:
    #     """Connect to model signals for automatic view updates."""
    #     self.model.segmentation_loaded.connect(self.view.show_segmentation_preview)
        
    def _initialize_view(self) -> None:
        """Initialize the view with data from the model."""
        seg_loader_names = self.model.seg_loader_names
        self.view.set_seg_loaders(seg_loader_names)
        
    def handle_user_action(self, action_name: str, action_data: Any) -> None:
        """
        Handle user actions from the view.
        
        Args:
            action_name: Name of the action performed
            action_data: Data associated with the action
        """
        if action_name == 'seg_type_selected':
            self._handle_seg_type_selection(action_data)
        elif action_name == 'frame_selected':
            self.view.show_roi_drawing(action_data)
        elif action_name == 'load_segmentation':
            self._handle_segmentation_loading(action_data)
        elif action_name == 'segmentation_confirmed':
            pass # Handle confirmation action in the application controller
        else:
            raise ValueError(f"Unknown action: {action_name}")
            
    def _handle_seg_type_selection(self, seg_type_name: str) -> None:
        """
        Handle segmentation type selection.
        
        Args:
            seg_type_name: Display name of selected segmentation type
        """
        success = self.model.set_seg_type(seg_type_name)
        if success:
            if seg_type_name == "Manual Segmentation":
                # Show ROI drawing interface for manual segmentation
                image_data = self.model.image_data
                if image_data.spatial_dims == 3:
                    self.view.show_voi_drawing()
                else:
                    self.view.show_roi_drawing()
                #         raise ValueError("Unsupported RF data dimensions for manual segmentation")
                # elif image_data.spatial_dims == 3:
                #     raise NotImplementedError("Manual segmentation for 3D data not implemented")
                # else:
                #     raise ValueError("Unsupported spatial dimensions for manual segmentation")
            else:
                # Update view with file extensions for this segmentation type
                file_extensions = self.model.get_seg_file_extensions()
                self.view.show_file_selection(file_extensions)
            
    def _handle_segmentation_loading(self, load_data: dict) -> None:
        """
        Handle segmentation loading request.
        
        Args:
            load_data: Dictionary with loading parameters
        """
        seg_path = load_data.get('seg_path', '')
        seg_loader_kwargs = load_data.get('seg_loader_kwargs', {})
        
        self.model.load_segmentation(seg_path, seg_loader_kwargs)
        
    def get_loaded_segmentation(self) -> CeusSeg:
        """
        Get the currently loaded segmentation data.
        
        Returns:
            CeusSeg: The loaded segmentation data, or None if no segmentation loaded
        """
        return self.model.seg_data
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model.cleanup()
