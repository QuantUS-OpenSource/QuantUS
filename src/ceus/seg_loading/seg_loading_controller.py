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
            view = SegLoadingViewCoordinator(image_data, bmode_image_data=model.bmode_image_data)

        super().__init__(model, view)

        # Connect to model signals
        model.segmentation_loaded.connect(self._on_segmentation_loaded)
        model.motion_comp_started.connect(self._on_mc_started)
        model.motion_comp_completed.connect(self._on_mc_completed)

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
        elif action_name == 'apply_preprocs_preview':
            self._handle_preprocs_preview(action_data)
        elif action_name == 'run_mc_from_mask':
            self.model.run_mc_from_mask(
                action_data['voi_mask'],
                action_data['reference_frame'],
                action_data['search_margin_ratio'],
                action_data.get('padding', 5),
            )
        elif action_name == 'rerun_motion_compensation':
            self.model.apply_motion_compensation(action_data)
        elif action_name == 'segmentation_confirmed':
            pass  # Handle confirmation action in the application controller
        else:
            raise ValueError(f"Unknown action: {action_name}")
        
    def _handle_preprocs_preview(self, preproc_data_list: list) -> None:
        """
        Handle multiple preprocessing functions update request.
        
        Args:
            preproc_data_list: List of dictionaries with 'name' and 'kwargs' keys
        """
        image_data = preproc_data_list[0]['image_data']
        frame_ix = preproc_data_list[0]['frame_ix']

        for preproc_data in preproc_data_list:
            preproc_data.pop('image_data', None)
            preproc_data.pop('frame_ix', None)

        image_data = self.model.apply_preprocessing_preview(preproc_data_list, image_data)

        self.view.preview_modified_image(image_data, frame_ix)
        
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
        mc_kwargs = load_data.get('mc_kwargs', {})
        seg_type = load_data.get('seg_type')

        if seg_type:
            # seg_type may be a raw loader key (e.g. 'nifti') rather than a display name.
            # Set it directly on the model if it's a known loader key.
            if seg_type in self.model._seg_loaders:
                self.model._selected_seg_type = seg_type
            else:
                self.model.set_seg_type(seg_type)

        self.model.load_segmentation(seg_path, seg_loader_kwargs, mc_kwargs=mc_kwargs or None)

    def _on_segmentation_loaded(self, seg_data: CeusSeg) -> None:
        """
        Handle a freshly loaded segmentation.

        For 3D images loaded from a file (e.g. NIfTI), show the VOI drawing widget
        so the user can review the mask and optionally apply motion compensation.
        If the file already contains motion compensation data, jump straight into
        MC review mode so the user can review the motion-compensated mask.
        For 2D images or when the widget is already shown (VOI drawn manually),
        proceed directly to confirmation.
        """
        image_data = self.model.image_data
        if image_data and getattr(image_data, 'spatial_dims', 2) == 3 \
                and self.view._voi_drawing_widget is None:
            self.view.show_voi_drawing_with_seg(seg_data)
            # If the loaded file already contains motion compensation data,
            # immediately enter MC review mode so the user can inspect the
            # motion-compensated mask instead of just the static reference VOI.
            if getattr(seg_data, 'use_mc', False) and getattr(seg_data, 'motion_compensation', None) is not None:
                self.view.show_motion_comp_review(
                    seg_data, self.model.image_data, self.model.bmode_image_data
                )
        else:
            self.view._emit_user_action('segmentation_confirmed', seg_data)

    def _on_mc_started(self) -> None:
        """Show loading state while motion compensation runs."""
        self.view.show_loading()

    def _on_mc_completed(self, seg_data: CeusSeg) -> None:
        """Enter or update MC review mode in the VOI drawing widget."""
        self.view.hide_loading()
        self.view.show_motion_comp_review(
            seg_data, self.model.image_data, self.model.bmode_image_data
        )
        
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
