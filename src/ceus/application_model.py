"""
Unified Application Model for QuantUS GUI MVC architecture

This model centralizes all data management and business logic for the entire application,
replacing the individual models for each component.
"""

import os
from typing import Dict, Any, Optional, List
from PyQt6.QtCore import QThread, pyqtSignal

from .mvc.base_model import BaseModel
from engines.ceus.src.image_loading.options import get_scan_loaders
from engines.ceus.src.seg_loading.options import get_seg_loaders
from engines.ceus.src.time_series_analysis.options import get_analysis_types
from engines.ceus.src.entrypoints import scan_loading_step, seg_loading_step
from engines.ceus.src.data_objs.image import UltrasoundImage
from engines.ceus.src.data_objs.seg import CeusSeg
from engines.ceus.src.time_series_analysis.curves.framework import CurvesAnalysis


class ScanLoadingWorker(QThread):
    """Worker thread for time-consuming scan loading operations."""
    finished = pyqtSignal(UltrasoundImage)
    error_msg = pyqtSignal(str)

    def __init__(self, scan_type: str, image_path: str, scan_loader_kwargs: Dict[str, Any]):
        super().__init__()
        self.scan_type = scan_type
        self.image_path = image_path
        self.scan_loader_kwargs = scan_loader_kwargs

    def run(self):
        """Execute the scan loading in background thread."""
        try:
            image_data = scan_loading_step(
                self.scan_type, 
                self.image_path,  
                **self.scan_loader_kwargs
            )
            self.finished.emit(image_data)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_msg.emit(f"Error loading image: {e}")


class SegLoadingWorker(QThread):
    """Worker thread for time-consuming segmentation loading operations."""
    finished = pyqtSignal(CeusSeg)
    error_msg = pyqtSignal(str)

    def __init__(self, seg_type: str, seg_path: str, image_data: UltrasoundImage, seg_loader_kwargs: Dict[str, Any]):
        super().__init__()
        self.seg_type = seg_type
        self.seg_path = seg_path
        self.image_data = image_data
        self.seg_loader_kwargs = seg_loader_kwargs

    def run(self):
        """Execute the segmentation loading in background thread."""
        try:
            seg_data = seg_loading_step(
                self.seg_type,
                self.image_data,
                self.seg_path,
                self.image_data.scan_path,
                **self.seg_loader_kwargs
            )
            
            self.finished.emit(seg_data)
            
        except Exception as e:
            print(f"DEBUG: Seg worker thread error: {e}")
            import traceback
            traceback.print_exc()
            self.error_msg.emit(f"Error loading segmentation: {e}")


class ApplicationModel(BaseModel):
    """
    Unified application model that manages all data and business logic for the QuantUS GUI.
    
    This centralizes:
    - Image loading and scan type management
    - Segmentation loading and processing
    - ROI/VOI creation and management
    - Application state and workflow coordination
    """
    
    # Additional signals for application-specific events
    image_loaded = pyqtSignal(UltrasoundImage)
    bmode_image_loaded = pyqtSignal(UltrasoundImage)
    segmentation_loaded = pyqtSignal(CeusSeg)

    def __init__(self):
        super().__init__()
        
        # Image loading state
        self._scan_loaders: Dict[str, Any] = {}
        self._selected_scan_type: Optional[str] = None
        self._image_data: Optional[UltrasoundImage] = None
        self._bmode_image_data: Optional[UltrasoundImage] = None
        self._scan_worker: Optional[ScanLoadingWorker] = None
        self._bmode_scan_worker: Optional[ScanLoadingWorker] = None
        self._pending_bmode_load: bool = False  # True when B-mode load is in progress
        
        # Segmentation loading state
        self._seg_loaders: Dict[str, Any] = {}
        self._selected_seg_type: Optional[str] = None
        self._seg_data: Optional[CeusSeg] = None
        self._seg_worker: Optional[SegLoadingWorker] = None
        
        # Initialize loaders
        self._load_scan_loaders()
        self._load_seg_loaders()
        self._load_analysis_types()
    
    def _load_scan_loaders(self) -> None:
        """Load available scan loaders from backend."""
        try:
            self._scan_loaders = get_scan_loaders()
        except Exception as e:
            self._emit_error(f"Failed to load scan loaders: {e}")
    
    def _load_seg_loaders(self) -> None:
        """Load available segmentation loaders from backend."""
        try:
            self._seg_loaders = get_seg_loaders()
        except Exception as e:
            self._emit_error(f"Failed to load seg loaders: {e}")

    def _load_analysis_types(self) -> None:
        """Load available analysis types from backend."""
        try:
            self._analysis_types, self._analysis_functions = get_analysis_types()
        except Exception as e:
            print(f"Error loading analysis types: {e}")
            self._analysis_types = {}
            self._analysis_functions = {}
    
    # Image Loading Properties and Methods
    @property
    def scan_loaders(self) -> Dict[str, Any]:
        """Get available scan loaders."""
        return self._scan_loaders
    
    @property
    def scan_loader_names(self) -> list:
        """Get formatted scan loader names for display."""
        if not self._scan_loaders:
            return []
        
        names = [s.replace("_", " ").capitalize() for s in self._scan_loaders.keys()]
        return [s.replace("rf", "RF").replace("iq", "IQ") for s in names]
    
    @property
    def selected_scan_type(self) -> Optional[str]:
        """Get currently selected scan type."""
        return self._selected_scan_type
    
    @property
    def image_data(self) -> Optional[UltrasoundImage]:
        """Get the currently loaded image data."""
        return self._image_data

    @property
    def bmode_image_data(self) -> Optional[UltrasoundImage]:
        """Get the currently loaded B-mode image data."""
        return self._bmode_image_data

    def set_scan_type(self, scan_type_display_name: str) -> bool:
        """
        Set the selected scan type.
        
        Args:
            scan_type_display_name: Display name of the scan type
            
        Returns:
            bool: True if successfully set, False otherwise
        """
        try:
            # Convert display name back to internal key
            loader_names = list(self._scan_loaders.keys())
            display_names = self.scan_loader_names
            
            if scan_type_display_name in display_names:
                index = display_names.index(scan_type_display_name)
                self._selected_scan_type = loader_names[index]
                return True
            else:
                self._emit_error(f"Invalid scan type: {scan_type_display_name}")
                return False
        except Exception as e:
            self._emit_error(f"Error setting scan type: {e}")
            return False
    
    def get_file_extensions(self) -> list:
        """
        Get file extensions for the selected scan type.
        
        Returns:
            list: File extensions supported by selected scan loader
        """
        if not self._selected_scan_type or self._selected_scan_type not in self._scan_loaders:
            return []
        
        try:
            loader = self._scan_loaders[self._selected_scan_type]['cls']
            return getattr(loader, 'extensions', [])
        except Exception as e:
            self._emit_error(f"Error getting file extensions: {e}")
            return []
        
    def get_image_loading_options(self) -> list:
        """
        Get required keyword arguments for the selected scan type.
        
        Returns:
            list: List of required keyword arguments
        """
        if not self._selected_scan_type or self._selected_scan_type not in self._scan_loaders:
            return []
        
        try:
            loader = self._scan_loaders[self._selected_scan_type]['cls']
            return getattr(loader, 'required_kwargs', [])
        except Exception as e:
            self._emit_error(f"Error getting required kwargs: {e}")
            return []
    
    def load_image(self, image_path: str, scan_loader_kwargs: Dict[str, Any] = None) -> None:
        """
        Load scan image data.

        Args:
            image_path: Path to image file
            scan_loader_kwargs: Additional loader arguments (optional)
        """
        if not self._selected_scan_type:
            self._emit_error("No scan type selected")
            return

        # Reset state for new load cycle
        self._image_data = None
        self._bmode_image_data = None
        self._pending_bmode_load = False

        if scan_loader_kwargs is None:
            scan_loader_kwargs = {}
        
        input_data = {
            'scan_type': self._selected_scan_type,
            'image_path': image_path,
            'scan_loader_kwargs': scan_loader_kwargs
        }
        
        if not self._validate_image_input(input_data):
            return
        
        # Stop any existing worker
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.quit()
            self._scan_worker.wait()
        
        # Create and start worker
        self._scan_worker = ScanLoadingWorker(
            self._selected_scan_type,
            image_path,
            scan_loader_kwargs
        )
        
        # Connect worker signals
        self._scan_worker.finished.connect(self._on_image_loading_complete)
        self._scan_worker.error_msg.connect(self._emit_error)
        
        # Start loading
        self._set_loading(True)
        self._scan_worker.start()

    # =========================================================================
    # Preprocessing Interface
    # =========================================================================

    def get_preprocessing_options(self) -> Dict[str, Any]:
        """
        Get available preprocessing functions from the engine.
        
        Returns:
            Dict[str, Any]: Dictionary of function names and function objects
        """
        from engines.ceus.src.image_preprocessing.options import get_im_preproc_funcs
        return get_im_preproc_funcs()

    def get_preprocessing_kwargs_requirements(self, func_names: list) -> list:
        """
        Get required keyword arguments for a list of preprocessing functions.
        
        Args:
            func_names: List of preprocessing function names
            
        Returns:
            list: List of required keyword arguments
        """
        from engines.ceus.src.image_preprocessing.options import get_required_im_preproc_kwargs
        return get_required_im_preproc_kwargs(func_names)

    def apply_preprocessing_preview(self, func_configs: List[Dict[str, Any]], image_data: Optional[UltrasoundImage] = None) -> UltrasoundImage:
        """
        Apply preprocessing to the given UltrasoundImage.
        This does not modify the image data in the model.
        
        Args:
            func_configs: List of dicts with 'name' and 'kwargs' for each function
            image_data: Optional UltrasoundImage to preprocess (if None, uses current image)
        """
        if not image_data and not self._image_data:
            self._emit_error("No image loaded to preprocess")
            return
        
        processed_image = image_data if image_data else self._image_data
            
        try:
            funcs = self.get_preprocessing_options()
            
            for config in func_configs:
                name = config['name']
                kwargs = config.get('kwargs', {})
                if name in funcs:
                    processed_image = funcs[name](processed_image, **kwargs)
                else:
                    print(f"WARNING: Preprocessing function {name} not found")

        except Exception as e:
            self._emit_error(f"Error during preprocessing: {e}")

        return processed_image

    def enhance_image(self, image: UltrasoundImage, func_configs: List[Dict[str, Any]]) -> UltrasoundImage:
        """
        Enhance a given UltrasoundImage and return the result.
        Does not modify the model state. Used for preview/on-the-fly enhancement.
        
        Args:
            image: UltrasoundImage object to enhance
            func_configs: List of dicts with 'name' and 'kwargs' for each function
            
        Returns:
            UltrasoundImage: The enhanced image object
        """
        try:
            funcs = self.get_preprocessing_options()
            processed_image = image
            
            for config in func_configs:
                name = config['name']
                kwargs = config.get('kwargs', {})
                if name in funcs:
                    processed_image = funcs[name](processed_image, **kwargs)
            
            return processed_image
        except Exception as e:
            print(f"DEBUG: enhance_image error: {e}")
            return image

    def _validate_image_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for scan loading.
        
        Args:
            input_data: Dictionary containing scan loading parameters
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        required_fields = ['scan_type', 'image_path']
        
        # Check required fields
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                self._emit_error(f"Missing required field: {field}")
                return False
        
        # Validate scan type
        if input_data['scan_type'] not in self._scan_loaders:
            self._emit_error(f"Invalid scan type: {input_data['scan_type']}")
            return False
        
        # Validate file paths exist
        if not os.path.exists(input_data['image_path']):
            self._emit_error(f"Image file not found: {input_data['image_path']}")
            return False
                
        return True

    def _on_image_loading_complete(self, image_data: UltrasoundImage) -> None:
        """
        Handle completion of scan loading.

        Args:
            image_data: Loaded ultrasound image data
        """
        # Check if loading was successful
        if isinstance(image_data, UltrasoundImage):
            self._image_data = image_data

            # Print NIfTI information if applicable
            scan_path = getattr(image_data, 'scan_path', '')
            if scan_path and scan_path.lower().endswith(('.nii', '.nii.gz')):
                print(f"\n--- NIfTI Image Loaded (QuantUS GUI) ---")
                print(f"Path: {scan_path}")
                print(f"Shape: {getattr(image_data.pixel_data, 'shape', 'Unknown')}")
                print(f"Pixel Dimensions: {getattr(image_data, 'pixdim', 'Unknown')}")
                print(f"Frame Rate: {getattr(image_data, 'frame_rate', 'Unknown')}")
                print(f"----------------------------------------\n")

            # If B-mode is still loading, wait for it before emitting
            if self._pending_bmode_load:
                return
            self._set_loading(False)
            self.image_loaded.emit(image_data)
        else:
            self._set_loading(False)
            print(f"DEBUG: Image loading failed - invalid image data:")
            print(f"  - scan_path: {getattr(image_data, 'scan_path', 'Missing')}")
            print(f"  - has pixel_data: {hasattr(image_data, 'pixel_data')}")
            print(f"  - pixel_data is None: {getattr(image_data, 'pixel_data', None) is None}")
            print(f"  - has intensity: {hasattr(image_data, 'intensities_for_analysis')}")
            print(f"  - intensities_for_analysis is None: {getattr(image_data, 'intensities_for_analysis', None) is None}")
            self._emit_error("Failed to load image data - image loading was unsuccessful")

    def load_bmode_image(self, bmode_path: str, scan_loader_kwargs: Dict[str, Any] = None) -> None:
        """
        Load B-mode image data in the background.

        Args:
            bmode_path: Path to B-mode image file
            scan_loader_kwargs: Additional loader arguments (optional)
        """
        if not self._selected_scan_type:
            self._emit_error("No scan type selected for B-mode loading")
            return

        if scan_loader_kwargs is None:
            scan_loader_kwargs = {}

        if not os.path.exists(bmode_path):
            self._emit_error(f"B-mode file not found: {bmode_path}")
            return

        self._pending_bmode_load = True

        # Stop any existing B-mode worker
        if self._bmode_scan_worker and self._bmode_scan_worker.isRunning():
            self._bmode_scan_worker.quit()
            self._bmode_scan_worker.wait()

        self._bmode_scan_worker = ScanLoadingWorker(
            self._selected_scan_type,
            bmode_path,
            scan_loader_kwargs
        )

        self._bmode_scan_worker.finished.connect(self._on_bmode_loading_complete)
        self._bmode_scan_worker.error_msg.connect(self._on_bmode_loading_error)
        self._bmode_scan_worker.start()

    def _on_bmode_loading_complete(self, image_data: UltrasoundImage) -> None:
        """Handle completion of B-mode image loading."""
        self._pending_bmode_load = False
        if isinstance(image_data, UltrasoundImage):
            self._bmode_image_data = image_data
            self.bmode_image_loaded.emit(image_data)
        else:
            self._emit_error("Failed to load B-mode image data")

        # If CEUS already finished loading, emit image_loaded now
        if self._image_data is not None:
            self._set_loading(False)
            self.image_loaded.emit(self._image_data)

    def _on_bmode_loading_error(self, error_msg: str) -> None:
        """Handle B-mode loading error — proceed with CEUS only."""
        self._pending_bmode_load = False
        self._emit_error(error_msg)
        # Still allow proceeding with CEUS image if it loaded successfully
        if self._image_data is not None:
            self._set_loading(False)
            self.image_loaded.emit(self._image_data)

    # Segmentation Loading Properties and Methods
    @property
    def seg_loaders(self) -> Dict[str, Any]:
        """Get available segmentation loaders."""
        return self._seg_loaders
    
    @property
    def seg_loader_names(self) -> list:
        """Get formatted segmentation loader names for display."""
        if not self._seg_loaders:
            return []
        
        names = [s.replace("_", " ").capitalize() for s in self._seg_loaders.keys()]
        names.append("Manual Segmentation")
        return names
    
    @property
    def selected_seg_type(self) -> Optional[str]:
        """Get currently selected segmentation type."""
        return self._selected_seg_type
    
    @property
    def seg_data(self) -> Optional[CeusSeg]:
        """Get the currently loaded segmentation."""
        return self._seg_data
    
    def set_seg_type(self, seg_type_display_name: str) -> bool:
        """
        Set the selected segmentation type.
        
        Args:
            seg_type_display_name: Display name of the segmentation type
            
        Returns:
            bool: True if successfully set, False otherwise
        """
        try:
            if seg_type_display_name == "Manual Segmentation":
                self._selected_seg_type = "pkl_roi"
                return True

            # Convert display name back to internal key
            loader_names = list(self._seg_loaders.keys())
            display_names = self.seg_loader_names
            
            if seg_type_display_name in display_names:
                index = display_names.index(seg_type_display_name)
                self._selected_seg_type = loader_names[index]
                return True
            else:
                self._emit_error(f"Invalid segmentation type: {seg_type_display_name}")
                return False
        except Exception as e:
            self._emit_error(f"Error setting segmentation type: {e}")
            return False
    
    def get_seg_file_extensions(self) -> list:
        """
        Get file extensions for the selected segmentation type.
        
        Returns:
            list: File extensions supported by selected seg loader
        """
        if not self._selected_seg_type or self._selected_seg_type not in self._seg_loaders:
            return []
        
        try:
            loader = self._seg_loaders[self._selected_seg_type]
            return getattr(loader, 'supported_extensions', [])
        except Exception as e:
            self._emit_error(f"Error getting seg file extensions: {e}")
            return []
    
    def load_segmentation(self, seg_path: str, seg_loader_kwargs: Dict[str, Any] = None,
                          mc_kwargs: Dict[str, Any] = None) -> None:
        """
        Load segmentation data.

        Args:
            seg_path: Path to segmentation file
            seg_loader_kwargs: Additional loader arguments (optional)
            mc_kwargs: Motion compensation kwargs (optional). When provided and non-empty,
                       motion compensation is applied after loading.
        """
        self._pending_mc_kwargs = mc_kwargs if mc_kwargs else None
        if not self._image_data:
            self._emit_error("No image loaded - cannot load segmentation")
            return
        
        if not self._selected_seg_type:
            self._emit_error("No segmentation type selected")
            return
        
        if seg_loader_kwargs is None:
            seg_loader_kwargs = {}
        
        # Validate input
        if not os.path.exists(seg_path):
            self._emit_error(f"Segmentation file not found: {seg_path}")
            return
        
        # Stop any existing worker
        if self._seg_worker and self._seg_worker.isRunning():
            self._seg_worker.quit()
            self._seg_worker.wait()
        
        # Create and start worker
        self._seg_worker = SegLoadingWorker(
            self._selected_seg_type,
            seg_path,
            self._image_data,
            seg_loader_kwargs
        )
        
        # Connect worker signals
        self._seg_worker.finished.connect(self._on_segmentation_loading_complete)
        self._seg_worker.error_msg.connect(self._emit_error)
        
        # Start loading
        self._set_loading(True)
        self._seg_worker.start()
    
    def _on_segmentation_loading_complete(self, seg_data: CeusSeg) -> None:
        """
        Handle completion of segmentation loading.

        Args:
            seg_data: Loaded segmentation data
        """
        self._set_loading(False)

        if not (seg_data and hasattr(seg_data, 'seg_mask') and seg_data.seg_mask is not None):
            print("DEBUG: Segmentation loading failed - invalid seg data")
            self._emit_error("Failed to load segmentation data")
            return

        # Print NIfTI information if applicable
        seg_path = getattr(self._seg_worker, 'seg_path', '')
        if seg_path and seg_path.lower().endswith(('.nii', '.nii.gz')):
            print(f"\n--- NIfTI Segmentation Loaded (QuantUS GUI) ---")
            print(f"Path: {seg_path}")
            print(f"Shape: {getattr(seg_data.seg_mask, 'shape', 'Unknown')}")
            print(f"Pixel Dimensions: {getattr(seg_data, 'pixdim', 'Unknown')}")
            print(f"-----------------------------------------------\n")

        if self._pending_mc_kwargs and self._bmode_image_data is not None:
            mc_kwargs = self._pending_mc_kwargs
            self._pending_mc_kwargs = None
            self._mc_worker = MotionCompWorker(
                seg_data, self._image_data, self._bmode_image_data, mc_kwargs
            )
            self._mc_worker.finished.connect(self._on_mc_complete)
            self._mc_worker.error_msg.connect(self._emit_error)
            self.motion_comp_started.emit()
            self._mc_worker.start()
        else:
            self._pending_mc_kwargs = None
            self._seg_data = seg_data
            self.segmentation_loaded.emit(seg_data)

    def _on_mc_complete(self, seg_data: CeusSeg) -> None:
        """Handle motion compensation completion."""
        self._seg_data = seg_data
        self.motion_comp_completed.emit(seg_data)

    def run_mc_from_mask(self, voi_mask, reference_frame: int,
                         search_margin_ratio: float, padding: int = 5) -> None:
        """
        Run motion compensation directly from an in-memory VOI mask (no file I/O).

        Creates a CeusSeg from the mask, then runs MotionCompWorker.
        """
        if self._bmode_image_data is None:
            self._emit_error("B-mode data required for motion compensation")
            return

        import numpy as np
        seg_data = CeusSeg()
        seg_data.seg_mask = voi_mask.astype(np.uint8)
        seg_data.pixdim = list(self._image_data.pixdim[:3])
        seg_data.seg_name = "Manual VOI (MC)"

        mc_kwargs = {
            'reference_frame': reference_frame,
            'search_margin_ratio': search_margin_ratio,
            'padding': padding,
        }

        if self._mc_worker and self._mc_worker.isRunning():
            self._mc_worker.quit()
            self._mc_worker.wait()

        self._mc_worker = MotionCompWorker(
            seg_data, self._image_data, self._bmode_image_data, mc_kwargs
        )
        self._mc_worker.finished.connect(self._on_mc_complete)
        self._mc_worker.error_msg.connect(self._emit_error)
        self.motion_comp_started.emit()
        self._mc_worker.start()

    def apply_motion_compensation(self, mc_kwargs: Dict[str, Any]) -> None:
        """
        Re-run motion compensation on the current segmentation with new kwargs.

        Used by the review widget when the user changes the search margin or re-anchors.
        """
        if self._seg_data is None or self._bmode_image_data is None:
            self._emit_error("Cannot run motion compensation: missing segmentation or B-mode data")
            return

        if self._mc_worker and self._mc_worker.isRunning():
            self._mc_worker.quit()
            self._mc_worker.wait()

        # Use a fresh copy of the base seg (without MC) so we don't accumulate MC
        base_seg = self._seg_data
        self._mc_worker = MotionCompWorker(
            base_seg, self._image_data, self._bmode_image_data, mc_kwargs
        )
        self._mc_worker.finished.connect(self._on_mc_complete)
        self._mc_worker.error_msg.connect(self._emit_error)
        self.motion_comp_started.emit()
        self._mc_worker.start()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.quit()
            self._scan_worker.wait()
            self._scan_worker = None

        if self._bmode_scan_worker and self._bmode_scan_worker.isRunning():
            self._bmode_scan_worker.quit()
            self._bmode_scan_worker.wait()
            self._bmode_scan_worker = None

        if self._seg_worker and self._seg_worker.isRunning():
            self._seg_worker.quit()
            self._seg_worker.wait()
            self._seg_worker = None

    # ============================================================================
    # ANALYSIS METHODS
    # ============================================================================
    
    def get_analysis_types(self) -> tuple:
        """Get available analysis types and functions."""
        return self._analysis_types, self._analysis_functions
    
    def set_analysis_type(self, analysis_type: str) -> bool:
        """
        Set the selected analysis type.
        
        Args:
            analysis_type: Analysis type to select
            
        Returns:
            bool: True if successful
        """
        if analysis_type in self._analysis_types:
            self._selected_analysis_type = analysis_type
            return True
        else:
            print(f"DEBUG: Invalid analysis type: {analysis_type}")
            return False

    def get_analysis_functions(self, analysis_type: str) -> dict:
        """
        Get available functions for an analysis type.
        
        Args:
            analysis_type: Analysis type

        Returns:
            dict: Available functions for the analysis type
        """
        # In CEUS engine, analysis_functions is a flat dict of all available curve functions
        # that are applicable to both 'curves' and 'curves_paramap' analysis types.
        if analysis_type in self._analysis_functions and isinstance(self._analysis_functions[analysis_type], dict):
            return self._analysis_functions[analysis_type]
        
        return self._analysis_functions

    def set_analysis_data(self, analysis_data: CurvesAnalysis) -> None:
        """
        Store completed analysis data.
        
        Args:
            analysis_data: Completed analysis data
        """
        self._analysis_data = analysis_data
        # Signal that analysis is complete
        self.analysis_completed.emit(analysis_data)

    def run_analysis(self, analysis_type: str, image_data: UltrasoundImage, 
                    config_data: Any, seg_data: CeusSeg, 
                    selected_functions: List[str], **kwargs) -> None:
        """
        Run the analysis in a background thread.
        """
        # Stop existing worker if running
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.quit()
            self._analysis_worker.wait()
            
        self._analysis_worker = AnalysisWorker(
            analysis_type, image_data, config_data, seg_data, selected_functions, kwargs
        )
        
        self._analysis_worker.finished.connect(self._on_analysis_worker_finished)
        self._analysis_worker.error_msg.connect(self._emit_error)
        
        self._set_loading(True)
        self._analysis_worker.start()
        
    def _on_analysis_worker_finished(self, analysis_obj: Any) -> None:
        """Handle analysis completion."""
        self._set_loading(False)
        self._analysis_data = analysis_obj
        self.analysis_completed.emit(analysis_obj)
