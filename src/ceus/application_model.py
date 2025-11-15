"""
Unified Application Model for QuantUS GUI MVC architecture

This model centralizes all data management and business logic for the entire application,
replacing the individual models for each component.
"""

import os
from typing import Dict, Any, Optional
from PyQt6.QtCore import QThread, pyqtSignal

from .mvc.base_model import BaseModel
from engines.ceus.src.image_loading.options import get_scan_loaders
from engines.ceus.src.seg_loading.options import get_seg_loaders
from engines.ceus.src.entrypoints import scan_loading_step, seg_loading_step
from engines.ceus.src.data_objs import UltrasoundImage, CeusSeg


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
    segmentation_loaded = pyqtSignal(CeusSeg)

    def __init__(self):
        super().__init__()
        
        # Image loading state
        self._scan_loaders: Dict[str, Any] = {}
        self._selected_scan_type: Optional[str] = None
        self._image_data: Optional[UltrasoundImage] = None
        self._scan_worker: Optional[ScanLoadingWorker] = None
        
        # Segmentation loading state
        self._seg_loaders: Dict[str, Any] = {}
        self._selected_seg_type: Optional[str] = None
        self._seg_data: Optional[CeusSeg] = None
        self._seg_worker: Optional[SegLoadingWorker] = None
        
        # Initialize loaders
        self._load_scan_loaders()
        self._load_seg_loaders()
    
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
        self._set_loading(False)
        
        # Check if loading was successful
        if isinstance(image_data, UltrasoundImage):
            self._image_data = image_data
            self.image_loaded.emit(image_data)
        else:
            print(f"DEBUG: Image loading failed - invalid image data:")
            print(f"  - scan_path: {getattr(image_data, 'scan_path', 'Missing')}")
            print(f"  - has pixel_data: {hasattr(image_data, 'pixel_data')}")
            print(f"  - pixel_data is None: {getattr(image_data, 'pixel_data', None) is None}")
            print(f"  - has intensity: {hasattr(image_data, 'intensities_for_analysis')}")
            print(f"  - intensities_for_analysis is None: {getattr(image_data, 'intensities_for_analysis', None) is None}")
            self._emit_error("Failed to load image data - image loading was unsuccessful")
    
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
    
    def load_segmentation(self, seg_path: str, seg_loader_kwargs: Dict[str, Any] = None) -> None:
        """
        Load segmentation data.
        
        Args:
            seg_path: Path to segmentation file
            seg_loader_kwargs: Additional loader arguments (optional)
        """
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
        
        # Check if loading was successful
        if seg_data and hasattr(seg_data, 'seg_mask') and seg_data.seg_mask is not None:
            self._seg_data = seg_data
            self.segmentation_loaded.emit(seg_data)
        else:
            print(f"DEBUG: Segmentation loading failed - invalid seg data")
            self._emit_error("Failed to load segmentation data")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.quit()
            self._scan_worker.wait()
            self._scan_worker = None
        
        if self._seg_worker and self._seg_worker.isRunning():
            self._seg_worker.quit()
            self._seg_worker.wait()
            self._seg_worker = None
