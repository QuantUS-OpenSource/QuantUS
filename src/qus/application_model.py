"""
Unified Application Model for QuantUS GUI MVC architecture

This model centralizes all data management and business logic for the entire application,
replacing the individual models for each component.
"""

import os
from typing import Dict, Any, Optional, Tuple
from PyQt6.QtCore import QThread, pyqtSignal

from src.qus.mvc.base_model import BaseModel
from engines.qus.quantus.image_loading.utc_loaders.options import get_scan_loaders
from engines.qus.quantus.seg_loading.options import get_seg_loaders
from engines.qus.quantus.analysis_config.utc_config.options import get_config_loaders
from engines.qus.quantus.analysis.options import get_analysis_types, get_required_kwargs
from engines.qus.quantus.visualizations.options import get_visualization_types, get_compatible_funcs
from engines.qus.quantus.data_export.options import get_data_export_types
from engines.qus.quantus.entrypoints import (
    scan_loading_step,
    seg_loading_step,
    analysis_config_step,
    analysis_step,
    visualization_step,
    data_export_step,
)
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig
from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis


class ScanLoadingWorker(QThread):
    """Worker thread for time-consuming scan loading operations."""
    finished = pyqtSignal(UltrasoundRfImage)
    error_msg = pyqtSignal(str)

    def __init__(self, scan_type: str, image_path: str, phantom_path: str, scan_loader_kwargs: Dict[str, Any]):
        super().__init__()
        self.scan_type = scan_type
        self.image_path = image_path
        self.phantom_path = phantom_path
        self.scan_loader_kwargs = scan_loader_kwargs

    def run(self):
        """Execute the scan loading in background thread."""
        try:
            image_data = scan_loading_step(
                self.scan_type, 
                self.image_path, 
                self.phantom_path, 
                **self.scan_loader_kwargs
            )
            self.finished.emit(image_data)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_msg.emit(f"Error loading image: {e}")


class SegLoadingWorker(QThread):
    """Worker thread for time-consuming segmentation loading operations."""
    finished = pyqtSignal(BmodeSeg)
    error_msg = pyqtSignal(str)

    def __init__(self, seg_type: str, seg_path: str, image_data: UltrasoundRfImage, seg_loader_kwargs: Dict[str, Any]):
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
                self.image_data.phantom_path,
                **self.seg_loader_kwargs
            )
            
            self.finished.emit(seg_data)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_msg.emit(f"Error loading segmentation: {e}")


class ConfigLoadingWorker(QThread):
    """Worker thread for time-consuming config loading operations."""
    finished = pyqtSignal(RfAnalysisConfig)
    error_msg = pyqtSignal(str)

    def __init__(self, config_type: str, config_path: str, scan_path: str, phantom_path: str, config_kwargs: Dict[str, Any]):
        super().__init__()
        self.config_type = config_type
        self.config_path = config_path
        self.scan_path = scan_path
        self.phantom_path = phantom_path
        self.config_kwargs = config_kwargs

    def run(self):
        """Execute the config loading in background thread."""
        try:
            config_data = analysis_config_step(
                self.config_type,
                self.config_path,
                self.scan_path,
                self.phantom_path,
                **self.config_kwargs
            )
            
            self.finished.emit(config_data)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_msg.emit(f"Error loading configuration: {e}")


class AnalysisWorker(QThread):
    """Worker thread for time-consuming analysis operations."""
    finished = pyqtSignal(ParamapAnalysis)
    error_msg = pyqtSignal(str)

    def __init__(self, analysis_type: str, image_data: UltrasoundRfImage, config_data: RfAnalysisConfig, 
                 seg_data: BmodeSeg, analysis_functions: list, analysis_kwargs: dict):
        super().__init__()
        self.analysis_type = analysis_type
        self.image_data = image_data
        self.config_data = config_data
        self.seg_data = seg_data
        self.analysis_functions = analysis_functions
        self.analysis_kwargs = analysis_kwargs

    def run(self):
        """Execute the analysis in background thread."""        
        try:
            analysis_data = analysis_step(
                self.analysis_type,
                self.image_data,
                self.config_data,
                self.seg_data,
                self.analysis_functions,
                **self.analysis_kwargs
            )
            self.finished.emit(analysis_data)
            
        except Exception as e:
            import traceback
            print(f"DEBUG: Exception in AnalysisWorker.run(): {e}")
            traceback.print_exc()
            self.error_msg.emit(f"Error during analysis: {e}")


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
    image_loaded = pyqtSignal(UltrasoundRfImage)
    segmentation_loaded = pyqtSignal(BmodeSeg)
    config_loaded = pyqtSignal(RfAnalysisConfig)
    analysis_completed = pyqtSignal(ParamapAnalysis)

    def __init__(self):
        super().__init__()
        
        # Image loading state
        self._scan_loaders: Dict[str, Any] = {}
        self._selected_scan_type: Optional[str] = None
        self._image_data: Optional[UltrasoundRfImage] = None
        self._scan_worker: Optional[ScanLoadingWorker] = None
        
        # Segmentation loading state
        self._seg_loaders: Dict[str, Any] = {}
        self._selected_seg_type: Optional[str] = None
        self._seg_data: Optional[BmodeSeg] = None
        self._seg_worker: Optional[SegLoadingWorker] = None
        
        # Config loading state
        self._config_loaders: Dict[str, Any] = {}
        self._selected_config_type: Optional[str] = None
        self._config_data: Optional[RfAnalysisConfig] = None
        self._config_worker: Optional[ConfigLoadingWorker] = None
        
        # Analysis state
        self._analysis_types: Dict[str, Any] = {}
        self._analysis_functions: Dict[str, Any] = {}
        self._selected_analysis_type: Optional[str] = None
        self._analysis_data: Optional[ParamapAnalysis] = None
        self._analysis_worker: Optional[AnalysisWorker] = None

        # Visualization state
        self._visualization_types: Dict[str, Any] = {}
        self._visualization_functions: Dict[str, Any] = {}

        # Data export state
        self._data_export_types: Dict[str, Any] = {}
        self._data_export_functions: Dict[str, Any] = {}
        
        # Initialize loaders
        self._load_scan_loaders()
        self._load_seg_loaders()
        self._load_config_loaders()
        self._load_analysis_types()
        self._load_visualization_types()
        self._load_data_export_types()
    
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
    
    def _load_config_loaders(self) -> None:
        """Load available config loaders from backend."""
        try:
            self._config_loaders = get_config_loaders()
        except Exception as e:
            self._emit_error(f"Failed to load config loaders: {e}")
    
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
    def image_data(self) -> Optional[UltrasoundRfImage]:
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
            loader = self._scan_loaders[self._selected_scan_type]
            return loader.get('file_exts', [])
        except Exception as e:
            self._emit_error(f"Error getting file extensions: {e}")
            return []
        
    def get_image_loading_options(self) -> list:
        """
        Get required keyword arguments for the selected scan type.
        
        Returns:
            Tuple[list]: List of required keyword arguments and their default values
        """
        if not self._selected_scan_type or self._selected_scan_type not in self._scan_loaders:
            return []
        
        try:
            loader = self._scan_loaders[self._selected_scan_type]
            return loader['gui_kwargs'], loader['default_gui_kwarg_vals']
        except Exception as e:
            self._emit_error(f"Error getting required kwargs: {e}")
            return [], []
        
    def get_config_loading_options(self) -> list:
        """
        Get required keyword arguments for the selected config type.

        Returns:
            Tuple[list]: List of required keyword arguments and their default values
        """
        if not self._selected_config_type or self._selected_config_type not in self._config_loaders:
            return [], []

        try:
            loader = self._config_loaders[self._selected_config_type]
            gui_kwargs = loader.gui_kwargs if hasattr(loader, 'gui_kwargs') else []
            default_gui_kwarg_vals = loader.default_gui_kwarg_vals if hasattr(loader, 'default_gui_kwarg_vals') else []
            return gui_kwargs, default_gui_kwarg_vals
        except Exception as e:
            self._emit_error(f"Error getting required kwargs: {e}")
            return [], []

    def get_compatible_visualization_funcs(self, visualization_type: str) -> list:
        """
        Get compatible visualization functions for the currently loaded analysis data.
        
        Args:
            visualization_type: Selected visualization type

        Returns:
            list: List of compatible visualization function names
        """
        if not self._analysis_data:
            return []
        
        try:
            compatible_funcs = get_compatible_funcs(
                visualization_type,
                self._analysis_data.function_names
            )
            return compatible_funcs
        except Exception as e:
            self._emit_error(f"Error getting compatible visualization functions: {e}")
            return []

    def load_image(self, image_path: str, phantom_path: str = "", scan_loader_kwargs: Dict[str, Any] = None) -> None:
        """
        Load scan image data.
        
        Args:
            image_path: Path to image file
            phantom_path: Path to phantom file (optional)
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
            'phantom_path': phantom_path,
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
            phantom_path,
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
        
        phantom_path = input_data.get('phantom_path', '')
        if phantom_path and phantom_path.strip() and not os.path.exists(phantom_path):
            self._emit_error(f"Phantom file not found: {phantom_path}")
            return False
        
        return True
    
    def _on_image_loading_complete(self, image_data: UltrasoundRfImage) -> None:
        """
        Handle completion of scan loading.
        
        Args:
            image_data: Loaded ultrasound image data
        """
        self._set_loading(False)
        
        # Check if loading was successful
        if (hasattr(image_data, 'scan_path') and image_data.scan_path and 
            hasattr(image_data, 'rf_data') and image_data.rf_data is not None and
            hasattr(image_data, 'bmode') and image_data.bmode is not None):
            
            self._image_data = image_data
            self.image_loaded.emit(image_data)
        else:
            print(f"DEBUG: Image loading failed - invalid image data:")
            print(f"  - scan_path: {getattr(image_data, 'scan_path', 'Missing')}")
            print(f"  - has rf_data: {hasattr(image_data, 'rf_data')}")
            print(f"  - rf_data is None: {getattr(image_data, 'rf_data', None) is None}")
            print(f"  - has bmode: {hasattr(image_data, 'bmode')}")
            print(f"  - bmode is None: {getattr(image_data, 'bmode', None) is None}")
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
    def seg_data(self) -> Optional[BmodeSeg]:
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
                if self._image_data.spatial_dims == 2:
                    self._selected_seg_type = "pkl_roi"
                elif self._image_data.spatial_dims == 3:
                    self._selected_seg_type = "nifti_voi"
                else:
                    self._emit_error("Unsupported spatial dimensions for manual segmentation")
                    return False
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
            return loader.get('exts', [])
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
    
    def _on_segmentation_loading_complete(self, seg_data: BmodeSeg) -> None:
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
    
    # Config Loading Properties and Methods
    @property
    def config_loaders(self) -> Dict[str, Any]:
        """Get available config loaders."""
        return self._config_loaders
    
    @property
    def config_loader_names(self) -> list:
        """Get formatted config loader names for display."""
        if not self._config_loaders:
            return []
        
        names = [s.replace("_", " ").capitalize() for s in self._config_loaders.keys()]
        return [s.replace("rf", "RF").replace("config", "Config") for s in names]
    
    @property
    def selected_config_type(self) -> Optional[str]:
        """Get currently selected config type."""
        return self._selected_config_type
    
    @property
    def config_data(self) -> Optional[RfAnalysisConfig]:
        """Get the currently loaded config data."""
        return self._config_data
    
    def set_config_type(self, config_type_display_name: str) -> bool:
        """
        Set the selected config type.
        
        Args:
            config_type_display_name: Display name of the config type
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._config_loaders:
            self._emit_error("No config loaders available")
            return False
        
        try:
            # Convert display name back to internal name
            loader_names = list(self._config_loaders.keys())
            display_names = self.config_loader_names
            
            if config_type_display_name in display_names:
                index = display_names.index(config_type_display_name)
                self._selected_config_type = loader_names[index]
                return True
            else:
                self._emit_error(f"Invalid config type: {config_type_display_name}")
                return False
        except Exception as e:
            self._emit_error(f"Error setting config type: {e}")
            return False
    
    def get_config_file_extensions(self) -> list:
        """
        Get file extensions for the selected config type.
        
        Returns:
            list: File extensions supported by selected config loader
        """
        if not self._selected_config_type or self._selected_config_type not in self._config_loaders:
            return []
        
        try:
            loader = self._config_loaders[self._selected_config_type]
            exts = loader.supported_extensions if hasattr(loader, 'supported_extensions') else []
            return exts
        except Exception as e:
            self._emit_error(f"Error getting config file extensions: {e}")
            return []
    
    def load_config(self, config_path: str = "", config_kwargs: Dict[str, Any] = None) -> None:
        """
        Load configuration data.
        
        Args:
            config_path: Path to config file (empty string for custom config)
            config_kwargs: Additional loader arguments (optional)
        """
        if not self._image_data:
            self._emit_error("No image loaded - cannot load configuration")
            return
        
        if not self._selected_config_type:
            self._emit_error("No config type selected")
            return
        
        if config_kwargs is None:
            config_kwargs = {}
        
        # Validate input for file-based configs
        if config_path and not os.path.exists(config_path):
            self._emit_error(f"Configuration file not found: {config_path}")
            return
        
        # Stop any existing worker
        if self._config_worker and self._config_worker.isRunning():
            self._config_worker.quit()
            self._config_worker.wait()
        
        # Create and start worker
        self._config_worker = ConfigLoadingWorker(
            self._selected_config_type,
            config_path,
            self._image_data.scan_path,
            self._image_data.phantom_path,
            config_kwargs
        )
        
        # Connect worker signals
        self._config_worker.finished.connect(self._on_config_loading_complete)
        self._config_worker.error_msg.connect(self._emit_error)
        
        # Start loading
        self._set_loading(True)
        self._config_worker.start()
    
    def set_config_data(self, config_data: RfAnalysisConfig) -> None:
        """
        Set configuration data directly (for custom configs).
        
        Args:
            config_data: Analysis configuration data
        """
        self._config_data = config_data
        self.config_loaded.emit(config_data)
    
    def _on_config_loading_complete(self, config_data: RfAnalysisConfig) -> None:
        """
        Handle completion of config loading.
        
        Args:
            config_data: Loaded configuration data
        """
        self._set_loading(False)
        
        # Check if loading was successful
        if config_data and hasattr(config_data, 'transducer_freq_band'):
            self._config_data = config_data
            self.config_loaded.emit(config_data)
        else:
            print(f"DEBUG: Config loading failed - invalid config data")
            self._emit_error("Failed to load configuration data")
    
    # ============================================================================
    # ANALYSIS METHODS
    # ============================================================================
    
    def _load_analysis_types(self) -> None:
        """Load available analysis types from backend."""
        try:
            self._analysis_types, self._analysis_functions = get_analysis_types()
        except Exception as e:
            print(f"Error loading analysis types: {e}")
            self._analysis_types = {}
            self._analysis_functions = {}

    def _load_visualization_types(self) -> None:
        """Load available visualization types from backend."""
        try:
            self._visualization_types, self._visualization_functions = get_visualization_types()
        except Exception as e:
            print(f"Error loading visualization types: {e}")
            self._visualization_types = {}
            self._visualization_functions = {}

    def _load_data_export_types(self) -> None:
        """Load available data export types from backend."""
        try:
            self._data_export_types, self._data_export_functions = get_data_export_types()
        except Exception as e:
            print(f"Error loading data export types: {e}")
            self._data_export_types = {}
            self._data_export_functions = {}
    
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

    def get_analysis_functions(self, analysis_type: str, spatial_dims: int) -> dict:
        """
        Get available functions for an analysis type.
        
        Args:
            analysis_type: Analysis type
            spatial_dims: Spatial dimensions of the data

        Returns:
            dict: Available functions for the analysis type
        """
        funcs = self._analysis_functions.get(analysis_type, {})
        out = {}
        for func_name, func_info in funcs.items():
            if spatial_dims not in getattr(func_info, 'supported_spatial_dims', []):
                continue
            out[func_name] = func_info
        return out

    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================

    def get_visualization_types(self) -> tuple:
        """Get available visualization types and functions."""
        return self._visualization_types, self._visualization_functions

    def get_data_export_types(self) -> tuple:
        """Get available data export types and functions."""
        return self._data_export_types, self._data_export_functions

    def get_required_visualization_params(self, visualization_type: str, visualization_functions: list) -> list:
        """
        Get required parameters for selected visualization functions.

        Currently, visualization functions do not declare explicit kwarg metadata
        in a consistent way, so we return an empty list to allow progressing
        without extra parameters. This can be extended when metadata is added.
        """
        # Placeholder: no required params enforced
        return []

    def execute_visualization(
        self,
        visualization_type: str,
        analysis_data: ParamapAnalysis,
        visualization_functions: list,
        **visualization_kwargs,
    ) -> Any:
        """
        Execute the requested visualization and return the visualization object.
        """
        if analysis_data is None:
            raise ValueError("No analysis data provided for visualization")

        # Optionally we could set and clear loading here; controller already does it
        visualization_obj = visualization_step(
            visualization_type,
            analysis_data,
            visualization_functions,
            **visualization_kwargs,
        )
        return visualization_obj

    def execute_data_export(
        self,
        data_export_type: str,
        visualization_obj,
        data_export_path: str,
        data_export_funcs: list,
        **data_export_kwargs,
    ) -> Any:
        """Execute the requested data export and return the export object."""
        if visualization_obj is None:
            raise ValueError("No visualization object provided for data export")

        export_obj = data_export_step(
            data_export_type,
            visualization_obj,
            data_export_path,
            data_export_funcs,
            **data_export_kwargs,
        )
        return export_obj

    def get_required_params(self, analysis_type: str, analysis_functions: list) -> dict:
        """
        Get required parameters for selected analysis functions.
        
        Args:
            analysis_type: Analysis type
            analysis_functions: List of selected function names
            
        Returns:
            dict: Required parameters and their default values organized by function name
        """
        try:
            required_params = get_required_kwargs(analysis_type, analysis_functions)
            return required_params
        except Exception as e:
            print(f"Error getting required parameters: {e}")
            return {}

    def run_analysis(self, analysis_type: str, image_data: UltrasoundRfImage, config_data: RfAnalysisConfig,
                    seg_data: BmodeSeg, analysis_functions: list, **analysis_kwargs) -> None:
        """
        Run analysis with the specified parameters.
        
        Args:
            analysis_type: Type of analysis to run
            image_data: Image data
            config_data: Configuration data
            seg_data: Segmentation data
            analysis_functions: List of analysis functions to run
            **analysis_kwargs: Additional analysis parameters
        """
        if not self._config_data:
            print("DEBUG: No configuration loaded - cannot run analysis")
            self._emit_error("No configuration loaded - cannot run analysis")
            return
        
        if not self._seg_data:
            print("DEBUG: No segmentation loaded - cannot run analysis")
            self._emit_error("No segmentation loaded - cannot run analysis")
            return
        
        if not self._image_data:
            print("DEBUG: No image loaded - cannot run analysis")
            self._emit_error("No image loaded - cannot run analysis")
            return
        
        # Stop any existing worker
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.quit()
            self._analysis_worker.wait()
        
        # Create and start worker
        self._analysis_worker = AnalysisWorker(
            analysis_type,
            image_data,
            config_data,
            seg_data,
            analysis_functions,
            analysis_kwargs
        )
        
        # Connect worker signals
        self._analysis_worker.finished.connect(self._on_analysis_complete)
        self._analysis_worker.error_msg.connect(self._emit_error)
        
        # Start analysis
        self._set_loading(True)
        self._analysis_worker.start()
    
    def set_analysis_data(self, analysis_data: ParamapAnalysis) -> None:
        """
        Set analysis data directly.
        
        Args:
            analysis_data: Analysis results data
        """
        self._analysis_data = analysis_data
    
    def _on_analysis_complete(self, analysis_data: ParamapAnalysis) -> None:
        """
        Handle completion of analysis.
        
        Args:
            analysis_data: Completed analysis data
        """
        self._set_loading(False)
        
        # Check if analysis was successful
        if analysis_data:
            self._analysis_data = analysis_data
            self.analysis_completed.emit(analysis_data)
        else:
            self._emit_error("Failed to complete analysis")
    
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
        
        if self._config_worker and self._config_worker.isRunning():
            self._config_worker.quit()
            self._config_worker.wait()
            self._config_worker = None
        
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.quit()
            self._analysis_worker.wait()
            self._analysis_worker = None
