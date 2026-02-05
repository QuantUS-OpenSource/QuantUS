import numpy as np
from pathlib import Path
from typing import Optional

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

class DicomLoader:
    """
    Utility class for loading and processing DICOM files for ultrasound imaging.
    """
    
    @staticmethod
    def load_dicom_file(dicom_file_path: str) -> Optional[np.ndarray]:
        """
        Load a DICOM file and return the processed pixel data.
        
        Args:
            dicom_file_path (str): Path to the DICOM file to load
            
        Returns:
            np.ndarray: Processed pixel data as a 2D numpy array, or None if failed
        """
        if not PYDICOM_AVAILABLE:
            print("pydicom is not installed. Cannot load DICOM files.")
            return None
            
        try:
            dicom_path = Path(dicom_file_path)
            if not dicom_path.exists() or not dicom_path.is_file():
                print(f"DICOM file not found: {dicom_file_path}")
                return None
                
            # Read the DICOM file
            dicom_data = pydicom.dcmread(str(dicom_path))
            
            # Extract pixel data
            if hasattr(dicom_data, 'pixel_array'):
                dicom_pixels = dicom_data.pixel_array
                
                # Convert to grayscale if needed (handle different DICOM formats)
                if len(dicom_pixels.shape) == 4:
                    # 4D DICOM (frames, height, width, channels) - take first frame
                    dicom_pixels = dicom_pixels[0]
                if len(dicom_pixels.shape) == 3:
                    # RGB or multi-frame DICOM - convert to grayscale
                    if dicom_pixels.shape[2] == 3:  # RGB
                        dicom_pixels = np.dot(dicom_pixels[...,:3], [0.2989, 0.5870, 0.1140])
                    elif dicom_pixels.shape[0] < dicom_pixels.shape[2]:  # Multi-frame
                        dicom_pixels = dicom_pixels[0]  # Take first frame
                
                # Normalize to 0-255 range
                if dicom_pixels.dtype != np.uint8:
                    dicom_pixels = ((dicom_pixels - dicom_pixels.min()) / 
                                  (max(1, dicom_pixels.max() - dicom_pixels.min())) * 255).astype(np.uint8)
                
                # Crop black regions from top and bottom
                dicom_pixels = DicomLoader.crop_black_regions(dicom_pixels)
                
                return dicom_pixels
            else:
                print("No pixel data found in DICOM file")
                return None
                
        except Exception as e:
            print(f"Failed to load DICOM file: {e}")
            return None

    @staticmethod
    def crop_black_regions(image: np.ndarray) -> np.ndarray:
        """
        Crop a fixed number of rows from top and bottom of the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Cropped image with specified rows removed
        """
        # Define fixed number of rows to crop from top and bottom
        crop_top = 175   # Number of rows to crop from top
        crop_bottom = 175  # Number of rows to crop from bottom
        
        # Ensure we don't crop more than the image height
        height = image.shape[0]
        if crop_top + crop_bottom >= height:
            # If we try to crop too much, crop half from each side
            crop_top = crop_bottom = height // 4
        
        # Crop the image
        cropped_image = image[crop_top:height - crop_bottom, :]
        
        print(f"DICOM cropped: {image.shape} -> {cropped_image.shape} (removed {crop_top} from top, {crop_bottom} from bottom)")
        return cropped_image
