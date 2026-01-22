import numpy as np
from skimage import exposure, filters
from skimage.restoration import denoise_wavelet, estimate_sigma
import cv2

from .decorators import required_kwargs
from engines.ceus.src.data_objs.image import UltrasoundImage
from .transforms import resample_to_spacing_2d, resample_to_spacing_3d

@required_kwargs('arr_to_standardize')
def standardize(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Standardize the pixel data and/or intensities for analysis of an UltrasoundImage object.

    Kwargs:
        arr_to_standardize (str): One of 'both', 'intensities', 'pixel_data' to specify which arrays to standardize.
                                   Default is 'both'.
    """
    arr_to_standardize = kwargs.get('arr_to_standardize', 'both')
    assert arr_to_standardize in ['both', 'intensities', 'pixel_data'], "arr_to_standardize must be one of ['both', 'intensities', 'pixel_data']"
    
    if arr_to_standardize in ['both', 'intensities']:
        mean = np.mean(image_data.intensities_for_analysis)
        std = np.std(image_data.intensities_for_analysis)
        if std > 0:
            image_data.intensities_for_analysis = (image_data.intensities_for_analysis - mean) / std
        else:
            image_data.intensities_for_analysis = image_data.intensities_for_analysis - mean
    if arr_to_standardize in ['both', 'pixel_data']:
        mean = np.mean(image_data.pixel_data)
        std = np.std(image_data.pixel_data)
        if std > 0:
            image_data.pixel_data = (image_data.pixel_data - mean) / std
        else:
            image_data.pixel_data = image_data.pixel_data - mean

    return image_data

@required_kwargs('target_vox_size', 'interp')
def resample(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Resample the image data to a new spacing.

    Kwargs:
        target_vox_size: tuple of (z, y, x) spacing in mm to resample the image to.
        interp: interpolation method, one of 'nearest', 'linear', 'cubic'.
    """
    target_vox_size = kwargs['target_vox_size']
    interp = kwargs['interp']

    if image_data.intensities_for_analysis.ndim == 4:
        image_data.pixel_data = resample_to_spacing_3d(image_data.pixel_data, image_data.pixdim, target_vox_size, interp=interp)
        image_data.intensities_for_analysis = resample_to_spacing_3d(image_data.intensities_for_analysis, image_data.pixdim, target_vox_size, interp=interp)
    elif image_data.intensities_for_analysis.ndim == 3:
        image_data.pixel_data = resample_to_spacing_2d(image_data.pixel_data, image_data.pixdim, target_vox_size, interp=interp)
        image_data.intensities_for_analysis = resample_to_spacing_2d(image_data.intensities_for_analysis, image_data.pixdim, target_vox_size, interp=interp)
    else:
        raise ValueError("Image data must be either 3D or 4D for resampling.")

    image_data.extras_dict['original_spacing'] = image_data.pixdim
    image_data.pixdim = target_vox_size

    return image_data

@required_kwargs('scale_factor', 'interp')
def enhance_spatial_resolution(volume, scale_factor=1.0, interp='cubic', **kwargs):
    """
    Enhance spatial resolution by upsampling the image (increasing pixel count).
    
    Args:
        volume: 2D or 3D array, or UltrasoundImage object
        scale_factor: Multiplier for the current resolution (e.g., 2.0 to double pixels)
        interp: Interpolation method ('nearest', 'linear', 'cubic')
    """
    if scale_factor == 1.0:
        return volume

    if isinstance(volume, UltrasoundImage):
        image_data = volume
        # Adjust spacing inversely to scale factor
        target_vox_size = tuple(np.array(image_data.pixdim) / scale_factor)
        image_data = resample(image_data, target_vox_size=target_vox_size, interp=interp)
        return image_data

    # For raw arrays (2D or 3D)
    if volume.ndim == 2:
        orig_spacing = (1.0, 1.0)
        target_spacing = (1.0 / scale_factor, 1.0 / scale_factor)
        return resample_to_spacing_2d(volume, orig_spacing, target_spacing, interp=interp)
    else:
        orig_spacing = (1.0, 1.0, 1.0)
        target_spacing = (1.0 / scale_factor, 1.0 / scale_factor, 1.0 / scale_factor)
        return resample_to_spacing_3d(volume, orig_spacing, target_spacing, interp=interp)

@required_kwargs('method')
def enhance_contrast_resolution(volume, method='clahe', **kwargs):
    """
    Enhance contrast resolution using various intensity transformation methods. 
    Supports both 2D and 3D inputs and works as a plugin.
    
    Args:
        volume: 2D or 3D array, or UltrasoundImage object
        method: Enhancement method ('clahe', 'gamma', 'log', 'sigmoid', 'adaptive_hist')
    """
    # Plugin support
    if isinstance(volume, UltrasoundImage):
        image_data = volume
        image_data.pixel_data = enhance_contrast_resolution(image_data.pixel_data, method=method, **kwargs)
        return image_data

    # Method validation and execution
    if method == 'gamma':
        gamma = kwargs.get('gamma', 0.7)
        return exposure.adjust_gamma(volume, gamma)
    elif method == 'log':
        gain = kwargs.get('gain', 1)
        return exposure.adjust_log(volume, gain=gain)
    elif method == 'adaptive_hist':
        clip_limit = kwargs.get('clip_limit', 0.01)
        return exposure.equalize_adapthist(volume, clip_limit=clip_limit)
    elif method == 'sigmoid':
        cutoff = kwargs.get('cutoff', 0.5)
        gain = kwargs.get('gain', 10)
        return exposure.adjust_sigmoid(volume, cutoff=cutoff, gain=gain)

    # Slice-based methods (e.g., OpenCV CLAHE)
    if method == 'clahe':
        is_2d = volume.ndim == 2
        if is_2d:
            volume = volume[:, :, np.newaxis]
            
        enhanced = np.zeros_like(volume)
        clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit', 3.0), tileGridSize=kwargs.get('tile_grid_size', (8, 8)))
        
        for z in range(volume.shape[2]):
            slice_2d = volume[:, :, z]
            # OpenCV requires uint8 or uint16
            if slice_2d.dtype not in [np.uint8, np.uint16]:
                s_min, s_max = slice_2d.min(), slice_2d.max()
                if s_max > s_min:
                    slice_to_proc = ((slice_2d - s_min) / (s_max - s_min) * 255).astype(np.uint8)
                else:
                    slice_to_proc = slice_2d.astype(np.uint8)
            else:
                slice_to_proc = slice_2d
                
            enhanced[:, :, z] = clahe.apply(slice_to_proc)
            
        return enhanced[:, :, 0] if is_2d else enhanced

    return volume

def enhance_image(volume, method='clahe', **kwargs):
    """Alias for backward compatibility."""
    return enhance_contrast_resolution(volume, method, **kwargs)

@required_kwargs('wavelet', 'sigma_scale')
def denoise_ceus_wavelet(volume_3d, wavelet='db1', sigma_scale=0.8):
    """
    Gentler wavelet denoising. Supports both 2D and 3D inputs and works as a plugin.
    
    Args:
        volume_3d: 2D or 3D array, or UltrasoundImage object
        sigma_scale: Scale factor for noise estimate (0.3-0.7 for gentle, 1.0 for normal)
    """
    if isinstance(volume_3d, UltrasoundImage):
        image_data = volume_3d
        image_data.pixel_data = denoise_ceus_wavelet(image_data.pixel_data, wavelet=wavelet, sigma_scale=sigma_scale)
        return image_data

    is_2d = volume_3d.ndim == 2
    if is_2d:
        volume_3d = volume_3d[:, :, np.newaxis]

    denoised = np.zeros_like(volume_3d, dtype=np.float32)
    
    for z in range(volume_3d.shape[2]):
        slice_2d = volume_3d[:, :, z].astype(np.float32)
        
        # Normalize to [0, 1] for stable denoising
        slice_min, slice_max = slice_2d.min(), slice_2d.max()
        range_val = slice_max - slice_min
        if range_val > 0:
            slice_norm = (slice_2d - slice_min) / range_val
        else:
            slice_norm = slice_2d
        
        # Estimate sigma and apply wavelet denoising
        sigma_est = estimate_sigma(slice_norm, average_sigmas=True)
        denoised_slice = denoise_wavelet(
            slice_norm,
            method='BayesShrink',
            mode='soft',
            wavelet=wavelet,
            rescale_sigma=True,
            sigma=sigma_est * sigma_scale
        )
        
        # Scale back to original range
        if range_val > 0:
            denoised[:, :, z] = denoised_slice * range_val + slice_min
        else:
            denoised[:, :, z] = denoised_slice
    
    return denoised[:, :, 0] if is_2d else denoised