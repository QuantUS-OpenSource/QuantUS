import numpy as np
from skimage import exposure, filters
from skimage.restoration import denoise_wavelet
import cv2

from .decorators import required_kwargs
from ..data_objs.image import UltrasoundImage
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

def enhance_image(volume, method='clahe', **kwargs):
    """
    Enhance image quality using various methods
    
    Args:
        volume: 3D volume (Z, Y, X)
        method: Enhancement method ('clahe', 'gamma', 'log', 'sigmoid', 'adaptive_hist')
    """
    
    enhanced = np.zeros_like(volume)
    if method == 'gamma':
        # Gamma correction
        gamma = kwargs.get('gamma', 0.7)
        enhanced = exposure.adjust_gamma(volume, gamma)
    elif method == 'adaptive_hist':
        clip_limit = kwargs.get('clip_limit', 0.05)
        enhanced = exposure.equalize_adapthist(volume, clip_limit=clip_limit)

    for z in range(volume.shape[2]):
        slice_2d = volume[:, :, z]
            
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit',3.0), tileGridSize=(8,8))
            enhanced[:,:,z] = clahe.apply(slice_2d)
            
        elif method == 'sigmoid':
            # Sigmoid transformation
            cutoff = kwargs.get('cutoff', 0.5)
            gain = kwargs.get('gain', 10)
            enhanced[:,:,z] = exposure.adjust_sigmoid(slice_2d, cutoff=cutoff, gain=gain)      
    return enhanced

def denoise_ceus_wavelet(volume_3d, wavelet='db1', sigma_scale=0.8):
    """
    Gentler wavelet denoising
    
    Args:
        sigma_scale: Scale factor for noise estimate (0.3-0.7 for gentle, 1.0 for normal)
    """
    denoised = np.zeros_like(volume_3d, dtype=np.float32)
    
    for z in range(volume_3d.shape[2]):
        slice_2d = volume_3d[:, :, z].astype(np.float32)
        
        # Normalize to [0, 1]
        slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
        
        # Estimate sigma and scale it down
        from skimage.restoration import estimate_sigma
        sigma_est = estimate_sigma(slice_norm, average_sigmas=True)
        
        # Apply wavelet denoising with reduced sigma
        denoised_slice = denoise_wavelet(
            slice_norm,
            method='BayesShrink',
            mode='soft',
            wavelet=wavelet,
            rescale_sigma=True,
            sigma=sigma_est * sigma_scale  # KEY: Reduce denoising strength
        )
        
        # Scale back
        denoised[:, :, z] = denoised_slice * (slice_2d.max() - slice_2d.min()) + slice_2d.min()
    
    return denoised