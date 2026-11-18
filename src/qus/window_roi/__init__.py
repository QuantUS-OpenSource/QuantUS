"""Power Guided ROI Selection: a segmentation type (correlation-length windows -> first-pass power
check -> ROI drawn with a live band check). Lightweight names here; the widget is in views/."""

POWER_GUIDED_ROI = "Power Guided ROI Selection"
# analysis config type: multitaper spectra over a Power Guided ROI -> band, windows (views/spectral_estimation_page.py)
POWER_CORRELATION_GUIDED = "Power and Correlation Guided"


def power_guided_roi_supported(image_data) -> bool:
    """Needs 2D (optionally multi-frame) RF data and reference phantom frames: a 2D phantom frame,
    or the loader's list of phantom frames (what get_phantom_frames() can return)."""
    rf = getattr(image_data, "rf_data", None)
    ref = getattr(image_data, "phantom_rf_data", None)
    if getattr(image_data, "spatial_dims", 2) != 2 or rf is None or getattr(rf, "ndim", 0) not in (2, 3):
        return False
    if getattr(image_data, "phantom_rf_frames", None) is not None:
        return True
    return ref is not None and getattr(ref, "ndim", 0) == 2
