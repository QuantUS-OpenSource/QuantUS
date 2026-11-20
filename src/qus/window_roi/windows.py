"""Recommended QUS window over part of the measured depth range.

The correlation lengths are measured over the full image depth, but the window is taken only over
the depths where the first pass finds usable signal (in sample AND reference): below the signal
the "correlation lengths" are noise and would make the window far too large.
"""

from typing import Optional, Tuple

import numpy as np

# Deepest depth (mm) the window is taken over, per probe preset: deeper than we image clinically,
# and on a sector probe the lateral window in mm keeps growing with depth. Measurement and plots
# still cover the full image depth.
WINDOW_MAX_DEPTH_MM = {
    "C3": 95.0,
}


def window_depth_range(usable: Optional[Tuple[float, float]], preset: Optional[str]) -> Optional[Tuple[float, float]]:
    """Depths the window is taken over: the first-pass usable depths (None = all), cut at the
    preset's WINDOW_MAX_DEPTH_MM."""
    cap = WINDOW_MAX_DEPTH_MM.get(preset)
    if cap is None:
        return usable
    if usable is None:
        return (-np.inf, cap)
    top, bottom = usable
    return (top, min(bottom, cap)) if top < cap else usable


def usable_depth_range(intersect) -> Optional[Tuple[float, float]]:
    """(top, bottom) mm of the first-pass depths with at least one block usable in sample AND
    reference, or None if no block is usable."""
    usable = np.asarray(intersect.usable_both_2d).any(axis=1)
    if not usable.any():
        return None
    depths = np.asarray(intersect.sample.depths, dtype=float)[usable]
    return float(depths.min()), float(depths.max())


def window_over_depths(result, depth_range: Optional[Tuple[float, float]]) -> dict:
    """result.recommended_window(), but the largest per-depth recommendation only over depths
    inside depth_range (mm, inclusive). None, or no finite value in the range -> whole range.
    Adds 'limited_to_mm': the range used, or None."""
    full = result.recommended_window()
    if depth_range is None:
        return dict(full, limited_to_mm=None)
    d = np.asarray(result.depths_mm, dtype=float)
    inside = (d >= depth_range[0] - 1e-9) & (d <= depth_range[1] + 1e-9)

    def largest(values):
        v = np.where(inside, np.asarray(values, dtype=float), np.nan)
        if not np.any(np.isfinite(v)):
            return np.nan, np.nan
        i = int(np.nanargmax(v))
        return float(v[i]), float(d[i])

    ax_mm, ax_mm_depth = largest(result.recommended_ax_window_mm)
    lat_mm, lat_mm_depth = largest(result.recommended_lat_window_mm)
    if not (np.isfinite(ax_mm) and np.isfinite(lat_mm)):
        return dict(full, limited_to_mm=None)
    ax_n, ax_depth = largest(result.recommended_ax_window_samples)
    lat_n, lat_depth = largest(result.recommended_lat_window_lines)
    return dict(full,
                ax_window_samples=ax_n, ax_at_depth_mm=ax_depth,
                ax_window_mm=ax_mm, ax_mm_at_depth_mm=ax_mm_depth,
                lat_window_lines=lat_n, lat_at_depth_mm=lat_depth,
                lat_window_mm=lat_mm, lat_mm_at_depth_mm=lat_mm_depth,
                limited_to_mm=(float(depth_range[0]), float(depth_range[1])))


def first_pass_band_for_roi(image_data, roi, correlation_kwargs: dict, report=None) -> Tuple[float, float]:
    """(fLow, fHigh) MHz first-pass band of an ROI that was not drawn in Power Guided ROI Selection
    (hand-drawn / loaded): the same steps, done automatically --
      correlation lengths of the reference inside the ROI (its A-lines and depths) -> window,
      first pass on reference + sample with that window, band check over the ROI.
    Fills roi.band. Raises ValueError if the ROI has no usable shared band."""
    from engines.qus.quantus.seg_loading.correlation_length import (
        calc_first_pass_from_image, correlation_lengths_from_image, intersect_first_pass,
    )
    from engines.qus.quantus.seg_loading.correlation_length.roi import check_sector_roi

    def say(frac, msg):
        if report is not None:
            report(frac, msg)

    say(0.0, "First-pass band for this ROI: correlation lengths inside the ROI")
    corr = correlation_lengths_from_image(image_data, source="phantom", line_range=tuple(roi.lines),
                                          depth_range_mm=tuple(roi.depth_range_mm), **correlation_kwargs)
    w = corr.recommended_window()
    ax_mm, lat_mm = w["ax_window_mm"], w["lat_window_mm"]
    if not (np.isfinite(ax_mm) and np.isfinite(lat_mm)):
        raise ValueError("No correlation-length window inside the ROI (the correlation never dropped below "
                         "the threshold); try a lower threshold or a larger ROI.")
    say(0.5, f"First-pass band for this ROI: first pass with a {ax_mm:.1f} x {lat_mm:.1f} mm window")
    fs = getattr(image_data, "sampling_frequency", None)
    kw = dict(window_length_mm=ax_mm, block_width_mm=lat_mm, ax_step_mm=round(0.2 * ax_mm, 2),
              sampling_frequency_mhz=float(fs) / 1e6 if fs else None)
    ref = calc_first_pass_from_image(image_data, "phantom", **kw)
    smp = calc_first_pass_from_image(image_data, "scan", block_lines=ref.block_lines, **kw)
    check_sector_roi(intersect_first_pass(smp, ref), roi)
    from engines.qus.quantus.seg_loading.correlation_length.roi import valid_band
    b = roi.band
    if b is None or not valid_band((b.f_low_roi, b.f_high_roi)):
        raise ValueError("No first-pass band shared by sample and reference inside this ROI "
                         f"({roi.band_error or 'no usable shared band'}). Go back and redraw the ROI where "
                         "the first pass shows signal in both (the magenta region), e.g. below the "
                         "transducer ring-down near the top of the image.")
    return float(b.f_low_roi), float(b.f_high_roi)
