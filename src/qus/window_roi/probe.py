"""Pick the correlation-length probe preset (C3 / L15 / L20, quantus.seg_loading.correlation_length
PROBE_PRESETS) from the scan's Clarius rf.yml. The yml has no probe name, but its probe block gives the
element pitch (um) and, for curvilinear probes, the radius (mm):
    C3 HD3: pitch 300, radius 45     L15 HD3: pitch 260     L20 HD3: pitch 130
"""

import re
from pathlib import Path
from typing import Dict, Optional

# preset -> (pitch um, radius mm; 0 = linear)
PROBE_SIGNATURES = {
    "C3": (300.0, 45.0),
    "L15": (260.0, 0.0),
    "L20": (130.0, 0.0),
}
_TOLERANCE = 0.05  # relative


def _rf_yml(scan_path) -> Optional[Path]:
    if not scan_path:
        return None
    p = Path(scan_path)
    for candidate in (p.with_suffix(".yml"), p.with_name(p.stem.replace("_env", "_rf") + ".yml")):
        if candidate.is_file():
            return candidate
    return None


def read_probe_block(yml_path) -> Dict[str, float]:
    """Numeric fields of the top-level 'probe:' block of a Clarius rf.yml (no yaml dependency)."""
    out: Dict[str, float] = {}
    in_probe = False
    with open(yml_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if not line[0].isspace():
                in_probe = line.strip().startswith("probe:")
                continue
            if in_probe and ":" in line:
                key, value = line.split(":", 1)
                m = re.search(r"[-+]?\d*\.?\d+", value)
                if m:
                    out[key.strip()] = float(m.group())
    return out


def _close(a: float, b: float) -> bool:
    return (a == 0 and b == 0) or (b != 0 and abs(a - b) <= _TOLERANCE * abs(b))


def detect_probe_preset(image_data) -> Optional[str]:
    """Preset name matching the scan's probe, or None (unknown probe / not a Clarius scan)."""
    try:
        yml = _rf_yml(getattr(image_data, "scan_path", None))
        if yml is None:
            return None
        probe = read_probe_block(yml)
    except OSError:
        return None
    pitch, radius = probe.get("pitch"), probe.get("radius", 0.0)
    if pitch is None:
        return None
    for name, (p, r) in PROBE_SIGNATURES.items():
        if _close(pitch, p) and _close(radius, r):
            return name
    return None


# Known spectral nulls (MHz) per (probe preset, software version prefix): kept out of the analysis band
# by the multitaper band (avoid_freq: peak searched above the null, band kept W away from it).
# L20 on software 13.0.0 has a null at ~7.5 MHz (S1Study DetermineSpectralParameters / DetermineBandwidth).
KNOWN_SPECTRAL_NULLS = {
    ("L20", "13.0.0"): (7.5,),
}


def read_software_version(yml_path) -> Optional[str]:
    """Top-level 'software version' of a Clarius rf.yml, e.g. '13.0.0-13049d'."""
    with open(yml_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("software version:"):
                return line.split(":", 1)[1].strip()
    return None


def known_spectral_nulls(image_data) -> tuple:
    """Known spectral nulls (MHz) for the scan's probe and software version, () if none / unknown."""
    preset = detect_probe_preset(image_data)
    if preset is None:
        return ()
    try:
        yml = _rf_yml(getattr(image_data, "scan_path", None))
        version = read_software_version(yml) if yml is not None else None
    except OSError:
        return ()
    if not version:
        return ()
    for (probe, prefix), nulls in KNOWN_SPECTRAL_NULLS.items():
        if probe == preset and version.startswith(prefix):
            return tuple(nulls)
    return ()
