"""
Resolve paths to bundled resources in both dev and frozen contexts.
"""
import os
import sys


def resource_path(relative_path: str) -> str:
    """
    Return the absolute path to a resource file.

    In a PyInstaller --onedir frozen build, resources are extracted to
    sys._MEIPASS. In dev, they're relative to the repo root.

    Usage:
        from src.qus.resource_path import resource_path
        icon = resource_path("Images/logo.png")
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        # Repo root: two levels up from this file (src/qus/resource_path.py)
        base = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    return os.path.join(base, relative_path)
