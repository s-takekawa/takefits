"""
Color mode enumeration for view settings.

This module defines ColorMode which specifies which view plane's color settings
are being configured. This is placed in core/ to avoid layer violations (ui/tools
importing from each other).
"""
from enum import Enum


class ColorMode(Enum):
    """
    Enumeration for different color mode contexts.
    
    - MAIN: Main viewer (xy plane)
    - INTEG: Integration result window
    - PV: PV diagram window
    - CHANNEL: Channel map window
    """
    MAIN = "main"
    INTEG = "integ"
    PV = "pv"
    CHANNEL = "channel"


def default_color_settings() -> dict:
    return {
        "min_val": None,
        "max_val": None,
        "log_scale": False,
        "gamma_value": 1.0,
        "invert": False,
        "color_pattern": None,
    }


def default_color_settings_map() -> dict:
    return {mode: default_color_settings() for mode in ColorMode}
