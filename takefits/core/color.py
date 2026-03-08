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
    - CHANNEL: Channel map window
    """
    MAIN = "main"
    INTEG = "integ"
    CHANNEL = "channel"
