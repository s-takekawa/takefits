"""Utilities for positioning click coordinate labels on Qt canvases."""

from typing import Tuple


DEFAULT_LABEL_POS_X = 0.99
DEFAULT_LABEL_POS_Y = 0.99
DEFAULT_LABEL_WIDTH = 250
DEFAULT_LABEL_HEIGHT = 30


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def compute_click_label_geometry(
    canvas_width: int,
    canvas_height: int,
    *,
    pos_x: float = DEFAULT_LABEL_POS_X,
    pos_y: float = DEFAULT_LABEL_POS_Y,
    requested_width: int = DEFAULT_LABEL_WIDTH,
    requested_height: int = DEFAULT_LABEL_HEIGHT,
) -> Tuple[int, int, int, int]:
    """
    Compute a clamped label rectangle anchored by the label's top-right corner.

    `pos_x` is interpreted as the right edge ratio from the canvas left.
    `pos_y` follows the existing legacy convention: ratio from canvas bottom.
    """
    width_px = max(0, int(canvas_width))
    height_px = max(0, int(canvas_height))
    if width_px <= 0 or height_px <= 0:
        return 0, 0, 0, 0

    target_w = max(0, _to_int(requested_width, DEFAULT_LABEL_WIDTH))
    target_h = max(0, _to_int(requested_height, DEFAULT_LABEL_HEIGHT))
    label_w = min(target_w, width_px)
    label_h = min(target_h, height_px)

    normalized_x = _to_float(pos_x, DEFAULT_LABEL_POS_X)
    normalized_y = _to_float(pos_y, DEFAULT_LABEL_POS_Y)
    anchor_x = width_px * normalized_x
    anchor_y = height_px * (1.0 - normalized_y)

    x = int(round(anchor_x - label_w))
    y = int(round(anchor_y))

    max_x = max(0, width_px - label_w)
    max_y = max(0, height_px - label_h)
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return x, y, label_w, label_h
