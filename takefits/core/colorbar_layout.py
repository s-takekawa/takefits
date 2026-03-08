from typing import Tuple


_PLACEMENT_ORIENTATION = {
    "right": "vertical",
    "left": "vertical",
    "top": "horizontal",
    "bottom": "horizontal",
    "inside-right": "vertical",
    "inside-left": "vertical",
    "inside-top": "horizontal",
    "inside-bottom": "horizontal",
}


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def orientation_for_placement(placement: str, fallback: str = "vertical") -> str:
    key = str(placement or "").strip().lower()
    orientation = _PLACEMENT_ORIENTATION.get(key)
    if orientation in ("vertical", "horizontal"):
        return orientation
    fb = str(fallback or "").strip().lower()
    if fb in ("vertical", "horizontal"):
        return fb
    return "vertical"


def _normalize_align(align: str) -> str:
    value = str(align or "").strip().lower()
    if value in ("start", "center", "end"):
        return value
    return "center"


def _normalize_length_mode(length_mode: str) -> str:
    value = str(length_mode or "").strip().lower()
    if value == "match":
        return "ratio"
    if value in ("ratio", "px"):
        return value
    return "ratio"


def _compute_length(
    full_length: float,
    fig_dim_px: float,
    length_mode: str,
    length_value: float,
    min_fraction: float,
) -> float:
    full_length = clamp_float(full_length, min_fraction, 1.0)
    fig_dim_px = max(1.0, float(fig_dim_px))
    raw_mode = str(length_mode or "").strip().lower()
    legacy_match = raw_mode == "match"
    mode = _normalize_length_mode(length_mode)

    if mode == "px":
        try:
            px = float(length_value)
        except Exception:
            px = float(full_length * fig_dim_px)
        length = px / fig_dim_px
    else:
        if legacy_match:
            ratio = 1.0
        else:
            try:
                ratio = float(length_value)
            except Exception:
                ratio = 1.0
        ratio = clamp_float(ratio, 0.05, 1.0)
        length = full_length * ratio

    length = clamp_float(length, min_fraction, full_length)
    return length


def compute_colorbar_geometry(
    ax_bounds,
    fig_w_px: float,
    fig_h_px: float,
    *,
    placement: str = "right",
    align: str = "center",
    gap_px: float = 24.0,
    gap_x_px: float = None,
    gap_y_px: float = None,
    thickness_px: float = 24.0,
    length_mode: str = "ratio",
    length_value: float = 1.0,
    min_fraction: float = 0.003,
) -> Tuple[float, float, float, float, str]:
    ax_x, ax_y, ax_w, ax_h = [float(v) for v in ax_bounds]
    ax_w = clamp_float(ax_w, min_fraction, 1.0)
    ax_h = clamp_float(ax_h, min_fraction, 1.0)
    ax_right = ax_x + ax_w
    ax_top = ax_y + ax_h

    fig_w_px = max(1.0, float(fig_w_px))
    fig_h_px = max(1.0, float(fig_h_px))

    orientation = orientation_for_placement(placement)
    place = str(placement or "").strip().lower()
    if place not in _PLACEMENT_ORIENTATION:
        place = "right"

    align = _normalize_align(align)
    try:
        gap_px = float(gap_px)
    except Exception:
        gap_px = 24.0
    try:
        gap_x_px = float(gap_x_px) if gap_x_px is not None else float(gap_px)
    except Exception:
        gap_x_px = float(gap_px)
    try:
        gap_y_px = float(gap_y_px) if gap_y_px is not None else float(gap_px)
    except Exception:
        gap_y_px = float(gap_px)
    try:
        thickness_px = float(thickness_px)
    except Exception:
        thickness_px = 24.0

    gap_px = clamp_float(gap_px, 0.0, 200.0)
    gap_x_px = clamp_float(gap_x_px, 0.0, 200.0)
    gap_y_px = clamp_float(gap_y_px, 0.0, 200.0)
    thickness_px = clamp_float(thickness_px, 1.0, 400.0)
    gap_x = gap_x_px / fig_w_px
    gap_y = gap_y_px / fig_h_px

    if orientation == "vertical":
        width = clamp_float(thickness_px / fig_w_px, min_fraction, 0.2)
        height = _compute_length(ax_h, fig_h_px, length_mode, length_value, min_fraction)

        if align == "start":
            pos_y = ax_y + gap_y
        elif align == "end":
            pos_y = ax_top - height - gap_y
        else:
            pos_y = ax_y + 0.5 * (ax_h - height)

        if place == "left":
            pos_x = ax_x - gap_x - width
        elif place == "inside-left":
            pos_x = ax_x + gap_x
        elif place == "inside-right":
            pos_x = ax_right - gap_x - width
        else:  # right
            pos_x = ax_right + gap_x
    else:
        height = clamp_float(thickness_px / fig_h_px, min_fraction, 0.2)
        width = _compute_length(ax_w, fig_w_px, length_mode, length_value, min_fraction)

        if align == "start":
            pos_x = ax_x + gap_x
        elif align == "end":
            pos_x = ax_right - width - gap_x
        else:
            pos_x = ax_x + 0.5 * (ax_w - width)

        if place == "bottom":
            pos_y = ax_y - gap_y - height
        elif place == "inside-bottom":
            pos_y = ax_y + gap_y
        elif place == "inside-top":
            pos_y = ax_top - gap_y - height
        else:  # top
            pos_y = ax_top + gap_y

    pos_x = clamp_float(pos_x, 0.0, max(0.0, 1.0 - width))
    pos_y = clamp_float(pos_y, 0.0, max(0.0, 1.0 - height))
    return pos_x, pos_y, width, height, orientation
