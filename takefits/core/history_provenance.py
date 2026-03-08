"""Common FITS HISTORY helpers for human-readable provenance lines."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


_ACTION_LABELS: Dict[str, str] = {
    "apply_mask_threshold": "Apply Mask Threshold",
    "apply_mask_external": "Apply External Mask",
    "apply_mask_moment_recipe": "Apply Moment Mask",
    "apply_smoothing": "Apply Smoothing",
    "apply_smoothing_to_resolution": "Apply Smoothing to Resolution",
    "apply_scaling": "Apply Scaling",
    "apply_baseline_subtraction": "Apply Baseline Subtraction",
    "convert_intensity_unit": "Convert Intensity Unit",
    "compute_arithmetic": "Compute Arithmetic",
    "compute_regrid": "Compute Regrid",
    "compute_moment": "Compute Moment",
    "compute_pv": "Compute PV",
    "compute_cutout": "Compute Cutout",
    "compute_channel_map": "Compute Channel Map",
    "run_clumpfind": "Run ClumpFind",
    "run_fellwalker": "Run FellWalker",
    "run_dendrogram": "Run Dendrogram",
}


_ACTION_PARAM_KEYS: Dict[str, List[str]] = {
    "apply_mask_threshold": ["threshold", "condition"],
    "apply_mask_external": ["mask_path"],
    "apply_mask_moment_recipe": [
        "algorithm",
        "preset",
        "polarity",
        "seed_sigma",
        "grow_sigma",
        "clip_sigma",
        "connectivity",
    ],
    "apply_smoothing": [
        "kernel_type",
        "smoothness_x",
        "smoothness_y",
        "smoothness_z",
    ],
    "apply_smoothing_to_resolution": [
        "target_bmaj",
        "target_bmin",
        "target_bpa",
    ],
    "apply_scaling": ["scale_factor"],
    "apply_baseline_subtraction": ["order", "world_ranges"],
    "convert_intensity_unit": ["from_unit", "to_unit", "method"],
    "compute_arithmetic": ["operation", "expression", "data_b_path"],
    "compute_regrid": ["mode", "target_system", "template_path", "interpolation"],
    "compute_moment": ["moment_type", "axis", "clip_threshold", "pixel_range", "world_range"],
    "compute_pv": ["x0", "y0", "x1", "y1", "width", "weight_mode"],
    "compute_cutout": ["pixel_bounds"],
    "compute_channel_map": ["axis", "start_channel", "channel_width", "num_channels"],
    "run_clumpfind": ["rms", "min_threshold_sigma", "step_sigma", "min_pixels"],
    "run_fellwalker": ["rms", "min_threshold_sigma", "min_dip_sigma", "min_pixels"],
    "run_dendrogram": ["rms", "min_value_sigma", "min_delta_sigma", "min_npix", "use_scimes"],
}


def _resolve_owner(target: Any) -> Any:
    return getattr(target, "main_window", None) or target


def _resolve_session(target: Any) -> Any:
    owner = _resolve_owner(target)
    return getattr(owner, "action_session", None)


def _iter_records(target: Any) -> Iterable[Any]:
    session = _resolve_session(target)
    if session is None:
        return ()
    try:
        cursor = int(getattr(session, "cursor", len(getattr(session, "history", []) or [])))
        history = list(getattr(session, "history", []) or [])
        return history[: max(0, cursor)]
    except Exception:
        return ()


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return f"{_format_value(value[0])} to {_format_value(value[1])}"
        return "[" + ", ".join(_format_value(item) for item in value[:6]) + ("]" if len(value) <= 6 else ", ...]")
    return str(value)


def _extract_action_params(action_name: str, raw_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(raw_params or {})
    if action_name == "compute_regrid":
        nested = params.get("params")
        if isinstance(nested, dict):
            return dict(nested)
    return params


def _summarize_action_params(action_name: str, params: Dict[str, Any]) -> str:
    keys = _ACTION_PARAM_KEYS.get(action_name)
    if not keys:
        keys = sorted(params.keys())[:5]

    parts: List[str] = []
    for key in keys:
        if key not in params:
            continue
        value = params.get(key)
        if value is None:
            continue
        if key == "mask_path":
            value = str(value).split("/")[-1]
        parts.append(f"{key}={_format_value(value)}")

    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"


def build_processing_history_lines(target: Any, max_entries: Optional[int] = None) -> List[str]:
    if max_entries is not None and max_entries <= 0:
        return []

    lines: List[str] = []
    for record in _iter_records(target):
        action_name = str(getattr(record, "action", "") or "").strip()
        if not action_name:
            continue
        label = _ACTION_LABELS.get(action_name)
        if label is None:
            continue
        raw_params = getattr(record, "params", {}) or {}
        if not isinstance(raw_params, dict):
            raw_params = {}
        params = _extract_action_params(action_name, raw_params)
        summary = _summarize_action_params(action_name, params)
        lines.append(f"Processing history: {label}{summary}")

    if not lines:
        return []
    if max_entries is None:
        return lines
    return lines[-max_entries:]
