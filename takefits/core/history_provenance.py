"""Common FITS HISTORY helpers for human-readable provenance lines."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from types import SimpleNamespace
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
    "export_moment_fits": "Compute Moment",
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
    "export_moment_fits": ["moment_type", "axis", "pixel_range", "world_range"],
    "compute_pv": [
        "x0", "y0", "x1", "y1", "start_world", "end_world", "width",
        "sample_spacing_pix", "weight_mode", "position_origin", "position_unit", "path_type",
        "center", "center_pix", "center_world", "semi_major_px", "semi_minor_px",
        "pa_rad", "pa_deg", "start_phi_rad", "end_phi_rad", "phi0_deg", "phi1_deg",
        "vertices", "vertices_world", "spline_type", "smoothness",
        "x_axis_mode", "sample_axis",
    ],
    "compute_cutout": ["pixel_bounds", "world_bounds"],
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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_fits_history_text(value: Any) -> str:
    """Return text that Astropy can store in a FITS HISTORY card."""
    ascii_text = str(value).encode("ascii", "replace").decode("ascii")
    return "".join(ch if " " <= ch <= "~" else "?" for ch in ascii_text)


def _coerce_record(record: Any) -> Any:
    if hasattr(record, "action") and hasattr(record, "params"):
        return record
    if isinstance(record, dict):
        return SimpleNamespace(
            action=str(record.get("action") or record.get("name") or ""),
            params=dict(record.get("params") or {}),
            timestamp=str(record.get("timestamp") or _utc_timestamp()),
        )
    return SimpleNamespace(action="", params={}, timestamp=_utc_timestamp())


def _resolve_state(target: Any) -> Any:
    session = _resolve_session(target)
    if session is not None:
        state = getattr(session, "state", None)
        if state is not None:
            return state
    owner = _resolve_owner(target)
    return getattr(owner, "app_state", None)


def _axis_label_from_ctype(ctype: Any, fallback: str) -> str:
    token = str(ctype or "").strip()
    if not token:
        return fallback
    base = token.split("-")[0].strip()
    return base or fallback


def _axis_label_from_target(target: Any, axis_index: int, fallback: Optional[str] = None) -> str:
    fallback_label = fallback or f"Axis {int(axis_index) + 1}"
    idx = int(axis_index)

    header = getattr(target, "header", None)
    if header is None:
        state = _resolve_state(target)
        header = getattr(state, "header", None)
    if header is not None:
        try:
            ctype = header.get(f"CTYPE{idx + 1}")
            if ctype:
                return _axis_label_from_ctype(ctype, fallback_label)
        except Exception:
            pass

    wcs = getattr(target, "wcs", None)
    if wcs is None:
        state = _resolve_state(target)
        wcs = getattr(state, "wcs", None)
    if wcs is not None and hasattr(wcs, "wcs"):
        try:
            ctype_values = list(getattr(wcs.wcs, "ctype", []) or [])
            if 0 <= idx < len(ctype_values):
                return _axis_label_from_ctype(ctype_values[idx], fallback_label)
        except Exception:
            pass

    return fallback_label


def _spectral_axis_label_from_target(target: Any) -> str:
    state = _resolve_state(target)
    spectral_meta = getattr(state, "spectral_metadata", {}) or {}
    axis_index = spectral_meta.get("axis_index")
    try:
        axis_idx = int(axis_index) - 1
        if axis_idx >= 0:
            return _axis_label_from_target(target, axis_idx, "Spectral")
    except Exception:
        pass

    wcs = getattr(target, "wcs", None)
    if wcs is None and state is not None:
        wcs = getattr(state, "wcs", None)
    if wcs is not None and hasattr(wcs, "wcs"):
        try:
            for idx, ctype in enumerate(list(getattr(wcs.wcs, "ctype", []) or [])):
                token = str(ctype or "").upper()
                if any(tag in token for tag in ("VRAD", "VELO", "VOPT", "FREQ", "WAVE")):
                    return _axis_label_from_ctype(ctype, "Spectral")
        except Exception:
            pass

    try:
        naxis = int(getattr(wcs, "naxis", 0) or 0)
    except Exception:
        naxis = 0
    fallback_axis = max(0, naxis - 1)
    return _axis_label_from_target(target, fallback_axis, "Spectral")


def _pv_world_axis_labels(target: Any) -> str:
    x_label = _axis_label_from_target(target, 0, "X")
    y_label = _axis_label_from_target(target, 1, "Y")
    return f"{x_label}/{y_label}"


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


def _format_path_value(value: Any) -> str:
    text = str(value or "")
    return text.replace("\\", "/").rstrip("/").split("/")[-1]


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
    else:
        keys = list(keys)

    if action_name in {"compute_moment", "export_moment_fits"} and "world_range" in params:
        keys = [key for key in keys if key != "pixel_range"]
    if action_name == "compute_cutout" and "world_bounds" in params:
        keys = [key for key in keys if key != "pixel_bounds"]
    if action_name == "compute_pv" and ("start_world" in params or "end_world" in params):
        keys = [key for key in keys if key not in {"x0", "y0", "x1", "y1"}]

    parts: List[str] = []
    for key in keys:
        if key not in params:
            continue
        value = params.get(key)
        if value is None:
            continue
        if key.endswith("_path"):
            value = _format_path_value(value)
        parts.append(f"{key}={_format_value(value)}")

    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"


def _resolve_pv_endpoint_values(
    params: Dict[str, Any],
    target: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    x0 = params.get("x0")
    y0 = params.get("y0")
    x1 = params.get("x1")
    y1 = params.get("y1")
    start_world = params.get("start_world")
    end_world = params.get("end_world")

    if None not in (x0, y0, x1, y1):
        return x0, y0, x1, y1, None, None, None, None

    if not (
        isinstance(start_world, (list, tuple))
        and len(start_world) >= 2
        and isinstance(end_world, (list, tuple))
        and len(end_world) >= 2
    ):
        return (
            x0 if x0 is not None else "?",
            y0 if y0 is not None else "?",
            x1 if x1 is not None else "?",
            y1 if y1 is not None else "?",
            None,
            None,
            None,
            None,
        )

    world0 = (start_world[0], start_world[1])
    world1 = (end_world[0], end_world[1])
    wcs = getattr(target, "wcs", None)
    if wcs is None:
        return "?", "?", "?", "?", world0[0], world0[1], world1[0], world1[1]

    try:
        from takefits.core.usecases.utils import get_axis_ctype, parse_world_coordinate

        def _parse_pair(pair: Any) -> list[float]:
            parsed = []
            for idx, value in enumerate(pair[:2]):
                ctype = get_axis_ctype(target, idx) if hasattr(target, "wcs") else ""
                if isinstance(value, str):
                    parsed.append(parse_world_coordinate(value, ctype))
                else:
                    parsed.append(float(value))
            return parsed

        w0_pair = _parse_pair(world0)
        w1_pair = _parse_pair(world1)
        w0_full = list(w0_pair) + [0.0] * max(0, int(getattr(wcs, "naxis", 2)) - 2)
        w1_full = list(w1_pair) + [0.0] * max(0, int(getattr(wcs, "naxis", 2)) - 2)
        pix0 = wcs.wcs_world2pix([w0_full], 0)[0]
        pix1 = wcs.wcs_world2pix([w1_full], 0)[0]
        return (
            float(pix0[0]),
            float(pix0[1]),
            float(pix1[0]),
            float(pix1[1]),
            float(w0_pair[0]),
            float(w0_pair[1]),
            float(w1_pair[0]),
            float(w1_pair[1]),
        )
    except Exception:
        return "?", "?", "?", "?", world0[0], world0[1], world1[0], world1[1]


def _coerce_point_pair(value: Any) -> Optional[tuple[Any, Any]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    return None


def _format_deg_param(
    params: Dict[str, Any],
    *,
    deg_key: str,
    rad_key: str,
) -> Optional[Any]:
    if params.get(deg_key) is not None:
        return params.get(deg_key)
    value = params.get(rad_key)
    if value is None:
        return None
    try:
        return f"{math.degrees(float(value)):.6g}"
    except Exception:
        return None


def _append_polyline_vertices_history(
    lines: List[str],
    params: Dict[str, Any],
    *,
    pv_world_labels: str,
) -> None:
    vertices = params.get("vertices")
    if isinstance(vertices, (list, tuple)) and vertices:
        lines.append(f"Polyline Vertices (pix): {len(vertices)}")
        for index, vertex in enumerate(vertices, start=1):
            point = _coerce_point_pair(vertex)
            if point is not None:
                lines.append(f"Polyline Vertex {index} (pix): ({point[0]}, {point[1]})")

    vertices_world = params.get("vertices_world")
    if isinstance(vertices_world, (list, tuple)) and vertices_world:
        lines.append(f"Polyline Vertices (world {pv_world_labels}): {len(vertices_world)}")
        for index, vertex in enumerate(vertices_world, start=1):
            point = _coerce_point_pair(vertex)
            if point is not None:
                lines.append(
                    f"Polyline Vertex {index} (world {pv_world_labels}): ({point[0]}, {point[1]})"
                )


def _build_verbose_history(action_name: str, params: Dict[str, Any], timestamp: Optional[str], target: Any) -> List[str]:
    """Build verbose history lines matching the historical GUI output formats."""
    lines: List[str] = []
    
    time_str = str(timestamp) if timestamp else "unknown time"
    # Some actions need the filename for "Source file:" logs
    filename = getattr(target, "filename", "unknown source")
    
    if action_name == "apply_smoothing":
        lines.append(f"Data smoothed using takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        ktype = params.get("kernel_type", "unknown")
        lines.append(f"Kernel: {ktype}")
        if ktype == "gaussian":
            sx = params.get("smoothness_x", 0)
            sy = params.get("smoothness_y", 0)
            sz = params.get("smoothness_z", 0)
            lines.append(f"Smoothness: X={sx}, Y={sy}, Z={sz}")
        elif ktype == "boxcar":
            w = params.get("width", 0)
            lines.append(f"Width: {w}")
    
    elif action_name == "apply_smoothing_to_resolution":
        lines.append(f"Data smoothed using takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        bmaj = params.get("target_bmaj")
        bmin = params.get("target_bmin")
        bpa = params.get("target_bpa")
        lines.append(f"Target resolution: BMAJ={bmaj}, BMIN={bmin}, BPA={bpa}")

    elif action_name == "apply_scaling":
        lines.append(f"Data scaled using takefits on {time_str}")
        factor = params.get("scale_factor", 1.0)
        lines.append(f"Manual scaling: multiplied by {factor}")

    elif action_name == "compute_arithmetic":
        lines.append(f"Arithmetic operation by takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        expr = params.get("expression", "")
        lines.append(f"Formula: {expr}")
        # GUI arithmetic panel also adds A: and B: lines. We'll simplify this slightly
        # since the session might not have the display name easily, but we can note paths.
        lines.append(f"A (source): {filename}")
        data_b = params.get("data_b_path")
        if data_b:
             lines.append(f"B: {str(data_b).split('/')[-1]}")

    elif action_name in ("apply_mask_threshold", "apply_mask_external", "apply_mask_moment_recipe"):
        # The GUI only adds "Data masked using takefits on {timestamp}" and "Source file: ..."
        # at the time of *saving*, and dumps all accumulated masking histories. 
        # Here we emit them per-action so they trace accurately.
        lines.append(f"Data masked using takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        if action_name == "apply_mask_threshold":
            cond = params.get("condition", "")
            thresh = params.get("threshold", 0)
            sym = "<" if cond == "less_than" else ">"
            lines.append(f"Threshold mask: value {sym} {thresh}")
        elif action_name == "apply_mask_external":
            m_path = str(params.get("mask_path", "")).split("/")[-1]
            lines.append(f"External mask applied: {m_path}")
        elif action_name == "apply_mask_moment_recipe":
            # Note: The GUI adds a long string of all params. We already have the
            # provenance summary string which covers this, but we can duplicate 
            # the specific GUI lines if necessary. The provenance string is cleaner.
            algo = params.get("algorithm", "")
            preset = params.get("preset", "")
            pol = params.get("polarity", "")
            lines.append(f"Moment mask algorithm={algo}")
            lines.append(f"Moment mask preset={preset}, polarity={pol}")

    elif action_name == "convert_intensity_unit":
        lines.append(f"Unit converted using takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        f_unit = params.get("from_unit", "")
        t_unit = params.get("to_unit", "")
        method = params.get("method", "")
        lines.append(f"Conversion: {f_unit} -> {t_unit}")
        lines.append(f"Method: {method}")

    elif action_name == "apply_baseline_subtraction":
        lines.append(f"Baseline subtracted using takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        order = params.get("order", "")
        lines.append(f"Polynomial order: {order}")
        wr = params.get("world_ranges", [])
        if wr:
             ranges_str = "; ".join([f"{r[0]} to {r[1]}" for r in wr if len(r) == 2])
             spectral_label = _spectral_axis_label_from_target(target)
             lines.append(f"{spectral_label} fitting windows: {ranges_str}")
             
    elif action_name == "compute_cutout":
        lines.append(f"Cutout generated by takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        # Resolve axis CTYPE names from WCS if available
        wcs = getattr(target, "wcs", None)
        def _axis_label(idx: int) -> str:
            if wcs is not None:
                try:
                    ctype = wcs.wcs.ctype[idx]
                    return ctype.split('-')[0].strip() or f"Axis {idx+1}"
                except Exception:
                    pass
            return f"Axis {idx+1}"
        pb = params.get("pixel_bounds")
        wb = params.get("world_bounds")
        if isinstance(pb, (list, tuple)):
            for i, bounds in enumerate(pb):
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    lines.append(f"{_axis_label(i)}: pixels {bounds[0]} to {bounds[1]}")
        elif isinstance(wb, (list, tuple)):
            for i, bounds in enumerate(wb):
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    lines.append(f"{_axis_label(i)}: {bounds[0]} to {bounds[1]}")

    elif action_name == "compute_moment":
        from takefits.core.usecases.moment import _moment_axis_name, _moment_range_history_line

        lines.append(f"Integration executed by takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        mtype = params.get("moment_type", "moment0")
        axis = params.get("axis", 0)
        mode_map = {
            "int": "Integration", "moment0": "Integration",
            "mom1": "Moment 1", "moment1": "Moment 1",
            "mom2": "Moment 2", "moment2": "Moment 2",
            "average": "Average", "peak_int": "Peak Intensity",
            "peak_corrd": "Peak Coordinate", "median_int": "Median",
            "rms": "RMS", "sigma": "Sigma (Std Dev)",
        }
        lines.append(f"Mode: {mode_map.get(mtype, mtype)}")
        try:
            axis_int = int(axis)
        except (TypeError, ValueError):
            axis_int = 0
        lines.append(f"Axis: {_moment_axis_name(axis_int)}")
        pr = params.get("pixel_range")
        wr = params.get("world_range")
        if wr:
            range_text = f"{_format_value(wr[0])} to {_format_value(wr[1])}"
            lines.append(_moment_range_history_line(target, axis_int, range_text))
        elif pr:
            range_text = f"ch {_format_value(pr[0])} to {_format_value(pr[1])}"
            lines.append(_moment_range_history_line(target, axis_int, range_text))
        clip = params.get("clip_threshold")
        if clip is not None:
            lines.append(f"Clipping: {clip}")

    elif action_name == "compute_pv":
        lines.append(f"PV Diagram created by takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        pv_world_labels = _pv_world_axis_labels(target)
        path_type = str(params.get("path_type") or "straight").strip().lower()
        if path_type in ("ellipse", "ellipse_arc"):
            lines.append("Path Type: ellipse arc" if path_type == "ellipse_arc" else "Path Type: ellipse")
            center_pix = params.get("center_pix") or params.get("center")
            if isinstance(center_pix, (list, tuple)) and len(center_pix) >= 2:
                lines.append(f"Ellipse Center (pix): ({center_pix[0]}, {center_pix[1]})")
            center_world = params.get("center_world")
            if isinstance(center_world, (list, tuple)) and len(center_world) >= 2:
                lines.append(f"Ellipse Center (world {pv_world_labels}): ({center_world[0]}, {center_world[1]})")
            for key, label in (
                ("semi_major_px", "Ellipse Semi-major (pix)"),
                ("semi_minor_px", "Ellipse Semi-minor (pix)"),
            ):
                value = params.get(key)
                if value is not None:
                    lines.append(f"{label}: {value}")
            for value, label in (
                (_format_deg_param(params, deg_key="pa_deg", rad_key="pa_rad"), "Ellipse PA (deg)"),
                (_format_deg_param(params, deg_key="phi0_deg", rad_key="start_phi_rad"), "Ellipse Phi0 (deg)"),
                (_format_deg_param(params, deg_key="phi1_deg", rad_key="end_phi_rad"), "Ellipse Phi1 (deg)"),
            ):
                if value is not None:
                    lines.append(f"{label}: {value}")
        elif path_type == "polyline":
            spline_type = str(params.get("spline_type") or "none").strip().lower()
            # Backward-compatible alias for the original boolean flag.
            if spline_type in ("", "none") and params.get("smooth"):
                spline_type = "catmull_rom"
            spline_labels = {"catmull_rom": "catmull-rom", "bspline": "b-spline"}
            if spline_type in spline_labels:
                try:
                    pct = int(round(float(params.get("smoothness", 1.0)) * 100))
                except (TypeError, ValueError):
                    pct = 100
                lines.append(
                    f"Path Type: polyline (spline: {spline_labels[spline_type]}, smoothness {pct}%)"
                )
            else:
                lines.append("Path Type: polyline")
            _append_polyline_vertices_history(lines, params, pv_world_labels=pv_world_labels)
        else:
            x0, y0, x1, y1, w0x, w0y, w1x, w1y = _resolve_pv_endpoint_values(params, target)
            lines.append(f"Slice Start (pix): ({x0}, {y0})")
            lines.append(f"Slice End (pix): ({x1}, {y1})")
            if None not in (w0x, w0y, w1x, w1y):
                lines.append(f"Slice Start (world {pv_world_labels}): ({w0x}, {w0y})")
                lines.append(f"Slice End (world {pv_world_labels}): ({w1x}, {w1y})")
            else:
                # Emit world-coordinate endpoints if WCS is available on target
                wcs = getattr(target, "wcs", None)
                if wcs is not None:
                    try:
                        _x0 = float(x0) if x0 != "?" else None
                        _y0 = float(y0) if y0 != "?" else None
                        _x1 = float(x1) if x1 != "?" else None
                        _y1 = float(y1) if y1 != "?" else None
                        if None not in (_x0, _y0, _x1, _y1):
                            npad = max(0, wcs.naxis - 2)
                            w0 = wcs.wcs_pix2world([[_x0, _y0] + [0] * npad], 0)[0]
                            w1 = wcs.wcs_pix2world([[_x1, _y1] + [0] * npad], 0)[0]
                            lines.append(f"Slice Start (world {pv_world_labels}): ({w0[0]}, {w0[1]})")
                            lines.append(f"Slice End (world {pv_world_labels}): ({w1[0]}, {w1[1]})")
                    except Exception:
                        pass
        width = params.get("width", 0)
        if width:
            lines.append(f"Slice Width (pix): {width}")
        sample_spacing = params.get("sample_spacing_pix")
        if sample_spacing is not None:
            lines.append(f"Sample Spacing (pix): {sample_spacing}")
        sample_axis = params.get("sample_axis")
        if sample_axis is not None:
            lines.append(f"Sample Axis: {sample_axis}")
        wm = params.get("weight_mode", 0)
        interp = "Gaussian" if wm == 1 else "Bilinear"
        lines.append(f"Interpolation Mode: {interp}")
        position_origin = str(params.get("position_origin") or "start").strip().lower()
        if position_origin in ("center", "centre", "middle"):
            lines.append("Position Origin: center")
        position_unit = params.get("position_unit")
        if position_unit is not None:
            lines.append(f"Position Unit: {position_unit}")
        x_axis_mode = params.get("x_axis_mode")
        if x_axis_mode is not None:
            lines.append(f"X Axis Mode: {x_axis_mode}")

    elif action_name in ("run_clumpfind", "run_fellwalker", "run_dendrogram"):
        algo_map = {
            "run_clumpfind": "ClumpFind",
            "run_fellwalker": "FellWalker",
            "run_dendrogram": "Dendrogram",
        }
        algo = algo_map.get(action_name, action_name)
        lines.append(f"Clump Finding Mask generated by takefits on {time_str}")
        lines.append(f"Source file: {filename}")
        lines.append(f"Algorithm: {algo}")
        for k, v in params.items():
            if k not in ("state",):
                lines.append(f"  {k}: {v}")

    return lines


def build_processing_history_lines_from_records(
    target: Any,
    records: Iterable[Any],
    max_entries: Optional[int] = None,
) -> List[str]:
    if max_entries is not None and max_entries <= 0:
        return []

    lines: List[str] = []

    # We will build up a list of history chunks, each consisting of:
    # 1. An action-specific "header" or "verbose description" (if applicable)
    # 2. The standard "Processing history: ..." provenance line

    for raw_record in records:
        record = _coerce_record(raw_record)
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
        provenance_line = f"Processing history: {label}{summary}"

        # Try to build verbose GUI-like history strings
        timestamp = getattr(record, "timestamp", None)
        verbose_lines = _build_verbose_history(action_name, params, timestamp, target)
        if verbose_lines:
            lines.extend(verbose_lines)

        lines.append(provenance_line)

    if not lines:
        return []
    safe_lines = [sanitize_fits_history_text(line) for line in lines]
    if max_entries is None:
        return safe_lines
    return safe_lines[-max_entries:]


def build_processing_history_lines_with_action(
    target: Any,
    action_name: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timestamp: Optional[str] = None,
    max_entries: Optional[int] = None,
) -> List[str]:
    records = list(_iter_records(target))
    records.append(
        {
            "action": str(action_name or ""),
            "params": dict(params or {}),
            "timestamp": str(timestamp or _utc_timestamp()),
        }
    )
    return build_processing_history_lines_from_records(target, records, max_entries=max_entries)


def build_processing_history_lines(target: Any, max_entries: Optional[int] = None) -> List[str]:
    return build_processing_history_lines_from_records(
        target,
        _iter_records(target),
        max_entries=max_entries,
    )
