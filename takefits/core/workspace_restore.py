"""Workspace restore compatibility helpers (PyQt-free)."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple


def normalize_wcs_axis_type(ctype: Any) -> str:
    token = str(ctype or "").strip().upper()
    if not token:
        return ""
    if token.startswith("RA"):
        return "RA"
    if "DEC" in token:
        return "DEC"
    if "GLON" in token:
        return "GLON"
    if "GLAT" in token:
        return "GLAT"
    if any(key in token for key in ("VRAD", "VELO", "VOPT")):
        return "VELO"
    if "FREQ" in token:
        return "FREQ"
    if "WAVE" in token:
        return "WAVE"
    return token.split("-")[0]


def normalize_wcs_unit(unit: Any) -> str:
    return str(unit or "").strip().lower().replace(" ", "")


def _as_shape_list(shape: Iterable[Any]) -> list[int]:
    try:
        return [int(v) for v in list(shape or [])]
    except Exception:
        return []


def _axis_type(axis_entry: Any) -> str:
    if not isinstance(axis_entry, dict):
        return ""
    token = str(axis_entry.get("axis_type") or "").strip().upper()
    if token:
        return token
    return normalize_wcs_axis_type(axis_entry.get("ctype"))


def _axis_unit(axis_entry: Any) -> str:
    if not isinstance(axis_entry, dict):
        return ""
    return normalize_wcs_unit(axis_entry.get("unit") or axis_entry.get("cunit"))


def evaluate_workspace_dataset_match(
    source: Dict[str, Any] | Any,
    *,
    current_path: str,
    current_shape: Iterable[Any],
) -> Tuple[bool, str]:
    if not isinstance(source, dict):
        return False, "invalid_source"

    saved_path = str(source.get("filepath") or "")
    active_path = str(current_path or "")
    if not saved_path:
        return False, "missing_saved_filepath"
    if not active_path:
        return False, "missing_current_filepath"

    path_match = False
    try:
        path_match = os.path.realpath(saved_path) == os.path.realpath(active_path)
    except Exception:
        path_match = False
    if not path_match:
        try:
            path_match = os.path.samefile(saved_path, active_path)
        except Exception:
            path_match = False
    if not path_match:
        return False, "filepath_mismatch"

    saved_shape = source.get("data_shape")
    if isinstance(saved_shape, list) and saved_shape:
        current_shape_list = _as_shape_list(current_shape)
        saved_shape_list = _as_shape_list(saved_shape)
        if current_shape_list and saved_shape_list and current_shape_list != saved_shape_list:
            return False, "shape_mismatch"
    return True, "match"


def evaluate_workspace_wcs_compatibility(
    source: Dict[str, Any] | Any,
    *,
    current_signature: Dict[str, Any] | Any,
    data_ndim: int,
) -> Tuple[bool, str]:
    if not isinstance(source, dict):
        return False, "invalid_source"
    saved_signature = source.get("wcs_signature")
    if not isinstance(saved_signature, dict):
        return False, "missing_saved_signature"
    if not isinstance(current_signature, dict):
        return False, "missing_current_signature"

    saved_axes = list(saved_signature.get("axes") or [])
    current_axes = list(current_signature.get("axes") or [])
    if len(saved_axes) < 2 or len(current_axes) < 2:
        return False, "insufficient_spatial_axes"

    for axis_idx in (0, 1):
        saved_type = _axis_type(saved_axes[axis_idx])
        current_type = _axis_type(current_axes[axis_idx])
        if saved_type != current_type:
            return False, f"spatial_axis_type_mismatch:{axis_idx}"
        saved_unit = _axis_unit(saved_axes[axis_idx])
        current_unit = _axis_unit(current_axes[axis_idx])
        if saved_unit and current_unit and saved_unit != current_unit:
            return False, f"spatial_axis_unit_mismatch:{axis_idx}"

    saved_family = str(saved_signature.get("celestial_family") or "").strip().lower()
    current_family = str(current_signature.get("celestial_family") or "").strip().lower()
    if saved_family and current_family and saved_family != current_family:
        return False, "celestial_family_mismatch"

    if int(data_ndim or 0) > 2 and len(saved_axes) > 2 and len(current_axes) > 2:
        saved_type = _axis_type(saved_axes[2])
        current_type = _axis_type(current_axes[2])
        if saved_type != current_type:
            return False, "spectral_axis_type_mismatch"
        saved_unit = _axis_unit(saved_axes[2])
        current_unit = _axis_unit(current_axes[2])
        if saved_unit and current_unit and saved_unit != current_unit:
            return False, "spectral_axis_unit_mismatch"
    return True, "compatible"


def build_workspace_restore_diagnostics(
    payload: Dict[str, Any] | Any,
    *,
    current_path: str,
    current_shape: Iterable[Any],
    current_signature: Dict[str, Any] | Any,
    data_ndim: int,
) -> Dict[str, Any]:
    source: Dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("source"), dict):
        source = dict(payload.get("source") or {})

    same_dataset, dataset_reason = evaluate_workspace_dataset_match(
        source,
        current_path=current_path,
        current_shape=current_shape,
    )
    wcs_compatible, wcs_reason = evaluate_workspace_wcs_compatibility(
        source,
        current_signature=current_signature,
        data_ndim=data_ndim,
    )
    source_path = str(source.get("filepath") or "")
    active_path = str(current_path or "")
    return {
        "same_dataset": bool(same_dataset),
        "dataset_reason": str(dataset_reason),
        "wcs_compatible": bool(wcs_compatible),
        "wcs_reason": str(wcs_reason),
        "has_wcs_signature": isinstance(source.get("wcs_signature"), dict),
        "source_filepath": source_path,
        "current_filepath": active_path,
    }


def compute_range_restore_mode(
    *,
    same_dataset: bool,
    wcs_compatible: bool,
    has_wcs_signature: bool,
) -> str:
    """
    Determine how range restoration should run.

    Returns:
        "view_then_world": same dataset (try view limits, fallback to world ranges)
        "world_only": different dataset but compatible/unknown signature
        "skip": incompatible WCS
    """
    if bool(same_dataset):
        return "view_then_world"
    if bool(wcs_compatible) or not bool(has_wcs_signature):
        return "world_only"
    return "skip"


def build_workspace_restore_status_line(
    *,
    same_dataset: bool,
    wcs_compatible: bool,
    has_wcs_signature: bool,
    restored_view: bool,
    restored_world: bool,
) -> str:
    if bool(same_dataset):
        if bool(restored_view):
            return "Panel visibility, geometry, and full workspace state restored."
        if bool(restored_world):
            return "Panel visibility/geometry restored; world-coordinate ranges restored."
        return "Panel visibility/geometry restored; no saved ranges applied."

    if (bool(wcs_compatible) or not bool(has_wcs_signature)) and bool(restored_world):
        if bool(has_wcs_signature):
            return (
                "Panel visibility/geometry restored. Different FITS with compatible WCS; "
                "world-coordinate ranges restored."
            )
        return (
            "Panel visibility/geometry restored. Different FITS; WCS signature unavailable, "
            "attempted world-range restore."
        )
    if bool(wcs_compatible) or not bool(has_wcs_signature):
        if bool(has_wcs_signature):
            return (
                "Panel visibility/geometry restored. Different FITS with compatible WCS; "
                "no saved ranges applied."
            )
        return (
            "Panel visibility/geometry restored. Different FITS; WCS signature unavailable, "
            "no saved ranges applied."
        )
    return "Panel visibility/geometry restored. Different FITS with incompatible WCS; range restore skipped."


__all__ = [
    "normalize_wcs_axis_type",
    "normalize_wcs_unit",
    "evaluate_workspace_dataset_match",
    "evaluate_workspace_wcs_compatibility",
    "build_workspace_restore_diagnostics",
    "compute_range_restore_mode",
    "build_workspace_restore_status_line",
]
