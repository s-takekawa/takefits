"""Helpers for persisting Range Control Panel state."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from takefits.core.wcs_frames import build_native_world_vector
from takefits.core.workspace_restore import normalize_wcs_axis_type, normalize_wcs_unit

RANGE_FILE_TYPE = "takefits.range"
RANGE_FILE_VERSION = 2
_RANGE_AXES = ("x", "y", "z")
_AXIS_INDEX_TO_KEY = {0: "x", 1: "y", 2: "z", 3: "s"}
_FAMILY_LABELS = {
    "equatorial": "Equatorial",
    "galactic": "Galactic",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def empty_range_payload() -> Dict[str, Any]:
    return {
        "type": RANGE_FILE_TYPE,
        "version": RANGE_FILE_VERSION,
        "saved_at": "",
        "source": {},
        "ranges": {},
    }


def _normalize_axis_entry(axis_entry: Any) -> Dict[str, Any]:
    if not isinstance(axis_entry, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key in ("min_text", "max_text", "ctype"):
        value = axis_entry.get(key)
        if value not in (None, ""):
            normalized[key] = str(value)
    for key in ("native_min", "native_max"):
        value = axis_entry.get(key)
        if value in (None, ""):
            continue
        normalized[key] = float(value)
    return normalized


def _normalize_source(source: Any) -> Dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    normalized: Dict[str, Any] = {}
    filepath = str(source.get("filepath") or "").strip()
    filename = str(source.get("filename") or "").strip()
    if filepath:
        normalized["filepath"] = filepath
    if filename:
        normalized["filename"] = filename
    signature = source.get("wcs_signature")
    if isinstance(signature, dict):
        normalized["wcs_signature"] = dict(signature)
    return normalized


def normalize_range_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return empty_range_payload()

    # Backward-compatible path for the short-lived entries-array format.
    if payload.get("type") == RANGE_FILE_TYPE and isinstance(payload.get("entries"), list):
        entries = [entry for entry in payload.get("entries") or [] if isinstance(entry, dict)]
        payload = entries[-1] if entries else {}

    normalized = empty_range_payload()
    if payload.get("type") == RANGE_FILE_TYPE:
        normalized["type"] = RANGE_FILE_TYPE
    normalized["version"] = int(payload.get("version", RANGE_FILE_VERSION) or RANGE_FILE_VERSION)
    normalized["saved_at"] = str(payload.get("saved_at") or "")
    normalized["source"] = _normalize_source(payload.get("source"))

    ranges = payload.get("ranges")
    if isinstance(ranges, dict):
        for axis_key in _RANGE_AXES:
            axis_entry = _normalize_axis_entry(ranges.get(axis_key))
            if axis_entry:
                normalized["ranges"][axis_key] = axis_entry
    return normalized


def build_range_payload(
    *,
    source: Dict[str, Any],
    ranges: Dict[str, Any],
    saved_at: str | None = None,
) -> Dict[str, Any]:
    payload = empty_range_payload()
    payload["saved_at"] = str(saved_at or utc_timestamp())
    payload["source"] = _normalize_source(source)
    for axis_key in _RANGE_AXES:
        axis_entry = _normalize_axis_entry((ranges or {}).get(axis_key))
        if axis_entry:
            payload["ranges"][axis_key] = axis_entry
    return payload


def _legacy_payload_from_text(text: str) -> Dict[str, Any]:
    meta = {"filename": ""}
    ranges: Dict[str, Dict[str, Any]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            lower = stripped.lower()
            if lower.startswith("#filename"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    meta["filename"] = parts[1].strip()
            continue
        parts = stripped.split()
        if len(parts) != 4:
            raise ValueError(f'Invalid legacy range line: "{stripped}"')
        axis_key = parts[0].strip().lower()
        if axis_key not in _RANGE_AXES:
            continue
        try:
            native_min = float(parts[2])
            native_max = float(parts[3])
        except ValueError as exc:
            raise ValueError(f'Invalid numeric values in legacy range line: "{stripped}"') from exc
        ranges[axis_key] = {
            "ctype": str(parts[1] or ""),
            "min_text": str(parts[2]),
            "max_text": str(parts[3]),
            "native_min": native_min,
            "native_max": native_max,
        }
    if "x" not in ranges or "y" not in ranges:
        raise ValueError("Legacy range file must contain X and Y entries.")
    return build_range_payload(source={"filename": meta["filename"]}, ranges=ranges, saved_at="")


def load_range_payload(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return empty_range_payload()
    if stripped.startswith("{"):
        return normalize_range_payload(json.loads(stripped))
    return _legacy_payload_from_text(text)


def save_range_payload(path: str, payload: Dict[str, Any]) -> None:
    normalized = normalize_range_payload(payload)
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _signature_axes(signature: Dict[str, Any] | Any) -> List[Dict[str, Any]]:
    if not isinstance(signature, dict):
        return []
    axes = signature.get("axes")
    if not isinstance(axes, list):
        return []
    return [axis for axis in axes if isinstance(axis, dict)]


def _axis_type(axis_entry: Dict[str, Any] | Any) -> str:
    if not isinstance(axis_entry, dict):
        return ""
    token = str(axis_entry.get("axis_type") or "").strip().upper()
    if token:
        return token
    return normalize_wcs_axis_type(axis_entry.get("ctype"))


def _axis_unit(axis_entry: Dict[str, Any] | Any) -> str:
    if not isinstance(axis_entry, dict):
        return ""
    return normalize_wcs_unit(axis_entry.get("unit") or axis_entry.get("cunit"))


def _payload_source(payload: Dict[str, Any] | Any) -> Dict[str, Any]:
    normalized = normalize_range_payload(payload)
    source = normalized.get("source")
    return source if isinstance(source, dict) else {}


def _payload_signature(payload: Dict[str, Any] | Any) -> Dict[str, Any] | Any:
    return _payload_source(payload).get("wcs_signature")


def describe_wcs_signature(signature: Dict[str, Any] | Any) -> str:
    axes = _signature_axes(signature)
    axis_types = [_axis_type(axis) for axis in axes]

    family = str((signature or {}).get("celestial_family") or "").strip().lower() if isinstance(signature, dict) else ""
    if not family and len(axis_types) >= 2:
        if axis_types[0] == "RA" and axis_types[1] == "DEC":
            family = "equatorial"
        elif axis_types[0] == "GLON" and axis_types[1] == "GLAT":
            family = "galactic"

    label = _FAMILY_LABELS.get(family, family.replace("_", " ").title() if family else "")
    details = []

    spatial_axes = [axis_type for axis_type in axis_types[:2] if axis_type]
    if spatial_axes:
        details.append("/".join(spatial_axes))

    spectral_axis = axis_types[2] if len(axis_types) > 2 else ""
    if spectral_axis:
        details.append(spectral_axis)

    if label and details:
        return f'{label} ({", ".join(details)})'
    if label:
        return label
    if details:
        return ", ".join(details)
    return "Unknown"


def describe_range_payload_coordinates(payload: Dict[str, Any] | Any) -> str:
    return describe_wcs_signature(_payload_signature(payload))


def build_coordinate_mismatch_message(
    payload: Dict[str, Any] | Any,
    current_signature: Dict[str, Any] | Any,
    *,
    prefix: str = "The saved range uses a different coordinate system. Load was skipped.",
) -> str:
    saved_coords = describe_range_payload_coordinates(payload)
    current_coords = describe_wcs_signature(current_signature)
    details = []
    if saved_coords:
        details.append(f"Saved range: {saved_coords}")
    if current_coords:
        details.append(f"Current FITS: {current_coords}")
    if not details:
        return prefix
    return f'{prefix}\n\n' + "\n".join(details)


def evaluate_range_payload_compatibility(
    payload: Dict[str, Any] | Any,
    *,
    current_signature: Dict[str, Any] | Any,
    data_ndim: int,
) -> Tuple[bool, str]:
    saved_signature = _payload_signature(payload)
    if not isinstance(saved_signature, dict):
        return True, "missing_saved_signature"
    if not isinstance(current_signature, dict):
        return False, "missing_current_signature"

    saved_axes = _signature_axes(saved_signature)
    current_axes = _signature_axes(current_signature)
    if len(saved_axes) < 2 or len(current_axes) < 2:
        return False, "insufficient_spatial_axes"

    saved_family = str(saved_signature.get("celestial_family") or "").strip().lower()
    current_family = str(current_signature.get("celestial_family") or "").strip().lower()
    if saved_family and current_family and saved_family != current_family:
        return False, "celestial_family_mismatch"

    for axis_idx in (0, 1):
        saved_unit = _axis_unit(saved_axes[axis_idx])
        current_unit = _axis_unit(current_axes[axis_idx])
        if saved_unit and current_unit and saved_unit != current_unit:
            return False, f"spatial_axis_unit_mismatch:{axis_idx}"

    if int(data_ndim or 0) > 2 and len(saved_axes) > 2 and len(current_axes) > 2:
        saved_spec = _axis_type(saved_axes[2])
        current_spec = _axis_type(current_axes[2])
        if saved_spec and current_spec and saved_spec != current_spec:
            return False, "spectral_axis_type_mismatch"
        saved_unit = _axis_unit(saved_axes[2])
        current_unit = _axis_unit(current_axes[2])
        if saved_unit and current_unit and saved_unit != current_unit:
            return False, "spectral_axis_unit_mismatch"

    return True, "compatible"


def extract_native_ranges(payload: Dict[str, Any] | Any) -> Dict[str, Tuple[float, float]]:
    normalized = normalize_range_payload(payload)
    ranges = normalized.get("ranges") if isinstance(normalized.get("ranges"), dict) else {}
    resolved: Dict[str, Tuple[float, float]] = {}
    for axis_key in _RANGE_AXES:
        axis_entry = ranges.get(axis_key)
        if not isinstance(axis_entry, dict):
            continue
        if "native_min" not in axis_entry or "native_max" not in axis_entry:
            continue
        resolved[axis_key] = (
            float(axis_entry["native_min"]),
            float(axis_entry["native_max"]),
        )
    return resolved


def native_ranges_to_pixel_limits(
    wcs,
    native_ranges: Dict[str, Tuple[float, float]] | Any,
    *,
    fallback_native_world: Sequence[object] | None = None,
) -> Dict[str, Tuple[float, float]]:
    if wcs is None:
        raise ValueError("WCS is not available for range conversion.")
    naxis = int(getattr(wcs, "naxis", 0) or 0)
    if naxis <= 0:
        raise ValueError("WCS axis information is unavailable.")

    base_world = build_native_world_vector(wcs, fallback_native_world)
    bounds = []
    for axis_index in range(naxis):
        axis_key = _AXIS_INDEX_TO_KEY.get(axis_index)
        if axis_key in native_ranges:
            lo, hi = native_ranges[axis_key]
            bounds.append((float(lo), float(hi)))
            continue
        fallback = 0.0
        if axis_index < len(base_world):
            try:
                fallback = float(base_world[axis_index])
            except Exception:
                fallback = 0.0
        bounds.append((fallback, fallback))

    corner_axes = [(lo,) if lo == hi else (lo, hi) for lo, hi in bounds]
    corners = [list(corner) for corner in product(*corner_axes)]
    try:
        pixel_values = wcs.wcs_world2pix(corners, 0)
    except Exception as exc:
        raise ValueError("Could not convert saved world ranges to pixels.") from exc
    if len(pixel_values) == 0:
        raise ValueError("No pixel coordinates were produced from the saved ranges.")

    pixel_limits: Dict[str, Tuple[float, float]] = {}
    for axis_index in range(naxis):
        axis_key = _AXIS_INDEX_TO_KEY.get(axis_index)
        if axis_key not in native_ranges:
            continue
        values = []
        for row in pixel_values:
            try:
                values.append(float(row[axis_index]))
            except Exception:
                continue
        if not values:
            continue
        lo = min(values)
        hi = max(values)
        if lo > hi:
            lo, hi = hi, lo
        pixel_limits[axis_key] = (lo, hi)
    return pixel_limits
