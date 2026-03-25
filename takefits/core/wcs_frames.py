from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame

SUPPORTED_DISPLAY_FRAMES = ("native", "icrs", "fk5", "fk4", "galactic")

_FRAME_ALIASES = {
    "NATIVE": "native",
    "WCS": "native",
    "DEFAULT": "native",
    "ICRS": "icrs",
    "FK5": "fk5",
    "J2000": "fk5",
    "FK4": "fk4",
    "B1950": "fk4",
    "GAL": "galactic",
    "GALACTIC": "galactic",
}

_FRAME_LABELS = {
    "native": "Native (WCS)",
    "icrs": "ICRS",
    "fk5": "FK5 (J2000)",
    "fk4": "FK4 (B1950)",
    "galactic": "Galactic",
}


def normalize_display_frame(name: Optional[str]) -> str:
    if name is None:
        return "native"
    key = str(name).strip().upper()
    return _FRAME_ALIASES.get(key, "native")


def display_frame_label(name: Optional[str]) -> str:
    frame = normalize_display_frame(name)
    return _FRAME_LABELS.get(frame, _FRAME_LABELS["native"])


def axis_type_for_index(wcs, axis_index: int) -> str:
    if wcs is None:
        return ""
    try:
        ctype = getattr(getattr(wcs, "wcs", None), "ctype", None)
        if ctype is None or axis_index < 0 or axis_index >= len(ctype):
            return ""
        return str(ctype[axis_index] or "")
    except Exception:
        return ""


def _axis_token(axis_type: Optional[str]) -> str:
    return str(axis_type or "").strip().upper()


def axis_is_longitude(axis_type: Optional[str]) -> bool:
    token = _axis_token(axis_type)
    if token.startswith("RA"):
        return True
    return "LON" in token


def axis_is_latitude(axis_type: Optional[str]) -> bool:
    token = _axis_token(axis_type)
    if token.startswith("DEC"):
        return True
    return "LAT" in token


def native_celestial_frame(wcs) -> Optional[str]:
    if wcs is None:
        return None
    try:
        frame = wcs_to_celestial_frame(wcs)
    except Exception:
        return None
    if frame is None:
        return None
    name = getattr(frame, "name", None) or frame.__class__.__name__
    normalized = normalize_display_frame(name)
    if normalized == "native":
        return None
    return normalized


def celestial_axis_indices(wcs) -> Optional[Tuple[int, int]]:
    if wcs is None:
        return None
    try:
        naxis = int(getattr(wcs, "naxis", 0) or 0)
    except Exception:
        naxis = 0
    if naxis <= 0:
        return None
    lon_index = None
    lat_index = None
    for idx in range(naxis):
        axis_type = axis_type_for_index(wcs, idx)
        if lon_index is None and axis_is_longitude(axis_type):
            lon_index = idx
        if lat_index is None and axis_is_latitude(axis_type):
            lat_index = idx
    if lon_index is None or lat_index is None:
        return None
    return (lon_index, lat_index)


def frame_is_available(wcs, frame: Optional[str]) -> bool:
    normalized = normalize_display_frame(frame)
    if normalized == "native":
        return True
    if normalized not in SUPPORTED_DISPLAY_FRAMES:
        return False
    return celestial_axis_indices(wcs) is not None and native_celestial_frame(wcs) is not None


def available_display_frames(wcs) -> List[str]:
    preferred = preferred_display_frame(wcs)
    if preferred == "native":
        return ["native"]
    frames: List[str] = [preferred]
    for frame in ("icrs", "fk5", "fk4", "galactic"):
        if frame == preferred:
            continue
        frames.append(frame)
    return frames


def preferred_display_frame(wcs) -> str:
    frame = native_celestial_frame(wcs)
    if frame in {"icrs", "fk5", "fk4", "galactic"}:
        return frame
    return "native"


def plane_axis_indices(plane: Optional[str], wcs) -> Optional[Tuple[int, int]]:
    if wcs is None:
        return None
    mapping = {
        "xy": (0, 1),
        "xz": (0, 2),
        "zy": (2, 1),
    }
    key = str(plane or "xy").lower()
    indices = mapping.get(key)
    if indices is None:
        return None
    try:
        naxis = int(getattr(wcs, "naxis", 0) or 0)
    except Exception:
        naxis = 0
    if naxis <= max(indices):
        return None
    return indices


def display_axis_type(axis_type: Optional[str], frame: Optional[str]) -> str:
    source = str(axis_type or "")
    normalized = normalize_display_frame(frame)
    if normalized == "native":
        return source
    if axis_is_longitude(source):
        return "GLON" if normalized == "galactic" else "RA"
    if axis_is_latitude(source):
        return "GLAT" if normalized == "galactic" else "DEC"
    return source


def build_native_world_vector(wcs, fallback_native_world: Optional[Sequence[object]] = None) -> List[float]:
    try:
        naxis = int(getattr(wcs, "naxis", 0) or 0)
    except Exception:
        naxis = 0
    if naxis <= 0:
        return []

    vector: List[float] = [0.0] * naxis
    try:
        crval = list(getattr(getattr(wcs, "wcs", None), "crval", []) or [])
    except Exception:
        crval = []
    for idx in range(min(len(crval), naxis)):
        try:
            vector[idx] = float(crval[idx])
        except Exception:
            vector[idx] = 0.0

    if fallback_native_world is None:
        return vector

    for idx in range(min(len(fallback_native_world), naxis)):
        try:
            value = fallback_native_world[idx]
            if value is None:
                continue
            parsed = float(value)
            if parsed != parsed:
                continue
            vector[idx] = parsed
        except Exception:
            continue
    return vector


def _skycoord_from_lon_lat(lon_deg: float, lat_deg: float, frame_name: str) -> SkyCoord:
    return SkyCoord(lon_deg * u.deg, lat_deg * u.deg, frame=frame_name)


def transform_world_vector_between_frames(
    world_vector: Sequence[float],
    wcs,
    source_frame: Optional[str],
    target_frame: Optional[str],
) -> List[float]:
    return transform_world_vector_between_frames_with_status(
        world_vector,
        wcs,
        source_frame,
        target_frame,
    )[0]


def transform_world_vector_between_frames_with_status(
    world_vector: Sequence[float],
    wcs,
    source_frame: Optional[str],
    target_frame: Optional[str],
) -> Tuple[List[float], bool]:
    vector = [float(v) for v in world_vector]
    transformed = False
    if wcs is None:
        return vector, transformed

    source = normalize_display_frame(source_frame)
    target = normalize_display_frame(target_frame)
    if source == target:
        return vector, transformed

    celestial_axes = celestial_axis_indices(wcs)
    native_frame = native_celestial_frame(wcs)
    if celestial_axes is None or native_frame is None:
        return vector, transformed

    source_actual = native_frame if source == "native" else source
    target_actual = native_frame if target == "native" else target
    if source_actual == target_actual:
        return vector, transformed

    lon_index, lat_index = celestial_axes
    if lon_index >= len(vector) or lat_index >= len(vector):
        return vector, transformed

    try:
        sky = _skycoord_from_lon_lat(
            float(vector[lon_index]),
            float(vector[lat_index]),
            source_actual,
        )
        transformed = sky.transform_to(target_actual)
        vector[lon_index] = float(transformed.spherical.lon.deg)
        vector[lat_index] = float(transformed.spherical.lat.deg)
        transformed = True
    except Exception:
        return vector, False
    return vector, transformed


def parse_world_value(
    value: object,
    axis_type: Optional[str],
    *,
    frame_for_longitude: Optional[str] = None,
) -> float:
    def parse_compact_sexagesimal(text_value: str) -> Optional[float]:
        token = str(text_value).strip()
        if not token:
            return None

        sign = 1.0
        if token[0] in {"+", "-"}:
            if token[0] == "-":
                sign = -1.0
            token = token[1:]
        if not token:
            return None
        if token.count(".") > 1:
            return None
        if any(ch not in "0123456789." for ch in token):
            return None

        if "." in token:
            whole, frac = token.split(".", 1)
        else:
            whole, frac = token, ""
        if not whole:
            return None

        digits = len(whole)
        if digits < 4 or digits > 6:
            return None

        head_digits = whole[:-4] if digits > 4 else whole[:-2]
        head_value = float(int(head_digits)) if head_digits else 0.0

        if digits > 4:
            minute_text = whole[-4:-2]
            second_text = whole[-2:] + (f".{frac}" if frac else "")
            minute_value = float(int(minute_text))
            second_value = float(second_text)
        else:
            minute_text = whole[-2:] + (f".{frac}" if frac else "")
            minute_value = float(minute_text)
            second_value = 0.0

        if minute_value >= 60.0 or second_value >= 60.0:
            return None

        total = sign * (head_value + (minute_value / 60.0) + (second_value / 3600.0))
        normalized_frame = normalize_display_frame(frame_for_longitude)
        is_hourangle = axis_is_longitude(axis_type) and normalized_frame in {"icrs", "fk5", "fk4"}
        if is_hourangle:
            return total * 15.0
        return total

    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty world coordinate")

    has_sexagesimal = any(char in text for char in ("h", "m", "s", "d", ":", " "))
    if has_sexagesimal:
        normalized_frame = normalize_display_frame(frame_for_longitude)
        try:
            if axis_is_longitude(axis_type) and normalized_frame in {"icrs", "fk5", "fk4"}:
                return Angle(text, unit=u.hourangle).degree
            return Angle(text, unit=u.deg).degree
        except Exception:
            return Angle(text, unit=u.deg).degree
    if axis_is_longitude(axis_type) or axis_is_latitude(axis_type):
        compact_value = parse_compact_sexagesimal(text)
        if compact_value is not None:
            return compact_value
    return float(text)


def plane_values_for_display(
    wcs,
    plane: str,
    x_native: float,
    y_native: float,
    *,
    frame: Optional[str],
    fallback_native_world: Optional[Sequence[object]] = None,
    coord_wrap: object = 180,
) -> Tuple[float, float, str, str]:
    world_native = world_vector_for_plane_values(
        wcs,
        plane,
        x_native,
        y_native,
        fallback_native_world=fallback_native_world,
    )
    indices = plane_axis_indices(plane, wcs)
    if indices is None or world_native is None:
        return (float(x_native), float(y_native), "", "")
    world_display, transformed = transform_world_vector_between_frames_with_status(
        world_native,
        wcs,
        "native",
        frame,
    )

    axis_type_x_native = axis_type_for_index(wcs, indices[0])
    axis_type_y_native = axis_type_for_index(wcs, indices[1])
    normalized_frame = normalize_display_frame(frame)
    native_frame = normalize_display_frame(native_celestial_frame(wcs))
    if normalized_frame == "native":
        axis_type_x_display = axis_type_x_native
        axis_type_y_display = axis_type_y_native
    elif transformed or normalized_frame == native_frame:
        axis_type_x_display = display_axis_type(axis_type_x_native, frame)
        axis_type_y_display = display_axis_type(axis_type_y_native, frame)
    else:
        # If frame conversion failed, keep native axis semantics to avoid mislabeled values.
        axis_type_x_display = axis_type_x_native
        axis_type_y_display = axis_type_y_native

    x_value = float(world_display[indices[0]])
    y_value = float(world_display[indices[1]])
    if axis_is_longitude(axis_type_x_display):
        x_value = _wrap_display_longitude_value(x_value, axis_type_x_display, coord_wrap)
    if axis_is_longitude(axis_type_y_display):
        y_value = _wrap_display_longitude_value(y_value, axis_type_y_display, coord_wrap)
    return (x_value, y_value, axis_type_x_display, axis_type_y_display)


def world_vector_for_plane_values(
    wcs,
    plane: str,
    x_native: float,
    y_native: float,
    *,
    fallback_native_world: Optional[Sequence[object]] = None,
) -> Optional[List[float]]:
    indices = plane_axis_indices(plane, wcs)
    if indices is None:
        return None

    world_native = build_native_world_vector(wcs, fallback_native_world)
    if len(world_native) <= max(indices):
        return None
    world_native[indices[0]] = float(x_native)
    world_native[indices[1]] = float(y_native)
    return world_native


def plane_inputs_to_native(
    wcs,
    plane: str,
    x_input: object,
    y_input: object,
    *,
    frame: Optional[str],
    fallback_native_world: Optional[Sequence[object]] = None,
) -> Optional[Tuple[float, float]]:
    indices = plane_axis_indices(plane, wcs)
    if indices is None:
        return None

    normalized_frame = normalize_display_frame(frame)
    if not frame_is_available(wcs, normalized_frame):
        normalized_frame = "native"

    native_base = build_native_world_vector(wcs, fallback_native_world)
    if len(native_base) <= max(indices):
        return None
    source_world = transform_world_vector_between_frames(native_base, wcs, "native", normalized_frame)

    axis_type_x = display_axis_type(axis_type_for_index(wcs, indices[0]), normalized_frame)
    axis_type_y = display_axis_type(axis_type_for_index(wcs, indices[1]), normalized_frame)
    try:
        source_world[indices[0]] = parse_world_value(
            x_input,
            axis_type_x,
            frame_for_longitude=normalized_frame,
        )
        source_world[indices[1]] = parse_world_value(
            y_input,
            axis_type_y,
            frame_for_longitude=normalized_frame,
        )
    except Exception:
        return None

    native_world = transform_world_vector_between_frames(source_world, wcs, normalized_frame, "native")
    try:
        x_native = float(native_world[indices[0]])
        y_native = float(native_world[indices[1]])
    except Exception:
        return None
    if not (x_native == x_native and y_native == y_native):
        return None
    return (
        x_native,
        y_native,
    )


def axis_value_for_display(
    wcs,
    axis_index: int,
    native_value: float,
    *,
    frame: Optional[str],
    fallback_native_world: Optional[Sequence[object]] = None,
    coord_wrap: object = 180,
) -> Tuple[float, str]:
    world_native = build_native_world_vector(wcs, fallback_native_world)
    if axis_index < 0 or axis_index >= len(world_native):
        return (float(native_value), axis_type_for_index(wcs, axis_index))
    world_native[axis_index] = float(native_value)
    world_display, transformed = transform_world_vector_between_frames_with_status(
        world_native,
        wcs,
        "native",
        frame,
    )
    axis_type_native = axis_type_for_index(wcs, axis_index)
    normalized_frame = normalize_display_frame(frame)
    native_frame = normalize_display_frame(native_celestial_frame(wcs))
    if normalized_frame == "native":
        axis_type_display = axis_type_native
    elif transformed or normalized_frame == native_frame:
        axis_type_display = display_axis_type(axis_type_native, frame)
    else:
        axis_type_display = axis_type_native
    value = float(world_display[axis_index])
    if axis_is_longitude(axis_type_display):
        value = _wrap_display_longitude_value(value, axis_type_display, coord_wrap)
    return (value, axis_type_display)


def _wrap_display_longitude_value(value: float, axis_type: Optional[str], coord_wrap: object = 180) -> float:
    axis_upper = str(axis_type or "").upper()
    wrapped = float(value)
    if axis_upper.startswith("RA"):
        return wrapped % 360.0
    if not axis_is_longitude(axis_type):
        return wrapped
    try:
        wrap_mode = int(coord_wrap)
    except Exception:
        wrap_mode = 180
    if wrap_mode == 360:
        wrapped %= 360.0
        if wrapped < 0.0:
            wrapped += 360.0
        return wrapped
    wrapped = ((wrapped + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and float(value) > 0.0:
        return 180.0
    return wrapped
