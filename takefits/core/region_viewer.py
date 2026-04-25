"""Viewer helpers shared by region tools and tests.

These functions intentionally avoid Qt imports so they stay usable in
headless test environments.
"""


def viewer_display_slices(viewer):
    displaymap = getattr(viewer, "displaymap", None)
    if displaymap is not None:
        slices = getattr(displaymap, "slices", None)
        if slices:
            return slices

    for attr in ("integ_slice", "projection_slices", "slice"):
        slices = getattr(viewer, attr, None)
        if slices and not callable(slices):
            return slices

    format_pix = getattr(viewer, "format_pix", None)
    if format_pix is not None:
        return getattr(format_pix, "slices", None)

    return None


def resolve_plane_pixel_limits(viewer) -> tuple[float, float]:
    """Return full-resolution pixel limits for the viewer's active plane."""
    width_limit = None
    height_limit = None

    slices = viewer_display_slices(viewer)

    header = getattr(viewer, "header", None)
    if header is not None and slices:
        for idx, entry in enumerate(slices):
            if entry not in {"x", "y"}:
                continue
            key = f"NAXIS{idx + 1}"
            try:
                axis_size = float(header[key])
            except Exception:
                continue
            if entry == "x":
                width_limit = axis_size
            else:
                height_limit = axis_size

    if width_limit is None or height_limit is None:
        data = getattr(viewer, "data", None)
        shape = getattr(data, "shape", None)
        plane = str(getattr(viewer, "plane", "xy") or "xy").lower()
        if shape and len(shape) >= 2:
            if plane == "xy":
                height_limit = float(shape[-2])
                width_limit = float(shape[-1])
            elif plane == "xz" and len(shape) >= 3:
                height_limit = float(shape[-3])
                width_limit = float(shape[-1])
            elif plane == "zy" and len(shape) >= 3:
                height_limit = float(shape[-2])
                width_limit = float(shape[-3])

    if width_limit is None or height_limit is None:
        array = getattr(getattr(viewer, "im", None), "get_array", lambda: None)()
        if array is not None:
            try:
                height_limit = float(array.shape[0])
                width_limit = float(array.shape[1])
            except Exception:
                width_limit = None
                height_limit = None

    if width_limit is None or height_limit is None:
        return (1000.0, 1000.0)
    return (float(width_limit), float(height_limit))


def shared_index(viewer, getter_name: str, upper_bound: int) -> int:
    getter = getattr(viewer, getter_name, None)
    try:
        value = getter() if callable(getter) else 0
        index = int(round(float(value)))
    except Exception:
        index = 0
    if upper_bound <= 0:
        return 0
    return max(0, min(index, upper_bound - 1))


def resolve_region_analysis_array(viewer, *, is_cube: bool):
    """Return the full-resolution array used for stats/moments."""
    plane = str(getattr(viewer, "plane", "xy") or "xy").lower()

    if not is_cube:
        projected = getattr(viewer, "integrated_data", None)
        if getattr(projected, "ndim", 0) == 2:
            return projected.T if plane == "zy" else projected

        app_state = None
        get_app_state = getattr(viewer, "get_app_state", None)
        if callable(get_app_state):
            try:
                app_state = get_app_state()
            except Exception:
                app_state = None
        elif hasattr(viewer, "app_state"):
            app_state = getattr(viewer, "app_state", None)

        app_data = getattr(app_state, "data", None)
        viewer_data = getattr(viewer, "data", None)
        if getattr(app_data, "ndim", 0) == 2 and getattr(viewer_data, "ndim", 0) > 2:
            return app_data.T if plane == "zy" else app_data

    data = getattr(viewer, "data", None)
    if data is not None:
        ndim = getattr(data, "ndim", 0)
        if is_cube:
            if ndim == 4:
                return data[0]
            if ndim == 3:
                return data
        else:
            if ndim == 2:
                return data
            if ndim == 3:
                if plane == "xy":
                    zpix = shared_index(viewer, "_get_shared_zpix", data.shape[0])
                    return data[zpix]
                if plane == "xz":
                    ypix = shared_index(viewer, "_get_shared_ypix", data.shape[1])
                    return data[:, ypix, :]
                if plane == "zy":
                    xpix = shared_index(viewer, "_get_shared_xpix", data.shape[2])
                    return data[:, :, xpix].T
            if ndim == 4:
                stokes = 0
                if plane == "xy":
                    zpix = shared_index(viewer, "_get_shared_zpix", data.shape[1])
                    return data[stokes, zpix]
                if plane == "xz":
                    ypix = shared_index(viewer, "_get_shared_ypix", data.shape[2])
                    return data[stokes, :, ypix, :]
                if plane == "zy":
                    xpix = shared_index(viewer, "_get_shared_xpix", data.shape[3])
                    return data[stokes, :, :, xpix].T

    if not is_cube and hasattr(viewer, "im"):
        return getattr(viewer.im, "get_array", lambda: None)()
    return None
