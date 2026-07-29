"""Headless contour rendering (TF-303).

Contours over a moment map are the most standard radio-astronomy figure, but
they had no action at all: `core/usecases_contour.py` is a thin wrapper over
`ContourManager`, which is a Qt singleton built around live viewers, overlay
sessions, and blit bookkeeping.

None of that is needed to *draw* contours. The pieces that matter are already
Qt-free:

- `contour_manager.resolve_contour_levels` - the level ladder the GUI uses
- `contour_external` - rms estimation, sigma levels, smoothing, and
  world-coordinate contour states built from a separate FITS file
- `contour_manager._world_coords_to_pixel` - reprojection onto a target WCS

This module renders with those, so a CLI figure matches the viewer without
instantiating the manager.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from takefits.core.contour_manager import (
    ContourParameters,
    _world_coords_to_pixel,
    resolve_contour_levels,
)


def resolve_levels(
    data: Optional[np.ndarray] = None,
    *,
    levels: Optional[Sequence[float]] = None,
    level_min: Optional[float] = None,
    level_max: Optional[float] = None,
    level_step: Optional[float] = None,
    sigma_levels: Optional[Sequence[float]] = None,
    rms: Optional[float] = None,
    include_negative: bool = False,
) -> List[float]:
    """Return contour levels in data units.

    Three mutually exclusive forms, checked in this order:

    1. ``sigma_levels`` - multiples of the rms, the usual way a paper states
       contours ("3, 6, 12 sigma"). ``rms`` may be given explicitly; otherwise
       it is estimated from *data*.
    2. ``levels`` - explicit values.
    3. ``level_min`` / ``level_max`` / ``level_step`` - a linear ladder.

    Raises:
        ValueError: when no form yields a usable level set.
    """
    if sigma_levels:
        from takefits.core.contour_external import estimate_rms
        from takefits.core.contour_external import sigma_levels as _sigma_levels

        noise = rms
        if noise is None:
            if data is None:
                raise ValueError(
                    "sigma_levels needs either an explicit rms or data to estimate it"
                )
            noise = estimate_rms(np.asarray(data))
        if noise is None or not np.isfinite(noise) or noise <= 0.0:
            raise ValueError("Could not determine a positive rms for sigma levels")
        resolved = _sigma_levels(
            float(noise), sigma_levels, include_negative=include_negative
        )
        if not resolved:
            raise ValueError("sigma_levels produced no finite levels")
        return [float(v) for v in resolved]

    if levels is None and level_min is None and level_max is None and level_step is None:
        # Without this, an empty spec would fall through to a degenerate
        # min == max == 0 ladder and silently draw a single zero contour.
        raise ValueError(
            "No usable contour levels; give sigma_levels, levels, "
            "or level_min/level_max/level_step"
        )

    if data is not None:
        finite = np.asarray(data)[np.isfinite(np.asarray(data))]
        data_min = float(finite.min()) if finite.size else 0.0
        data_max = float(finite.max()) if finite.size else 0.0
    else:
        data_min, data_max = 0.0, 0.0

    params = ContourParameters(
        level_min=level_min,
        level_max=level_max,
        level_step=level_step,
        levels=list(levels) if levels is not None else None,
    )
    resolved = resolve_contour_levels(params, data_min, data_max)
    if not resolved:
        raise ValueError("No usable contour levels; give levels or min/max/step")
    return [float(v) for v in resolved]


def draw_contours_on_axes(
    ax: Any,
    data: np.ndarray,
    *,
    levels: Optional[Sequence[float]] = None,
    level_min: Optional[float] = None,
    level_max: Optional[float] = None,
    level_step: Optional[float] = None,
    sigma_levels: Optional[Sequence[float]] = None,
    rms: Optional[float] = None,
    include_negative: bool = False,
    color: str = "white",
    linewidth: float = 1.0,
    linestyle: str = "solid",
    negative_linestyle: str = "dashed",
    smoothing: float = 0.0,
    alpha: Optional[float] = None,
) -> Any:
    """Draw contours of *data* on *ax*, returning the ``QuadContourSet``.

    Negative levels get ``negative_linestyle`` so a paper figure distinguishes
    absorption or negative residuals, which is the usual convention.
    """
    array = np.asarray(data, dtype=float)
    if smoothing and float(smoothing) > 0.0:
        from takefits.core.contour_external import smooth_plane

        array = smooth_plane(array, float(smoothing))

    resolved = resolve_levels(
        array,
        levels=levels,
        level_min=level_min,
        level_max=level_max,
        level_step=level_step,
        sigma_levels=sigma_levels,
        rms=rms,
        include_negative=include_negative,
    )

    styles = [
        negative_linestyle if value < 0 else linestyle for value in resolved
    ]
    contour_set = ax.contour(
        array,
        levels=resolved,
        colors=color,
        linewidths=linewidth,
        linestyles=styles,
        alpha=alpha,
    )
    return contour_set


def draw_contour_state_on_axes(
    ax: Any,
    contour_state: Any,
    target_wcs: Any,
    *,
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    linestyle: Optional[str] = None,
) -> int:
    """Draw a world-coordinate `ContourState` reprojected onto *target_wcs*.

    This is the external-FITS overlay path: contours are computed on the source
    grid, carried as world coordinates, and mapped onto the target pixel grid
    here. No regridding of the science data takes place.

    Returns:
        The number of polylines drawn.
    """
    from matplotlib.collections import LineCollection

    if contour_state is None or target_wcs is None:
        return 0

    segments: List[np.ndarray] = []
    for item in getattr(contour_state, "items", None) or []:
        for segment in getattr(item, "segments", None) or []:
            world = getattr(segment, "world", None)
            if world is None:
                pixels = getattr(segment, "pixels", None)
                if pixels is None:
                    continue
                points = np.asarray(pixels, dtype=float)
            else:
                points = _world_coords_to_pixel(
                    np.asarray(world, dtype=float),
                    target_wcs,
                    getattr(contour_state, "world_frame", None),
                )
                if points is None:
                    continue
                points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[0] < 2:
                continue
            finite = np.isfinite(points).all(axis=1)
            if finite.sum() < 2:
                continue
            segments.append(points[finite])

    if not segments:
        return 0

    parameters = getattr(contour_state, "parameters", None)
    collection = LineCollection(
        segments,
        colors=color or getattr(parameters, "color", None) or "white",
        linewidths=(
            linewidth
            if linewidth is not None
            else float(getattr(parameters, "linewidth", 1.0) or 1.0)
        ),
        linestyles=linestyle or "solid",
        zorder=7,
    )
    ax.add_collection(collection)
    return len(segments)


def _external_plane(filepath: str, channel: Optional[int] = None) -> np.ndarray:
    """The 2D plane a contour source FITS contributes, for rms estimation."""
    from takefits.core.contour_external import _slice_xy_plane
    from takefits.core.usecases import load_fits_data

    return _slice_xy_plane(load_fits_data(str(filepath)), channel)


def draw_contour_specs_on_axes(
    ax: Any,
    specs: Iterable[Dict[str, Any]],
    *,
    data: Optional[np.ndarray] = None,
    target_wcs: Any = None,
    plane: str = "xy",
) -> List[Any]:
    """Draw every contour spec that belongs to *plane*.

    A spec either contours the rendered image itself (no ``filepath``) or loads
    an external FITS and overlays it by world coordinate.
    """
    drawn: List[Any] = []
    for entry in specs or ():
        spec = dict(entry)
        if str(spec.pop("plane", "xy") or "xy") != str(plane):
            continue

        filepath = spec.pop("filepath", None)
        if filepath:
            from takefits.core.contour_external import build_contour_state_from_fits

            channel = spec.pop("channel", None)
            external_levels = spec.pop("levels", None)
            sigma = spec.pop("sigma_levels", None)
            rms = spec.pop("rms", None)
            include_negative = bool(spec.pop("include_negative", False))
            level_min = spec.pop("level_min", None)
            level_max = spec.pop("level_max", None)
            level_step = spec.pop("level_step", None)
            smoothing = float(spec.pop("smoothing", 0.0) or 0.0)

            # Resolve every supported level form before building the external
            # contour state.  Previously the linear ladder fields were simply
            # discarded, leaving the builder with an empty level list.
            source_plane = _external_plane(str(filepath), channel)
            if smoothing > 0.0:
                from takefits.core.contour_external import smooth_plane

                source_plane = smooth_plane(source_plane, smoothing)
            external_levels = resolve_levels(
                source_plane,
                levels=external_levels,
                level_min=level_min,
                level_max=level_max,
                level_step=level_step,
                sigma_levels=sigma,
                rms=rms,
                include_negative=include_negative,
            )

            state = build_contour_state_from_fits(
                str(filepath),
                external_levels,
                channel=channel,
                color=spec.get("color", "white"),
                linewidth=spec.get("linewidth", 1.0),
                smoothing=smoothing,
                label=spec.pop("label", None),
            )
            count = draw_contour_state_on_axes(
                ax,
                state,
                target_wcs,
                color=spec.get("color"),
                linewidth=spec.get("linewidth"),
                linestyle=spec.get("linestyle"),
            )
            if count:
                drawn.append(state)
            continue

        if data is None:
            continue
        spec.pop("label", None)
        spec.pop("channel", None)
        drawn.append(draw_contours_on_axes(ax, data, **spec))
    return drawn
