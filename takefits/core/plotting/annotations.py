"""Headless annotation rendering (TF-302).

``core/marker.py`` already builds pure-Matplotlib artists for every marker
kind, and ``MarkerSpec.to_dict()`` (headless state) is payload-compatible with
``MarkerState.from_dict()`` (artist layer). This module is the small bridge
that lets a headless export draw ``state.markers`` with the same code the GUI
uses, so an image produced by the CLI matches what the viewer shows.

It also resolves the two things a hand-written manifest should not have to
compute itself:

- an angular marker length (``metadata.unit`` = arcsec/arcmin/deg with
  ``metadata.length_source``) becomes pixels via the WCS pixel scale
- ``metadata.anchor_frac`` places a marker by axes fraction instead of data
  pixels, so "bottom right" does not depend on the image dimensions
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from takefits.core.marker import MarkerState, marker_from_state
from takefits.core.marker_utils import (
    angular_length_to_pixels,
    resolve_anchor_fraction,
)
from takefits.core.region import (
    CircleRegion,
    EllipseRegion,
    RectangleRegion,
)


def _resolve_angular_length(
    metadata: Dict[str, Any],
    wcs: Any,
) -> None:
    """Rewrite ``metadata['length']`` in pixels when the unit is angular.

    ``length`` is what the renderer consumes and is always pixels; ``unit`` and
    ``length_source`` carry the user's intent. The GUI does this conversion in
    its Qt-bound marker manager, so headless renders previously ignored the
    unit and drew the raw pixel value.
    """
    unit = metadata.get("unit")
    if unit is None:
        return
    source = metadata.get("length_source", metadata.get("length"))
    if source is None:
        return
    pixels = angular_length_to_pixels(source, unit, wcs)
    if pixels is not None:
        metadata["length"] = float(pixels)


def _resolve_anchor(
    payload: Dict[str, Any],
    metadata: Dict[str, Any],
    shape: Optional[Sequence[int]],
) -> None:
    """Replace ``payload['pixel']`` from ``metadata['anchor_frac']``.

    For a line marker the anchor point is the segment centre by default;
    ``metadata['anchor']`` of ``start`` or ``end`` puts the corresponding
    endpoint on the fraction instead, which is what a bottom-right scale bar
    wants without the caller doing the arithmetic.
    """
    anchor_frac = metadata.get("anchor_frac")
    if anchor_frac is None or shape is None:
        return
    resolved = resolve_anchor_fraction(anchor_frac, shape)
    if resolved is None:
        return

    x, y = resolved
    if str(payload.get("kind") or "symbol") == "line":
        anchor = str(metadata.get("anchor") or "center").strip().lower()
        if anchor in ("start", "end"):
            length = float(metadata.get("length", 0.0) or 0.0)
            angle = math.radians(float(metadata.get("angle_deg", 0.0) or 0.0))
            half_x = 0.5 * length * math.cos(angle)
            half_y = 0.5 * length * math.sin(angle)
            if anchor == "start":
                x, y = x + half_x, y + half_y
            else:
                x, y = x - half_x, y - half_y
        # The centre is now explicit, so keep the constructor from re-applying
        # its own legacy start-to-centre conversion.
        metadata["anchor"] = "center"

    payload["pixel"] = [x, y]


def draw_markers_on_axes(
    ax: Any,
    markers: Iterable[Any],
    plane: str = "xy",
    *,
    wcs: Any = None,
    shape: Optional[Sequence[int]] = None,
) -> List[Any]:
    """Draw every marker belonging to *plane* onto *ax*.

    Args:
        ax: Matplotlib axes (plain or WCSAxes) in pixel data coordinates.
        markers: Iterable of ``MarkerSpec`` (or payload dicts).
        plane: Only markers whose plane matches are drawn.
        wcs: WCS of the rendered image, used to convert angular lengths.
        shape: ``(ny, nx)`` of the rendered image, used for ``anchor_frac``.

    Returns:
        The runtime ``Marker`` objects that were attached, in draw order.
    """
    drawn: List[Any] = []
    for entry in markers or ():
        payload = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        if str(payload.get("plane") or "xy") != str(plane):
            continue

        metadata = dict(payload.get("metadata") or {})
        _resolve_angular_length(metadata, wcs)
        _resolve_anchor(payload, metadata, shape)
        payload["metadata"] = metadata

        marker = marker_from_state(MarkerState.from_dict(payload))
        marker.add_to_axes(ax)
        drawn.append(marker)
    return drawn


def _region_from_payload(payload: Dict[str, Any]) -> Optional[Any]:
    """Build a runtime ``Region`` from a ``RegionSpec`` payload."""
    region_type = str(payload.get("type") or payload.get("kind") or "circle").lower()
    center_x = float(payload.get("center_x", 0.0))
    center_y = float(payload.get("center_y", 0.0))
    params = dict(payload.get("params") or {})
    style = dict(payload.get("style") or {})

    if region_type == "circle":
        return CircleRegion(
            center=(center_x, center_y),
            radius=float(params.get("radius", 1.0)),
            style=style,
        )

    width = float(params.get("width", 1.0))
    height = float(params.get("height", 1.0))
    angle = float(params.get("angle", 0.0) or 0.0)

    if region_type in ("rectangle", "cube"):
        # RectangleRegion takes the lower-left corner; RegionSpec stores the
        # centre, the same convention `tools/spectrum.py` uses.
        region = RectangleRegion(
            xy=(center_x - width / 2.0, center_y - height / 2.0),
            width=width,
            height=height,
            style=style,
        )
    elif region_type == "ellipse":
        region = EllipseRegion(
            center=(center_x, center_y),
            width=width,
            height=height,
            style=style,
        )
    else:
        return None

    if angle:
        region.set_angle(angle)
    return region


def draw_regions_on_axes(
    ax: Any,
    regions: Iterable[Any],
    plane: str = "xy",
) -> List[Any]:
    """Draw every region belonging to *plane* onto *ax*.

    The GUI creates region patches with ``animated=True`` because it blits
    them; Matplotlib skips animated artists during a normal draw, so a static
    export has to clear the flag or the region never appears.

    Args:
        ax: Matplotlib axes in pixel data coordinates.
        regions: Iterable of ``RegionSpec`` (or payload dicts).
        plane: Only regions whose plane matches are drawn.

    Returns:
        The runtime ``Region`` objects that were attached, in draw order.
    """
    drawn: List[Any] = []
    for entry in regions or ():
        payload = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        if str(payload.get("plane") or "xy") != str(plane):
            continue

        region = _region_from_payload(payload)
        if region is None:
            continue

        region.add_to_axes(ax)
        if region.mpl_patch is not None:
            region.mpl_patch.set_animated(False)

        label = str(payload.get("label") or "")
        if label:
            region.set_label_text(label)
        if region.label_artist is not None:
            region.label_artist.set_animated(False)

        drawn.append(region)
    return drawn
