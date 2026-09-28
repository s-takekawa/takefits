"""Validate a selected-result render snapshot and turn it into CLI actions.

The GUI adapter deliberately emits only JSON-compatible values.  This module
stays Qt-free so manifest generation, tests, and future AI callers all apply
the same supported-field and coordinate rules.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from takefits.core.app_state import MarkerSpec, RegionSpec
from takefits.core.config import DEFAULT_CONFIG_KEYS
from takefits.core.wcs_frames import normalize_display_frame

from .diagnostics import PipelineDiagnostic


RENDER_SNAPSHOT_SCHEMA = 1
MOMENT_RENDER_CONFIG_KEYS = frozenset(
    key
    for key in DEFAULT_CONFIG_KEYS
    if key.startswith(
        (
            "ax_",
            "axislabel_",
            "beam_",
            "cbar_",
            "colorbar_",
            "grid_",
            "tick_",
            "x_mtick_",
            "x_tick_",
            "y_mtick_",
            "y_tick_",
            "z_mtick_",
            "z_tick_",
        )
    )
    or key
    in {
        "colorscale",
        "fig_background_color",
        "bad_color",
        "decimal",
        "number_decimals",
        "coord_wrap",
        "default_ticks_position",
        "mtick_length",
        "xticklabel_position",
        "yticklabel_position",
    }
)
_TOP_LEVEL_KEYS = {
    "schema",
    "target",
    "image",
    "view",
    "grid",
    "render_config",
    "annotations",
    "contours",
    "beam",
    "figure",
}
_CONTOUR_KEYS = {
    "plane",
    "filepath",
    "channel",
    "levels",
    "color",
    "linewidth",
    "linestyle",
    "negative_linestyle",
    "smoothing",
    "label",
}
_MARKER_KINDS = {"symbol", "line", "text"}
_REGION_TYPES = {"circle", "rectangle", "ellipse", "cube"}


@dataclass(frozen=True)
class MomentRenderPlan:
    """State-setting actions and explicit image parameters for one snapshot."""

    actions: Tuple[Dict[str, Any], ...]
    image_params: Dict[str, Any]
    diagnostics: Tuple[PipelineDiagnostic, ...]

    @property
    def ok(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)


def _add(
    diagnostics: list[PipelineDiagnostic],
    severity: str,
    code: str,
    message: str,
    location: str,
) -> None:
    diagnostics.append(PipelineDiagnostic(severity, code, message, location))


def _finite_pair(
    value: Any,
    *,
    diagnostics: list[PipelineDiagnostic],
    code: str,
    location: str,
    allow_equal: bool = False,
) -> Optional[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        _add(diagnostics, "error", code, "Expected two finite numbers.", location)
        return None
    try:
        pair = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        _add(diagnostics, "error", code, "Expected two finite numbers.", location)
        return None
    if not all(math.isfinite(item) for item in pair) or (
        not allow_equal and pair[0] == pair[1]
    ):
        qualifier = "finite" if allow_equal else "different finite"
        _add(diagnostics, "error", code, f"Expected two {qualifier} numbers.", location)
        return None
    return pair


def _json_copy(
    value: Any,
    *,
    diagnostics: list[PipelineDiagnostic],
    location: str,
) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        _add(
            diagnostics,
            "error",
            "render_snapshot_not_json_compatible",
            f"Render snapshot value is not JSON-compatible: {exc}",
            location,
        )
        return None


def _portable_path(raw_path: str, manifest_dir: Path) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        # Forward slashes keep a manifest written on Windows runnable elsewhere.
        return Path(os.path.relpath(absolute, start=manifest_dir)).as_posix()
    except ValueError:
        return str(absolute)


def _validate_annotations(
    raw: Any,
    *,
    kind: str,
    diagnostics: list[PipelineDiagnostic],
) -> list[Dict[str, Any]]:
    location = f"render_snapshot.annotations.{kind}"
    if not isinstance(raw, list):
        _add(diagnostics, "error", "invalid_render_annotations", f"{kind} must be a list.", location)
        return []
    validated: list[Dict[str, Any]] = []
    factory = MarkerSpec.from_dict if kind == "markers" else RegionSpec.from_dict
    for index, value in enumerate(raw):
        item_location = f"{location}[{index}]"
        if not isinstance(value, Mapping):
            _add(diagnostics, "error", "invalid_render_annotation", "Annotation must be an object.", item_location)
            continue
        payload = _json_copy(dict(value), diagnostics=diagnostics, location=item_location)
        if payload is None:
            continue
        if str(payload.get("plane") or "xy").lower() != "xy":
            _add(
                diagnostics,
                "error",
                "unsupported_annotation_plane",
                "The first render-snapshot slice supports only XY result annotations.",
                f"{item_location}.plane",
            )
            continue
        if kind == "markers":
            marker_kind = str(payload.get("kind") or "symbol").lower()
            if marker_kind not in _MARKER_KINDS:
                _add(
                    diagnostics,
                    "error",
                    "unsupported_marker_kind",
                    f"Unsupported marker kind {marker_kind!r}.",
                    f"{item_location}.kind",
                )
                continue
        else:
            region_type = str(
                payload.get("type") or payload.get("kind") or "circle"
            ).lower()
            if region_type not in _REGION_TYPES:
                _add(
                    diagnostics,
                    "error",
                    "unsupported_region_type",
                    f"Unsupported region type {region_type!r}.",
                    f"{item_location}.type",
                )
                continue
        try:
            factory(payload)
        except Exception as exc:
            _add(diagnostics, "error", "invalid_render_annotation", str(exc), item_location)
            continue
        validated.append(payload)
    return validated


def _validate_contours(
    raw: Any,
    *,
    manifest_dir: Path,
    diagnostics: list[PipelineDiagnostic],
) -> list[Dict[str, Any]]:
    location = "render_snapshot.contours"
    if not isinstance(raw, list):
        _add(diagnostics, "error", "invalid_render_contours", "contours must be a list.", location)
        return []
    validated: list[Dict[str, Any]] = []
    for index, value in enumerate(raw):
        item_location = f"{location}[{index}]"
        if not isinstance(value, Mapping):
            _add(diagnostics, "error", "invalid_render_contour", "Contour must be an object.", item_location)
            continue
        payload = dict(value)
        unknown = sorted(set(payload) - _CONTOUR_KEYS)
        if unknown:
            _add(
                diagnostics,
                "error",
                "unsupported_render_contour_field",
                f"Unsupported contour field(s): {', '.join(unknown)}.",
                item_location,
            )
            continue
        if str(payload.get("plane") or "xy").lower() != "xy":
            _add(diagnostics, "error", "unsupported_contour_plane", "Only XY contours are supported.", f"{item_location}.plane")
            continue
        levels = payload.get("levels")
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or not levels:
            _add(diagnostics, "error", "invalid_render_contour_levels", "Contour levels must be a non-empty list.", f"{item_location}.levels")
            continue
        try:
            payload["levels"] = [float(level) for level in levels]
        except (TypeError, ValueError):
            _add(diagnostics, "error", "invalid_render_contour_levels", "Contour levels must be finite numbers.", f"{item_location}.levels")
            continue
        if not all(math.isfinite(level) for level in payload["levels"]):
            _add(diagnostics, "error", "invalid_render_contour_levels", "Contour levels must be finite numbers.", f"{item_location}.levels")
            continue
        try:
            linewidth = float(payload.get("linewidth", 1.0))
            smoothing = float(payload.get("smoothing", 0.0))
        except (TypeError, ValueError):
            linewidth = smoothing = float("nan")
        if (
            not math.isfinite(linewidth)
            or linewidth <= 0.0
            or not math.isfinite(smoothing)
            or smoothing < 0.0
        ):
            _add(
                diagnostics,
                "error",
                "invalid_render_contour_style",
                "Contour linewidth must be positive and smoothing non-negative.",
                item_location,
            )
            continue
        payload["linewidth"] = linewidth
        payload["smoothing"] = smoothing
        color = payload.get("color", "white")
        if not isinstance(color, str) or not color.strip():
            _add(
                diagnostics,
                "error",
                "invalid_render_contour_style",
                "Contour color must be a non-empty string.",
                f"{item_location}.color",
            )
            continue
        payload["color"] = color
        channel = payload.get("channel")
        if channel is not None and (
            isinstance(channel, bool) or not isinstance(channel, int) or channel < 0
        ):
            _add(
                diagnostics,
                "error",
                "invalid_external_contour_channel",
                "External contour channel must be a non-negative integer.",
                f"{item_location}.channel",
            )
            continue
        filepath = payload.get("filepath")
        if filepath is not None:
            if not isinstance(filepath, str) or not filepath.strip():
                _add(diagnostics, "error", "invalid_external_contour_path", "External contour filepath must be non-empty.", f"{item_location}.filepath")
                continue
            absolute = Path(filepath).expanduser()
            if not absolute.is_absolute():
                absolute = manifest_dir / absolute
            absolute = Path(os.path.abspath(os.fspath(absolute)))
            if not absolute.is_file():
                _add(diagnostics, "error", "external_contour_not_found", f"External contour FITS file does not exist: {absolute}", f"{item_location}.filepath")
                continue
            payload["filepath"] = _portable_path(filepath, manifest_dir)
        payload = _json_copy(payload, diagnostics=diagnostics, location=item_location)
        if payload is not None:
            validated.append(payload)
    return validated


def build_moment_render_plan(
    snapshot: Mapping[str, Any],
    *,
    manifest_dir: Path,
    selected_result_tag: str = "",
) -> MomentRenderPlan:
    """Validate one GUI snapshot and produce deterministic manifest fragments."""
    diagnostics: list[PipelineDiagnostic] = []
    if not isinstance(snapshot, Mapping):
        _add(diagnostics, "error", "invalid_render_snapshot", "render_snapshot must be an object.", "render_snapshot")
        return MomentRenderPlan((), {}, tuple(diagnostics))
    unknown = sorted(set(snapshot) - _TOP_LEVEL_KEYS)
    if unknown:
        _add(diagnostics, "error", "unsupported_render_snapshot_field", f"Unsupported render snapshot field(s): {', '.join(unknown)}.", "render_snapshot")
    if snapshot.get("schema") != RENDER_SNAPSHOT_SCHEMA:
        _add(diagnostics, "error", "unsupported_render_snapshot_schema", f"Expected render snapshot schema {RENDER_SNAPSHOT_SCHEMA}.", "render_snapshot.schema")

    target = snapshot.get("target")
    if not isinstance(target, Mapping):
        _add(diagnostics, "error", "invalid_render_target", "target must be an object.", "render_snapshot.target")
        target = {}
    if target.get("kind") != "integration_result" or str(target.get("plane") or "").lower() != "xy":
        _add(diagnostics, "error", "unsupported_render_target", "The first slice supports an XY integration result window.", "render_snapshot.target")
    if target.get("coordinate_basis") != "source_xy_pixel":
        _add(diagnostics, "error", "unsupported_coordinate_basis", "Expected source_xy_pixel coordinates for the selected result.", "render_snapshot.target.coordinate_basis")
    snapshot_tag = str(target.get("result_tag") or "").strip()
    if selected_result_tag and snapshot_tag != selected_result_tag:
        _add(diagnostics, "error", "render_result_tag_mismatch", "The render snapshot belongs to a different Moment result.", "render_snapshot.target.result_tag")

    image = snapshot.get("image")
    if not isinstance(image, Mapping):
        _add(diagnostics, "error", "invalid_render_image", "image must be an object.", "render_snapshot.image")
        image = {}
    origin = str(image.get("origin") or "lower").lower()
    if origin != "lower":
        _add(diagnostics, "error", "unsupported_image_origin", "Integration-result snapshots currently require origin='lower'.", "render_snapshot.image.origin")
    norm = image.get("norm")
    if not isinstance(norm, Mapping):
        _add(diagnostics, "error", "invalid_render_norm", "image.norm must be an object.", "render_snapshot.image.norm")
        norm = {}
    norm_type = str(norm.get("type") or "").lower()
    if norm_type not in {"linear", "log"}:
        _add(diagnostics, "error", "unsupported_render_norm", "Only linear and log normalization are supported.", "render_snapshot.image.norm.type")
    clim = _finite_pair(image.get("clim"), diagnostics=diagnostics, code="invalid_render_clim", location="render_snapshot.image.clim")
    if clim is not None and norm_type == "log" and clim[0] <= 0:
        _add(diagnostics, "error", "invalid_log_render_clim", "Log normalization requires a positive lower limit.", "render_snapshot.image.clim")
    cmap = image.get("cmap")
    if not isinstance(cmap, Mapping):
        _add(diagnostics, "error", "invalid_render_colormap", "image.cmap must be an object.", "render_snapshot.image.cmap")
        cmap = {}
    cmap_name = str(cmap.get("name") or "").strip()
    rgba = cmap.get("rgba")
    if not cmap_name and rgba is None:
        _add(diagnostics, "error", "missing_render_colormap", "A named or sampled colormap is required.", "render_snapshot.image.cmap")
    rgba_copy = None
    if rgba is not None:
        if not isinstance(rgba, list) or len(rgba) < 2:
            _add(diagnostics, "error", "invalid_render_colormap_samples", "Colormap samples must contain at least two RGBA rows.", "render_snapshot.image.cmap.rgba")
        else:
            try:
                rgba_copy = [[float(component) for component in row] for row in rgba]
            except (TypeError, ValueError):
                rgba_copy = None
            if rgba_copy is None or any(len(row) != 4 or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in row) for row in rgba_copy):
                _add(diagnostics, "error", "invalid_render_colormap_samples", "Every colormap sample must be four finite values in [0, 1].", "render_snapshot.image.cmap.rgba")
                rgba_copy = None

    view = snapshot.get("view")
    if not isinstance(view, Mapping):
        _add(diagnostics, "error", "invalid_render_view", "view must be an object.", "render_snapshot.view")
        view = {}
    xlim = _finite_pair(view.get("xlim"), diagnostics=diagnostics, code="invalid_render_xlim", location="render_snapshot.view.xlim")
    ylim = _finite_pair(view.get("ylim"), diagnostics=diagnostics, code="invalid_render_ylim", location="render_snapshot.view.ylim")

    grid = snapshot.get("grid")
    if not isinstance(grid, Mapping) or not isinstance(grid.get("visible"), bool) or not isinstance(grid.get("keep_native"), bool):
        _add(diagnostics, "error", "invalid_render_grid", "grid needs boolean visible and keep_native values.", "render_snapshot.grid")
        grid = {}
    frame = normalize_display_frame(grid.get("frame", "native"))

    render_config = snapshot.get("render_config")
    if not isinstance(render_config, Mapping):
        _add(diagnostics, "error", "invalid_render_config", "render_config must be an object.", "render_snapshot.render_config")
        render_config = {}
    unknown_config = sorted(set(render_config) - MOMENT_RENDER_CONFIG_KEYS)
    if unknown_config:
        _add(diagnostics, "error", "unsupported_render_config_key", f"Unknown render config key(s): {', '.join(unknown_config)}.", "render_snapshot.render_config")
    render_config_copy = _json_copy(dict(render_config), diagnostics=diagnostics, location="render_snapshot.render_config") or {}

    annotations = snapshot.get("annotations")
    if not isinstance(annotations, Mapping):
        _add(diagnostics, "error", "invalid_render_annotations", "annotations must be an object.", "render_snapshot.annotations")
        annotations = {}
    markers = _validate_annotations(annotations.get("markers"), kind="markers", diagnostics=diagnostics)
    regions = _validate_annotations(annotations.get("regions"), kind="regions", diagnostics=diagnostics)
    contours = _validate_contours(snapshot.get("contours"), manifest_dir=manifest_dir, diagnostics=diagnostics)

    beam = snapshot.get("beam")
    if not isinstance(beam, Mapping) or not isinstance(beam.get("visible"), bool):
        _add(diagnostics, "error", "invalid_render_beam", "beam.visible must be boolean.", "render_snapshot.beam")
        beam = {}

    figure = snapshot.get("figure")
    if not isinstance(figure, Mapping):
        _add(diagnostics, "error", "invalid_render_figure", "figure must be an object.", "render_snapshot.figure")
        figure = {}
    figsize = _finite_pair(
        figure.get("figsize"),
        diagnostics=diagnostics,
        code="invalid_render_figsize",
        location="render_snapshot.figure.figsize",
        allow_equal=True,
    )
    if figsize is not None and min(figsize) <= 0:
        _add(diagnostics, "error", "invalid_render_figsize", "Figure dimensions must be positive.", "render_snapshot.figure.figsize")
    dpi = figure.get("dpi")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        _add(diagnostics, "error", "invalid_render_dpi", "figure.dpi must be a positive integer.", "render_snapshot.figure.dpi")

    if any(item.severity == "error" for item in diagnostics):
        return MomentRenderPlan((), {}, tuple(diagnostics))

    assert clim is not None and xlim is not None and ylim is not None and figsize is not None
    image_params: Dict[str, Any] = {
        "origin": origin,
        "vmin": clim[0],
        "vmax": clim[1],
        "log_scale": norm_type == "log",
        "xlim": xlim,
        "ylim": ylim,
        "grid": bool(grid["visible"]),
        "grid_frame": frame,
        "grid_keep_native": bool(grid["keep_native"]),
        "figsize": figsize,
        "dpi": int(dpi),
        "draw_markers": bool(markers),
        "draw_regions": bool(regions),
        "draw_contours": bool(contours),
        "draw_beam": bool(beam["visible"]),
    }
    if rgba_copy is not None:
        image_params["cmap_rgba"] = rgba_copy
    else:
        image_params["cmap"] = cmap_name

    actions = (
        {"action": "set_render_config", "params": {"overrides": render_config_copy, "replace": True}},
        {"action": "set_view_range", "params": {"plane": "xy", "xlim": xlim, "ylim": ylim}},
        {"action": "set_coordinate_grid", "params": {"visible": bool(grid["visible"]), "frame": frame, "keep_native": bool(grid["keep_native"])}},
        {"action": "set_markers", "params": {"markers": markers}},
        {"action": "set_regions", "params": {"regions": regions}},
        {"action": "set_contours", "params": {"contours": contours}},
    )
    return MomentRenderPlan(actions, image_params, tuple(diagnostics))
