"""Capture a selected Moment result window as a portable render snapshot."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from takefits.core.annotation_serialization import (
    snapshot_marker_specs,
    snapshot_region_specs,
)
from takefits.core.config import build_default_config
from takefits.core.pipeline.diagnostics import PipelineDiagnostic
from takefits.core.pipeline.render_snapshot import (
    MOMENT_RENDER_CONFIG_KEYS,
    RENDER_SNAPSHOT_SCHEMA,
)


@dataclass(frozen=True)
class MomentRenderSnapshotResult:
    """Captured JSON snapshot plus any actionable compatibility diagnostics."""

    snapshot: Optional[Dict[str, Any]]
    diagnostics: Tuple[PipelineDiagnostic, ...]

    @property
    def ok(self) -> bool:
        return self.snapshot is not None and not any(
            item.severity == "error" for item in self.diagnostics
        )


def _add(
    diagnostics: list[PipelineDiagnostic],
    code: str,
    message: str,
    location: str,
) -> None:
    diagnostics.append(PipelineDiagnostic("error", code, message, location))


def _pair(value: Any) -> Optional[list[float]]:
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(values) != 2 or not all(math.isfinite(item) for item in values):
        return None
    return values


def _capture_colormap(image: Any, diagnostics: list[PipelineDiagnostic]) -> Dict[str, Any]:
    try:
        cmap = image.get_cmap()
        name = str(getattr(cmap, "name", "") or "")
    except Exception:
        _add(diagnostics, "render_colormap_unavailable", "The selected image colormap could not be read.", "render_snapshot.image.cmap")
        return {}

    payload: Dict[str, Any] = {"name": name}
    needs_samples = not name or name == "from_list"
    if not needs_samples:
        try:
            import matplotlib as mpl

            mpl.colormaps[name]
        except Exception:
            needs_samples = True
    if needs_samples:
        try:
            import numpy as np

            count = max(2, min(256, int(getattr(cmap, "N", 256) or 256)))
            payload["rgba"] = np.asarray(
                cmap(np.linspace(0.0, 1.0, count)), dtype=float
            ).tolist()
        except Exception:
            _add(diagnostics, "render_colormap_samples_unavailable", "The custom colormap could not be sampled.", "render_snapshot.image.cmap")
    return payload


def _capture_image(image: Any, diagnostics: list[PipelineDiagnostic]) -> Dict[str, Any]:
    if image is None:
        _add(diagnostics, "render_image_unavailable", "The selected result has no image artist.", "render_snapshot.image")
        return {}
    clim = None
    try:
        clim = _pair(image.get_clim())
    except Exception:
        pass
    if clim is None or clim[0] == clim[1]:
        _add(diagnostics, "render_clim_unavailable", "The selected image needs two different finite color limits.", "render_snapshot.image.clim")

    norm_type = None
    try:
        import matplotlib as mpl

        norm = getattr(image, "norm", None)
        if isinstance(norm, mpl.colors.LogNorm):
            norm_type = "log"
        elif type(norm) is mpl.colors.Normalize:
            norm_type = "linear"
        else:
            _add(
                diagnostics,
                "unsupported_render_norm",
                f"Normalization {type(norm).__name__!r} is not supported; use linear or log.",
                "render_snapshot.image.norm",
            )
    except Exception:
        _add(diagnostics, "render_norm_unavailable", "The selected image normalization could not be read.", "render_snapshot.image.norm")

    return {
        "cmap": _capture_colormap(image, diagnostics),
        "clim": clim,
        "norm": {"type": norm_type},
        "origin": str(getattr(image, "origin", "lower") or "lower"),
    }


def _capture_render_config(window: Any, diagnostics: list[PipelineDiagnostic]) -> Dict[str, Any]:
    defaults = build_default_config()
    source = getattr(window, "config", None)
    source = source if isinstance(source, Mapping) else {}
    config = {
        key: source.get(key, defaults[key])
        for key in sorted(MOMENT_RENDER_CONFIG_KEYS)
        if key in defaults
    }

    # Manual colorbar placement is live state, not reliably reflected back into
    # config.  Auto layout is instead reproduced from its semantic settings.
    auto_layout = bool(config.get("colorbar_auto_layout", False))
    checker = getattr(window, "_is_colorbar_auto_layout_enabled", None)
    if callable(checker):
        try:
            auto_layout = bool(checker())
        except Exception:
            pass
    config["colorbar_auto_layout"] = auto_layout
    if not auto_layout:
        try:
            left, bottom, width, height = (
                float(item) for item in window.cax.get_position().bounds
            )
            config.update(
                {
                    "cbar_pos_x": left,
                    "cbar_pos_y": bottom,
                    "cbar_width": width,
                    "cbar_height": height,
                }
            )
        except Exception:
            _add(diagnostics, "render_colorbar_bounds_unavailable", "Manual colorbar bounds could not be captured.", "render_snapshot.render_config")
    try:
        orientation = str(window.colorbar.orientation or "").lower()
        if orientation in {"vertical", "horizontal"}:
            config["colorbar_orientation"] = orientation
    except Exception:
        pass
    return config


def _contour_spec_from_state(
    state: Any,
    *,
    external: bool,
    diagnostics: list[PipelineDiagnostic],
    location: str,
) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    plane = str(getattr(state, "plane", "xy") or "xy").lower()
    if plane != "xy":
        _add(diagnostics, "unsupported_contour_plane", "Only XY result contours are supported.", location)
        return None
    levels = []
    try:
        levels = [float(value) for value in (getattr(state, "levels", None) or [])]
    except (TypeError, ValueError):
        levels = []
    if not levels or not all(math.isfinite(value) for value in levels):
        _add(diagnostics, "render_contour_levels_unavailable", "Contour levels could not be captured.", location)
        return None
    params = getattr(state, "parameters", None)
    color = str(getattr(params, "color", "white") or "white")
    if color.lower() == "rainbow":
        _add(diagnostics, "unsupported_rainbow_contours", "Per-segment rainbow contour colors are not yet reproducible headlessly.", location)
        return None
    spec: Dict[str, Any] = {
        "plane": "xy",
        "levels": levels,
        "color": color,
    }
    try:
        spec["linewidth"] = float(getattr(params, "linewidth", 1.0) or 1.0)
        spec["smoothing"] = float(getattr(params, "smoothing", 0.0) or 0.0)
    except (TypeError, ValueError):
        _add(
            diagnostics,
            "render_contour_style_unavailable",
            "Contour line width or smoothing could not be captured.",
            location,
        )
        return None
    if external:
        meta = getattr(state, "source_meta", None)
        if not isinstance(meta, Mapping) or meta.get("type") != "external_fits":
            _add(diagnostics, "unsupported_imported_contours", "Only external-FITS contour overlays can be reconstructed in this slice.", location)
            return None
        path = meta.get("path")
        if not isinstance(path, str) or not path.strip():
            _add(diagnostics, "external_contour_path_unavailable", "The external contour source path is missing.", location)
            return None
        spec["filepath"] = os.path.abspath(os.path.expanduser(path))
        if meta.get("channel") is not None:
            try:
                spec["channel"] = int(meta["channel"])
            except (TypeError, ValueError):
                _add(diagnostics, "invalid_external_contour_channel", "The external contour channel is invalid.", location)
                return None
        label = str(getattr(state, "label", "") or "")
        if label:
            spec["label"] = label
    return spec


def _capture_contours(
    window: Any,
    contour_manager: Any,
    diagnostics: list[PipelineDiagnostic],
) -> list[Dict[str, Any]]:
    layer_id = getattr(window, "_contour_layer_id", None)
    if not layer_id:
        return []
    if contour_manager is None:
        try:
            from takefits.core.contour_manager import ContourManager

            contour_manager = ContourManager.instance()
        except Exception:
            _add(diagnostics, "render_contour_manager_unavailable", "Contour state could not be accessed.", "render_snapshot.contours")
            return []
    layer = getattr(contour_manager, "_layers", {}).get(layer_id)
    if layer is None:
        _add(diagnostics, "render_contour_layer_unavailable", "The selected result's contour layer is missing.", "render_snapshot.contours")
        return []

    specs: list[Dict[str, Any]] = []
    try:
        generated = contour_manager.export_layer_state(layer_id)
    except Exception:
        generated = None
    spec = _contour_spec_from_state(
        generated,
        external=False,
        diagnostics=diagnostics,
        location="render_snapshot.contours.generated",
    )
    if spec is not None:
        specs.append(spec)
    try:
        overlays = list(layer.overlay_states())
    except Exception:
        overlays = []
    for index, state in enumerate(overlays):
        spec = _contour_spec_from_state(
            state,
            external=True,
            diagnostics=diagnostics,
            location=f"render_snapshot.contours.overlays[{index}]",
        )
        if spec is not None:
            specs.append(spec)
    return specs


def capture_moment_render_snapshot(
    window: Any,
    *,
    contour_manager: Any = None,
) -> MomentRenderSnapshotResult:
    """Capture one live XY integration result without retaining mutable objects."""
    diagnostics: list[PipelineDiagnostic] = []
    plane = str(getattr(window, "plane", "") or "").lower()
    if plane != "xy":
        _add(diagnostics, "unsupported_render_target", "Select an XY integration result window.", "render_snapshot.target.plane")
    result_tag = str(getattr(window, "_workspace_action_tag", "") or "").strip()
    if not result_tag:
        _add(diagnostics, "missing_render_result_tag", "The result window is not linked to a recorded Moment action.", "render_snapshot.target.result_tag")

    flush = getattr(window, "_flush_pending_annotation_commits", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            pass
    marker_getter = getattr(window, "_marker_specs_snapshot", None)
    region_getter = getattr(window, "_region_specs_snapshot", None)
    try:
        markers = list(marker_getter() if callable(marker_getter) else snapshot_marker_specs(getattr(window, "marker_manager", None)))
    except Exception:
        markers = []
        _add(diagnostics, "render_markers_unavailable", "Markers could not be serialized.", "render_snapshot.annotations.markers")
    try:
        regions = list(region_getter() if callable(region_getter) else snapshot_region_specs(getattr(window, "region_manager", None), default_plane=plane or "xy"))
    except Exception:
        regions = []
        _add(diagnostics, "render_regions_unavailable", "Regions could not be serialized.", "render_snapshot.annotations.regions")

    ax = getattr(window, "ax", None)
    xlim = ylim = None
    try:
        xlim = _pair(ax.get_xlim())
        ylim = _pair(ax.get_ylim())
    except Exception:
        pass
    if xlim is None or ylim is None:
        _add(diagnostics, "render_view_unavailable", "The selected result view limits could not be read.", "render_snapshot.view")

    controller = getattr(window, "displaymap", None)
    if bool(getattr(controller, "large_data_mode", False)):
        _add(
            diagnostics,
            "unsupported_large_data_render_snapshot",
            "Downsampled large-data result rendering is not yet reproducible by this snapshot slice.",
            "render_snapshot.target",
        )
    grid_visible = bool(getattr(controller, "grid_visible", getattr(window, "grid_visible", False)))
    grid_frame = str(
        getattr(controller, "grid_effective_frame", None)
        or getattr(controller, "grid_frame", "native")
        or "native"
    ).lower()
    grid_keep_native = bool(getattr(controller, "grid_keep_native", True))

    figure = getattr(window, "fig", None)
    figsize = None
    dpi = None
    try:
        figsize = _pair(figure.get_size_inches())
        dpi = int(round(float(figure.get_dpi())))
    except Exception:
        pass
    if figsize is None or dpi is None or dpi <= 0:
        _add(diagnostics, "render_figure_unavailable", "Figure dimensions or DPI could not be read.", "render_snapshot.figure")

    snapshot = {
        "schema": RENDER_SNAPSHOT_SCHEMA,
        "target": {
            "kind": "integration_result",
            "plane": plane,
            "result_tag": result_tag,
            "coordinate_basis": "source_xy_pixel",
        },
        "image": _capture_image(getattr(window, "im", None), diagnostics),
        "view": {"xlim": xlim, "ylim": ylim},
        "grid": {
            "visible": grid_visible,
            "frame": grid_frame,
            "keep_native": grid_keep_native,
        },
        "render_config": _capture_render_config(window, diagnostics),
        "annotations": {"markers": markers, "regions": regions},
        "contours": _capture_contours(window, contour_manager, diagnostics),
        "beam": {"visible": getattr(window, "hpbw", None) is not None},
        "figure": {"figsize": figsize, "dpi": dpi},
    }
    try:
        snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
    except (TypeError, ValueError) as exc:
        _add(diagnostics, "render_snapshot_not_json_compatible", f"Captured state is not JSON-compatible: {exc}", "render_snapshot")
    if any(item.severity == "error" for item in diagnostics):
        return MomentRenderSnapshotResult(None, tuple(diagnostics))
    return MomentRenderSnapshotResult(snapshot, tuple(diagnostics))


__all__ = ["MomentRenderSnapshotResult", "capture_moment_render_snapshot"]
