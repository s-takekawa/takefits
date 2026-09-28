"""Build a bounded, runnable manifest for one recorded Moment-0 result."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from takefits.core.spectral_records import document_spectral_unit
from takefits.core.spectral_units import SPECTRAL_UNIT_KEY

from .diagnostics import PipelineDiagnostic
from .render_snapshot import build_moment_render_plan
from .validation import ManifestValidationError, validate_pipeline_document


_MOMENT_PARAMS = {
    "moment_type",
    "axis",
    "clip_threshold",
    "pixel_range",
    "world_range",
}
_GUI_MOMENT_METADATA = {"_window_action_tag"}
_OUTPUT_TARGET_KEYS = {
    "fits_path",
    "image_path",
    "image_params",
    "report_path",
    "history_path",
}
_IMAGE_RESERVED_PARAMS = _MOMENT_PARAMS | {"output_path"}
_PASSIVE_CONTEXT_ACTIONS = {"set_cursor", "set_slice"}
_RENDER_CONTEXT_ACTIONS = {
    "set_color_settings",
    "set_view_range",
    "set_coordinate_grid",
    "set_render_config",
    "clear_render_config",
    "set_contours",
    "add_contour",
    "clear_contours",
    "set_regions",
    "clear_regions",
    "add_region",
    "update_region",
    "delete_region",
    "set_markers",
    "clear_markers",
    "add_marker",
    "update_marker",
    "delete_marker",
}


@dataclass(frozen=True)
class MomentManifestResult:
    """Result of converting one selected GUI Moment-0 record."""

    manifest: Optional[Dict[str, Any]]
    diagnostics: Tuple[PipelineDiagnostic, ...]
    selected_record_index: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.manifest is not None and not any(
            diagnostic.severity == "error" for diagnostic in self.diagnostics
        )


def _diagnostic(
    diagnostics: list[PipelineDiagnostic],
    severity: str,
    code: str,
    message: str,
    location: str = "",
) -> None:
    diagnostics.append(
        PipelineDiagnostic(
            severity=severity,
            code=code,
            message=message,
            location=location,
        )
    )


def _has_errors(diagnostics: Sequence[PipelineDiagnostic]) -> bool:
    return any(diagnostic.severity == "error" for diagnostic in diagnostics)


def _document_absolute_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return Path(os.path.abspath(os.fspath(path)))


def _portable_document_path(raw_path: str, base_dir: Path) -> str:
    absolute = _document_absolute_path(raw_path, base_dir)
    try:
        # Forward slashes keep a manifest written on Windows runnable elsewhere.
        return Path(os.path.relpath(absolute, start=base_dir)).as_posix()
    except ValueError:
        return str(absolute)


def _load_params(source: Mapping[str, Any]) -> Dict[str, Any]:
    """The manifest's load_fits parameters: HDU 0 and the spectral-axis mode the GUI opened with."""
    params: Dict[str, Any] = {"hdu": 0}
    mode = source.get("spectral_axis") if isinstance(source, Mapping) else None
    frequency_axis = mode.get("frequency_axis") if isinstance(mode, Mapping) else None
    if frequency_axis == "frequency":  # velocity is load_fits's default
        params["frequency_axis"] = frequency_axis
    return params


def _record_payload(
    value: Any,
    *,
    index: int,
    diagnostics: list[PipelineDiagnostic],
) -> Optional[Dict[str, Any]]:
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict()
        except Exception as exc:
            _diagnostic(
                diagnostics,
                "error",
                "invalid_action_record",
                f"Could not serialize action record {index}: {exc}",
                f"history[{index}]",
            )
            return None
    if not isinstance(value, Mapping):
        _diagnostic(
            diagnostics,
            "error",
            "invalid_action_record",
            f"Action record {index} must be an object.",
            f"history[{index}]",
        )
        return None
    action = value.get("action") or value.get("name")
    if not isinstance(action, str) or not action.strip():
        _diagnostic(
            diagnostics,
            "error",
            "invalid_action_record",
            f"Action record {index} has no action name.",
            f"history[{index}]",
        )
        return None
    params = value.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        _diagnostic(
            diagnostics,
            "error",
            "invalid_action_params",
            f"Action record {index} params must be an object.",
            f"history[{index}].params",
        )
        return None
    record_tag = str(value.get("tag") or "").strip()
    param_tag = str(params.get("_window_action_tag") or "").strip()
    if record_tag and param_tag and record_tag != param_tag:
        _diagnostic(
            diagnostics,
            "error",
            "inconsistent_moment_tag",
            f"Action record {index} has different record and window-result tags.",
            f"history[{index}]",
        )
        return None
    return {
        "action": action.strip(),
        "params": dict(params),
        "tag": value.get("tag"),
    }


def _record_result_tag(record: Mapping[str, Any]) -> str:
    params = record.get("params", {})
    param_tag = params.get("_window_action_tag") if isinstance(params, Mapping) else None
    return str(record.get("tag") or param_tag or "").strip()


def _validate_source(
    source: Any,
    *,
    manifest_dir: Path,
    diagnostics: list[PipelineDiagnostic],
) -> Optional[Path]:
    if not isinstance(source, Mapping):
        _diagnostic(
            diagnostics,
            "error",
            "invalid_source",
            "Source descriptor must be an object.",
            "source",
        )
        return None

    filepath = source.get("filepath")
    if not isinstance(filepath, str) or not filepath.strip():
        _diagnostic(
            diagnostics,
            "error",
            "missing_source_path",
            "Source descriptor needs a non-empty filepath.",
            "source.filepath",
        )
        source_path = None
    else:
        source_path = _document_absolute_path(filepath, manifest_dir)
        if not source_path.is_file():
            _diagnostic(
                diagnostics,
                "error",
                "source_not_found",
                f"Source FITS file does not exist: {source_path}",
                "source.filepath",
            )

    hdu_index = source.get("hdu_index")
    if hdu_index is None:
        _diagnostic(
            diagnostics,
            "error",
            "source_hdu_unproven",
            "The selected HDU is not recorded; re-save the Recipe before exporting.",
            "source.hdu_index",
        )
    elif isinstance(hdu_index, bool) or not isinstance(hdu_index, int):
        _diagnostic(
            diagnostics,
            "error",
            "invalid_source_hdu",
            "source.hdu_index must be an integer.",
            "source.hdu_index",
        )
    elif hdu_index != 0:
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_source_hdu",
            "The first TF-301 slice supports only primary HDU 0.",
            "source.hdu_index",
        )

    shape = source.get("data_shape")
    valid_shape = (
        isinstance(shape, (list, tuple))
        and len(shape) == 3
        and all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in shape
        )
    )
    if not valid_shape:
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_source_shape",
            "The first TF-301 slice requires a recorded three-dimensional data shape.",
            "source.data_shape",
        )

    if source.get("wcs_naxis") != 3:
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_source_wcs",
            "The first TF-301 slice requires a three-axis WCS.",
            "source.wcs_naxis",
        )

    signature = source.get("wcs_signature")
    family = (
        str(signature.get("celestial_family") or "").strip().lower()
        if isinstance(signature, Mapping)
        else ""
    )
    if family not in {"equatorial", "galactic"}:
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_celestial_wcs",
            "The selected Moment-0 result must have equatorial or galactic XY WCS.",
            "source.wcs_signature.celestial_family",
        )

    return source_path


def _validate_leading_load(
    record: Mapping[str, Any],
    *,
    source_path: Optional[Path],
    manifest_dir: Path,
    index: int,
    diagnostics: list[PipelineDiagnostic],
) -> None:
    params = record.get("params", {})
    filepath = params.get("filepath")
    if filepath is not None:
        if not isinstance(filepath, str) or not filepath.strip():
            _diagnostic(
                diagnostics,
                "error",
                "invalid_lineage_load",
                "A recorded load_fits filepath must be a non-empty string.",
                f"history[{index}].params.filepath",
            )
        elif source_path is not None:
            loaded_path = _document_absolute_path(filepath, manifest_dir)
            if loaded_path != source_path:
                _diagnostic(
                    diagnostics,
                    "error",
                    "source_lineage_mismatch",
                    "The recorded load_fits path differs from the Recipe source.",
                    f"history[{index}].params.filepath",
                )
    hdu = params.get("hdu")
    if hdu not in (None, 0):
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_lineage_hdu",
            "A recorded load_fits action must select primary HDU 0.",
            f"history[{index}].params.hdu",
        )


def _validate_lineage(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_index: int,
    source_path: Optional[Path],
    manifest_dir: Path,
    image_requested: bool,
    render_snapshot_supplied: bool,
    diagnostics: list[PipelineDiagnostic],
) -> None:
    for index, record in enumerate(records[:selected_index]):
        action = str(record.get("action") or "")
        if action == "load_fits":
            _validate_leading_load(
                record,
                source_path=source_path,
                manifest_dir=manifest_dir,
                index=index,
                diagnostics=diagnostics,
            )
        elif action == "compute_moment":
            _diagnostic(
                diagnostics,
                "info",
                "ignored_earlier_moment",
                "An earlier Moment result is outside the selected result lineage.",
                f"history[{index}]",
            )
        elif action in _PASSIVE_CONTEXT_ACTIONS or action.startswith("export_"):
            _diagnostic(
                diagnostics,
                "info",
                "ignored_context_action",
                f"Context-only action {action!r} does not affect this Moment computation.",
                f"history[{index}]",
            )
        elif action in _RENDER_CONTEXT_ACTIONS:
            severity = (
                "warning"
                if image_requested and not render_snapshot_supplied
                else "info"
            )
            code = (
                "render_context_superseded_by_snapshot"
                if render_snapshot_supplied
                else "render_context_not_captured"
            )
            detail = (
                "the selected result's effective render snapshot supersedes it."
                if render_snapshot_supplied
                else "the result-render snapshot package will capture it."
            )
            _diagnostic(
                diagnostics,
                severity,
                code,
                f"Render action {action!r} is not part of the numerical export plan; "
                f"{detail}",
                f"history[{index}]",
            )
        else:
            _diagnostic(
                diagnostics,
                "error",
                "unsupported_lineage_action",
                f"Action {action!r} occurs before the selected Moment result and may "
                "change its data lineage.",
                f"history[{index}]",
            )


def _target_path(
    target: Mapping[str, Any],
    key: str,
    *,
    diagnostics: list[PipelineDiagnostic],
) -> Optional[str]:
    value = target.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        _diagnostic(
            diagnostics,
            "error",
            "invalid_output_path",
            f"{key} must be a non-empty string when present.",
            f"output_target.{key}",
        )
        return None
    return value


def _active_records(
    operation_records: Any,
    *,
    cursor: Optional[int],
    diagnostics: list[PipelineDiagnostic],
) -> Optional[list[Dict[str, Any]]]:
    """Normalize the records before the active history cursor.

    Returns ``None`` when the history or one of its records is malformed.
    """
    if not isinstance(operation_records, Sequence) or isinstance(
        operation_records, (str, bytes)
    ):
        _diagnostic(
            diagnostics,
            "error",
            "invalid_history",
            "operation_records must be a sequence of action records.",
            "history",
        )
        return None

    total_records = len(operation_records)
    if cursor is None:
        active_cursor = total_records
    elif isinstance(cursor, bool) or not isinstance(cursor, int):
        _diagnostic(
            diagnostics,
            "error",
            "invalid_history_cursor",
            "cursor must be an integer.",
            "cursor",
        )
        active_cursor = total_records
    elif cursor < 0 or cursor > total_records:
        _diagnostic(
            diagnostics,
            "error",
            "invalid_history_cursor",
            f"cursor must be between 0 and {total_records}.",
            "cursor",
        )
        active_cursor = total_records
    else:
        active_cursor = cursor

    records: list[Dict[str, Any]] = []
    record_diagnostic_start = len(diagnostics)
    for index, raw_record in enumerate(operation_records[:active_cursor]):
        normalized = _record_payload(
            raw_record,
            index=index,
            diagnostics=diagnostics,
        )
        if normalized is not None:
            records.append(normalized)

    if _has_errors(diagnostics[record_diagnostic_start:]):
        return None
    return records


def _select_moment_record(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_action_tag: Optional[str],
    diagnostics: list[PipelineDiagnostic],
) -> Optional[int]:
    """Index of the selected ``compute_moment`` record, or ``None``."""
    moment_indexes = [
        index
        for index, record in enumerate(records)
        if record.get("action") == "compute_moment"
    ]
    selected_tag = str(selected_action_tag or "").strip()
    if selected_tag:
        selected_matches = [
            index
            for index in moment_indexes
            if _record_result_tag(records[index]) == selected_tag
        ]
        if not selected_matches:
            _diagnostic(
                diagnostics,
                "error",
                "selected_moment_not_found",
                f"No active Moment result has tag {selected_tag!r}.",
                "selected_action_tag",
            )
            return None
        if len(selected_matches) > 1:
            _diagnostic(
                diagnostics,
                "error",
                "duplicate_moment_tag",
                f"More than one active Moment result has tag {selected_tag!r}.",
                "selected_action_tag",
            )
            return None
        return selected_matches[0]
    if len(moment_indexes) == 1:
        return moment_indexes[0]
    if not moment_indexes:
        _diagnostic(
            diagnostics,
            "error",
            "moment_result_not_found",
            "No active compute_moment record is available.",
            "history",
        )
        return None
    _diagnostic(
        diagnostics,
        "error",
        "ambiguous_moment_result",
        "Multiple active Moment results exist; provide selected_action_tag.",
        "selected_action_tag",
    )
    return None


def _validate_selected_record(
    raw_params: Mapping[str, Any],
    *,
    selected_index: int,
    diagnostics: list[PipelineDiagnostic],
) -> None:
    unknown_params = sorted(
        set(raw_params) - _MOMENT_PARAMS - _GUI_MOMENT_METADATA
    )
    for key in unknown_params:
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_moment_parameter",
            f"Recorded Moment parameter {key!r} is not supported by this converter.",
            f"history[{selected_index}].params.{key}",
        )

    moment_type = str(raw_params.get("moment_type") or "").strip().lower()
    if moment_type != "moment0":
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_moment_type",
            "The first TF-301 converter slice supports only moment_type='moment0' "
            "(the Integration mode); this result is "
            f"{raw_params.get('moment_type')!r}.",
            f"history[{selected_index}].params.moment_type",
        )
    axis = raw_params.get("axis", 0)
    if isinstance(axis, bool) or not isinstance(axis, int) or axis != 0:
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_moment_axis",
            "The selected Moment-0 result must integrate spectral axis 0 "
            f"(an XY result); this result uses axis={axis!r}.",
            f"history[{selected_index}].params.axis",
        )
    if raw_params.get("pixel_range") is None and raw_params.get("world_range") is None:
        _diagnostic(
            diagnostics,
            "error",
            "missing_moment_range",
            "The selected Moment-0 record must contain an explicit pixel_range or world_range.",
            f"history[{selected_index}].params",
        )


def build_moment_export_manifest(
    source: Mapping[str, Any],
    operation_records: Sequence[Any],
    output_target: Mapping[str, Any],
    *,
    manifest_path: Path,
    registry: Any,
    selected_action_tag: Optional[str] = None,
    cursor: Optional[int] = None,
    render_snapshot: Optional[Mapping[str, Any]] = None,
) -> MomentManifestResult:
    """Convert one supported GUI Moment-0 result into a validated manifest.

    This first TF-301 converter slice supports an unmodified primary-HDU 3-D
    cube, a celestial XY Moment-0 result on spectral axis 0, and an explicit
    pixel or world integration range. ``clip_threshold`` is optional but, when
    recorded, is copied unchanged to compute, FITS export, and image export.
    Unsupported or unprovable lineage returns diagnostics and no manifest.
    """
    diagnostics: list[PipelineDiagnostic] = []
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent

    source_path = _validate_source(
        source,
        manifest_dir=manifest_dir,
        diagnostics=diagnostics,
    )

    records = _active_records(
        operation_records,
        cursor=cursor,
        diagnostics=diagnostics,
    )
    if records is None:
        return MomentManifestResult(None, tuple(diagnostics))

    selected_index = _select_moment_record(
        records,
        selected_action_tag=selected_action_tag,
        diagnostics=diagnostics,
    )
    if selected_index is None:
        return MomentManifestResult(None, tuple(diagnostics))

    selected_record = records[selected_index]
    raw_params = selected_record.get("params", {})
    _validate_selected_record(
        raw_params,
        selected_index=selected_index,
        diagnostics=diagnostics,
    )

    if not isinstance(output_target, Mapping):
        _diagnostic(
            diagnostics,
            "error",
            "invalid_output_target",
            "output_target must be an object.",
            "output_target",
        )
        return MomentManifestResult(None, tuple(diagnostics), selected_index)
    for key in sorted(set(output_target) - _OUTPUT_TARGET_KEYS):
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_output_option",
            f"Output option {key!r} is not supported.",
            f"output_target.{key}",
        )

    fits_path = _target_path(output_target, "fits_path", diagnostics=diagnostics)
    image_path = _target_path(output_target, "image_path", diagnostics=diagnostics)
    report_path = _target_path(output_target, "report_path", diagnostics=diagnostics)
    history_path = _target_path(output_target, "history_path", diagnostics=diagnostics)
    if fits_path is None and image_path is None:
        _diagnostic(
            diagnostics,
            "error",
            "missing_output_target",
            "At least one of fits_path or image_path is required.",
            "output_target",
        )

    if fits_path is not None and Path(fits_path).suffix.lower() not in {
        ".fit",
        ".fits",
        ".fts",
    }:
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_fits_extension",
            "fits_path must use a .fit, .fits, or .fts extension.",
            "output_target.fits_path",
        )
    if image_path is not None and Path(image_path).suffix.lower() not in {
        ".pdf",
        ".png",
    }:
        _diagnostic(
            diagnostics,
            "error",
            "unsupported_image_extension",
            "image_path must use a .png or .pdf extension.",
            "output_target.image_path",
        )

    image_params = output_target.get("image_params", {})
    if image_params is None:
        image_params = {}
    if not isinstance(image_params, Mapping):
        _diagnostic(
            diagnostics,
            "error",
            "invalid_image_params",
            "image_params must be an object.",
            "output_target.image_params",
        )
        image_params = {}
    for key in sorted(set(image_params) & _IMAGE_RESERVED_PARAMS):
        _diagnostic(
            diagnostics,
            "error",
            "reserved_image_parameter",
            f"image_params cannot override recorded scientific parameter {key!r}.",
            f"output_target.image_params.{key}",
        )

    output_paths = {
        key: path
        for key, path in {
            "fits_path": fits_path,
            "image_path": image_path,
            "report_path": report_path,
            "history_path": history_path,
        }.items()
        if path is not None
    }
    resolved_outputs = {
        key: _document_absolute_path(path, manifest_dir)
        for key, path in output_paths.items()
    }
    if source_path is not None and source_path in resolved_outputs.values():
        _diagnostic(
            diagnostics,
            "error",
            "output_overwrites_source",
            "An export output must not overwrite the source FITS file.",
            "output_target",
        )
    manifest_absolute = Path(os.path.abspath(os.fspath(manifest_path)))
    if manifest_absolute in resolved_outputs.values():
        _diagnostic(
            diagnostics,
            "error",
            "output_overwrites_manifest",
            "An export output must not overwrite its pipeline manifest.",
            "output_target",
        )
    if len(resolved_outputs) != len(set(resolved_outputs.values())):
        _diagnostic(
            diagnostics,
            "error",
            "duplicate_output_path",
            "Every export, report, and history output must use a different path.",
            "output_target",
        )

    _validate_lineage(
        records,
        selected_index=selected_index,
        source_path=source_path,
        manifest_dir=manifest_dir,
        image_requested=image_path is not None,
        render_snapshot_supplied=render_snapshot is not None,
        diagnostics=diagnostics,
    )
    if selected_index + 1 < len(records):
        _diagnostic(
            diagnostics,
            "info",
            "ignored_post_selection_records",
            f"Ignored {len(records) - selected_index - 1} record(s) after the "
            "selected result creation point.",
            f"history[{selected_index + 1}:]",
        )
    render_actions: list[Dict[str, Any]] = []
    render_image_params: Dict[str, Any] = {}
    if render_snapshot is not None and image_path is None:
        _diagnostic(
            diagnostics,
            "info",
            "unused_render_snapshot",
            "The render snapshot is unused because no image output was requested.",
            "render_snapshot",
        )
    elif render_snapshot is not None:
        render_plan = build_moment_render_plan(
            render_snapshot,
            manifest_dir=manifest_dir,
            selected_result_tag=_record_result_tag(selected_record),
        )
        diagnostics.extend(render_plan.diagnostics)
        if render_plan.ok:
            render_actions = [dict(action) for action in render_plan.actions]
            render_image_params = dict(render_plan.image_params)
    elif image_path is not None:
        _diagnostic(
            diagnostics,
            "warning",
            "render_snapshot_pending",
            "The manifest reproduces the Moment calculation, but full selected-window "
            "render styling and overlays require the next render-snapshot package.",
            "output_target.image_path",
        )

    if _has_errors(diagnostics) or source_path is None:
        return MomentManifestResult(None, tuple(diagnostics), selected_index)

    moment_params = {
        key: raw_params[key]
        for key in _MOMENT_PARAMS
        if key in raw_params and raw_params[key] is not None
    }
    actions = [
        {"action": "load_fits", "params": _load_params(source)},
    ]
    actions.extend(render_actions)
    actions.append({"action": "compute_moment", "params": dict(moment_params)})
    if fits_path is not None:
        actions.append(
            {
                "action": "export_moment_fits",
                "params": {
                    "output_path": _portable_document_path(fits_path, manifest_dir),
                    **moment_params,
                },
            }
        )
    if image_path is not None:
        actions.append(
            {
                "action": "export_moment_image",
                "params": {
                    "output_path": _portable_document_path(image_path, manifest_dir),
                    **moment_params,
                    **render_image_params,
                    **dict(image_params),
                },
            }
        )

    manifest: Dict[str, Any] = {
        "version": 1,
        "inputs": [
            {
                "id": "source",
                "path": _portable_document_path(str(source_path), manifest_dir),
            }
        ],
        "actions": actions,
        "run_options": {
            "stop_on_error": True,
            "input_fingerprint": "metadata",
        },
    }
    spectral_unit = document_spectral_unit({"source": dict(source)}) if isinstance(source, Mapping) else None
    if spectral_unit:
        manifest[SPECTRAL_UNIT_KEY] = spectral_unit  # the unit of world_range
    outputs = {}
    if report_path is not None:
        outputs["report_path"] = _portable_document_path(report_path, manifest_dir)
    if history_path is not None:
        outputs["history_path"] = _portable_document_path(history_path, manifest_dir)
    if outputs:
        manifest["outputs"] = outputs

    try:
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
    except (TypeError, ValueError) as exc:
        _diagnostic(
            diagnostics,
            "error",
            "manifest_not_json_compatible",
            f"Generated manifest values are not JSON-compatible: {exc}",
            "manifest",
        )
        return MomentManifestResult(None, tuple(diagnostics), selected_index)

    try:
        validate_pipeline_document(
            manifest,
            source_path=manifest_path,
            registry=registry,
        )
    except ManifestValidationError as exc:
        _diagnostic(
            diagnostics,
            "error",
            "generated_manifest_invalid",
            str(exc),
            "manifest",
        )
        return MomentManifestResult(None, tuple(diagnostics), selected_index)

    return MomentManifestResult(manifest, tuple(diagnostics), selected_index)


def check_moment_result_support(
    source: Mapping[str, Any],
    operation_records: Sequence[Any],
    *,
    selected_action_tag: Optional[str] = None,
    cursor: Optional[int] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[PipelineDiagnostic, ...]:
    """Return the errors that stop one recorded Moment result from exporting.

    This applies the source, record, and lineage rules of
    :func:`build_moment_export_manifest` without output paths or render state,
    so a GUI can mark an unsupported result before asking where to save it.
    An empty tuple means the result is exportable. Relative paths resolve
    against ``base_dir`` (default: the current directory).
    """
    diagnostics: list[PipelineDiagnostic] = []
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    source_path = _validate_source(source, manifest_dir=base, diagnostics=diagnostics)
    records = _active_records(operation_records, cursor=cursor, diagnostics=diagnostics)
    if records is not None:
        selected_index = _select_moment_record(
            records,
            selected_action_tag=selected_action_tag,
            diagnostics=diagnostics,
        )
        if selected_index is not None:
            _validate_selected_record(
                records[selected_index].get("params", {}),
                selected_index=selected_index,
                diagnostics=diagnostics,
            )
            _validate_lineage(
                records,
                selected_index=selected_index,
                source_path=source_path,
                manifest_dir=base,
                image_requested=False,
                render_snapshot_supplied=False,
                diagnostics=diagnostics,
            )
    return tuple(item for item in diagnostics if item.severity == "error")
