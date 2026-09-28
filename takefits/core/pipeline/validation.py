"""Normalize and validate takefits action documents without importing Qt.

The functions in this module accept an ActionRegistry-compatible object by
duck typing. This keeps the validation boundary usable by the CLI, GUI export
adapters, and future automation without importing the concrete registry or any
application/UI modules.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from takefits.core.actions import canonical_action_name
from takefits.core.spectral_records import document_spectral_unit


PATH_PARAM_KEYS = {
    "filepath",
    "output_path",
    "mask_path",
    "template_path",
    "data_b_path",
}
INPUT_FINGERPRINT_MODES = {"metadata", "sha256", "none"}


class ManifestValidationError(ValueError):
    """Raised when an action document or pipeline manifest is invalid."""


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    """Resolve a user path relative to the document that contains it."""
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def load_pipeline_document(path: Path) -> Any:
    """Load one JSON or YAML action document from ``path``."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _normalize_actions(raw_actions: Any, *, source_label: str) -> List[Dict[str, Any]]:
    if not isinstance(raw_actions, list):
        raise ManifestValidationError(
            f"{source_label} must be a list of action objects."
        )

    normalized: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw_actions, start=1):
        if not isinstance(entry, dict):
            raise ManifestValidationError(
                f"{source_label}[{index}] must be an object."
            )
        action_name = entry.get("action") or entry.get("name")
        if not isinstance(action_name, str) or not action_name.strip():
            raise ManifestValidationError(
                f"{source_label}[{index}] is missing a non-empty 'action' field."
            )
        params = entry.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ManifestValidationError(
                f"{source_label}[{index}].params must be an object."
            )
        normalized.append({"action": canonical_action_name(action_name.strip()), "params": dict(params)})
    return normalized


def _is_manifest_v1_document(doc: Any) -> bool:
    if not isinstance(doc, dict):
        return False
    return any(key in doc for key in ("version", "inputs", "outputs", "run_options"))


def _is_history_v1_document(doc: Any) -> bool:
    if not isinstance(doc, dict) or doc.get("version") != 1:
        return False
    return "history" in doc or "runs" in doc


def _validate_history_v1(doc: Dict[str, Any]) -> Dict[str, Any]:
    if "runs" in doc:
        raise ManifestValidationError(
            "Multi-run history cannot be replayed as one action pipeline; "
            "use the original batch manifest."
        )
    actions = _normalize_actions(doc.get("history"), source_label="history")
    return {
        "kind": "history_v1",
        "actions": actions,
        # A saved history without the record is in takefits's old units (TF-415 slice A).
        "spectral_unit": document_spectral_unit(doc),
        "spectral_unit_legacy": True,
        "inputs": [],
        "outputs": {},
        "run_options": {
            "stop_on_error": True,
            "input_fingerprint": "metadata",
        },
    }


def _validate_manifest_v1(doc: Dict[str, Any], *, source_path: Path) -> Dict[str, Any]:
    version = doc.get("version")
    if version != 1:
        raise ManifestValidationError(
            f"Manifest version must be 1 (got {version!r})."
        )

    actions = _normalize_actions(doc.get("actions"), source_label="manifest.actions")

    raw_inputs = doc.get("inputs", [])
    if raw_inputs is None:
        raw_inputs = []
    if not isinstance(raw_inputs, list):
        raise ManifestValidationError("manifest.inputs must be a list.")

    normalized_inputs: List[Dict[str, str]] = []
    for idx, item in enumerate(raw_inputs, start=1):
        if not isinstance(item, dict):
            raise ManifestValidationError(f"manifest.inputs[{idx}] must be an object.")
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ManifestValidationError(
                f"manifest.inputs[{idx}].path must be a non-empty string."
            )
        input_id = item.get("id")
        if input_id is None or not str(input_id).strip():
            input_id = f"input_{idx}"
        normalized_inputs.append(
            {
                "id": str(input_id),
                "path": str(resolve_path(source_path.parent, raw_path)),
            }
        )

    raw_outputs = doc.get("outputs", {})
    if raw_outputs is None:
        raw_outputs = {}
    if not isinstance(raw_outputs, dict):
        raise ManifestValidationError("manifest.outputs must be an object.")

    normalized_outputs: Dict[str, str] = {}
    for key in ("report_path", "history_path"):
        value = raw_outputs.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ManifestValidationError(
                f"manifest.outputs.{key} must be a non-empty string when present."
            )
        normalized_outputs[key] = str(resolve_path(source_path.parent, value))

    raw_run_options = doc.get("run_options", {})
    if raw_run_options is None:
        raw_run_options = {}
    if not isinstance(raw_run_options, dict):
        raise ManifestValidationError("manifest.run_options must be an object.")
    stop_on_error = raw_run_options.get("stop_on_error", True)
    if not isinstance(stop_on_error, bool):
        raise ManifestValidationError("manifest.run_options.stop_on_error must be boolean.")
    input_fingerprint = raw_run_options.get("input_fingerprint", "metadata")
    if not isinstance(input_fingerprint, str):
        raise ManifestValidationError(
            "manifest.run_options.input_fingerprint must be a string."
        )
    input_fingerprint = input_fingerprint.strip().lower()
    if input_fingerprint not in INPUT_FINGERPRINT_MODES:
        modes = ", ".join(sorted(INPUT_FINGERPRINT_MODES))
        raise ManifestValidationError(
            f"manifest.run_options.input_fingerprint must be one of: {modes}."
        )

    return {
        "kind": "manifest_v1",
        "actions": actions,
        # A manifest without the record is in the unit the cube is shown in.
        "spectral_unit": document_spectral_unit(doc),
        "spectral_unit_legacy": False,
        "inputs": normalized_inputs,
        "outputs": normalized_outputs,
        "run_options": {
            "stop_on_error": stop_on_error,
            "input_fingerprint": input_fingerprint,
        },
    }


def _validate_legacy_actions(doc: Any) -> Dict[str, Any]:
    if isinstance(doc, dict):
        if "actions" not in doc:
            raise ManifestValidationError(
                "Actions file must be a list or a dict with an 'actions' list."
            )
        actions = _normalize_actions(doc.get("actions"), source_label="actions")
    else:
        actions = _normalize_actions(doc, source_label="actions")

    return {
        "kind": "legacy",
        "actions": actions,
        "inputs": [],
        "outputs": {},
        "run_options": {
            "stop_on_error": True,
            "input_fingerprint": "metadata",
        },
    }


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    return True


def _validate_schema_value(
    value: Any,
    schema: Any,
    *,
    value_path: str,
    ignored_required: Optional[set[str]] = None,
) -> None:
    """Validate the JSON-Schema subset used by ActionRegistry definitions."""
    if not isinstance(schema, dict):
        return
    expected = schema.get("type")
    if isinstance(expected, str) and not _schema_type_matches(value, expected):
        raise ManifestValidationError(
            f"{value_path} must be {expected} (got {type(value).__name__})."
        )

    if "enum" in schema and value not in schema["enum"]:
        options = ", ".join(repr(item) for item in schema["enum"])
        raise ManifestValidationError(f"{value_path} must be one of: {options}.")

    if expected in ("integer", "number") and _schema_type_matches(value, expected):
        if "minimum" in schema and value < schema["minimum"]:
            raise ManifestValidationError(
                f"{value_path} must be >= {schema['minimum']}."
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise ManifestValidationError(
                f"{value_path} must be <= {schema['maximum']}."
            )

    if expected == "array" and isinstance(value, (list, tuple)):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ManifestValidationError(
                f"{value_path} must contain at least {schema['minItems']} items."
            )
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ManifestValidationError(
                f"{value_path} must contain at most {schema['maxItems']} items."
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_value(
                    item,
                    item_schema,
                    value_path=f"{value_path}[{index}]",
                )

    if expected == "object" and isinstance(value, dict):
        ignored = ignored_required or set()
        for required_name in schema.get("required", []) or []:
            if required_name not in value and required_name not in ignored:
                raise ManifestValidationError(
                    f"{value_path} is missing required parameter "
                    f"{required_name!r}."
                )
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, item in value.items():
                item_schema = properties.get(key)
                if item_schema is not None:
                    _validate_schema_value(
                        item,
                        item_schema,
                        value_path=f"{value_path}.{key}",
                    )
                elif schema.get("additionalProperties") is False:
                    raise ManifestValidationError(
                        f"{value_path} has unsupported parameter {key!r}."
                    )


def validate_registered_actions(config: Dict[str, Any], registry: Any) -> None:
    """Validate normalized action entries against a registry-like object."""
    implicit_input_load = bool(config.get("inputs"))
    if config["kind"] == "manifest_v1":
        label = "manifest.actions"
    elif config["kind"] == "history_v1":
        label = "history"
    else:
        label = "actions"
    for index, entry in enumerate(config["actions"], start=1):
        action_name = entry["action"]
        action = registry.get_action(action_name)
        if action is None:
            raise ManifestValidationError(f"Action '{action_name}' not found.")
        if not action.cli_supported:
            raise ManifestValidationError(
                f"Action '{action_name}' is GUI-only and cannot run from the CLI."
            )
        ignored_required: set[str] = set()
        if implicit_input_load and action_name == "load_fits":
            ignored_required.add("filepath")
        _validate_schema_value(
            entry.get("params", {}),
            action.parameters,
            value_path=f"{label}[{index}].params",
            ignored_required=ignored_required,
        )


def validate_pipeline_document(
    document: Any,
    *,
    source_path: Path,
    registry: Optional[Any] = None,
) -> Dict[str, Any]:
    """Normalize and validate an in-memory pipeline/action document.

    ``source_path`` establishes the base directory for relative manifest input
    and report paths. When ``registry`` is provided, action availability, CLI
    support, and the ActionRegistry JSON-Schema subset are also checked.
    """
    source_path = Path(source_path)
    if _is_history_v1_document(document):
        config = _validate_history_v1(document)
    elif _is_manifest_v1_document(document):
        config = _validate_manifest_v1(document, source_path=source_path)
    else:
        config = _validate_legacy_actions(document)
    if registry is not None:
        validate_registered_actions(config, registry)
    return config


def load_and_validate(
    path: Path,
    registry: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load, normalize, and validate a JSON/YAML pipeline document."""
    path = Path(path)
    return validate_pipeline_document(
        load_pipeline_document(path),
        source_path=path,
        registry=registry,
    )
