"""Action session runner for headless/CLI execution."""
from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from takefits.core.actions import ActionRegistry
from takefits.core.app_state import AppState

_USE_CURRENT_STATE = object()


@dataclass
class ActionRecord:
    action: str
    params: Dict[str, Any]
    timestamp: str
    tag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action": self.action,
            "params": dict(self.params),
            "timestamp": self.timestamp,
        }
        if self.tag is not None:
            payload["tag"] = self.tag
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActionRecord":
        if not isinstance(payload, dict):
            raise TypeError("ActionRecord payload must be a mapping")
        action = payload.get("action") or payload.get("name")
        if not action:
            raise ValueError("ActionRecord payload is missing 'action'/'name'")
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("ActionRecord 'params' must be an object")
        timestamp = str(payload.get("timestamp") or _utc_timestamp())
        tag = payload.get("tag")
        return cls(
            action=str(action),
            params=dict(params),
            timestamp=timestamp,
            tag=str(tag) if tag is not None else None,
        )


@dataclass
class ActionSession:
    registry: ActionRegistry
    state: Optional[AppState] = None
    history: List[ActionRecord] = field(default_factory=list)
    last_result: Any = None
    defer_initial_state_seed: bool = False
    _initial_state_seed: Optional[AppState] = field(default=None, init=False, repr=False)
    _initial_seed_set: bool = field(default=False, init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.defer_initial_state_seed:
            self.set_initial_state_seed()
        self._cursor = len(self.history)

    def execute(
        self,
        name: str,
        *,
        record_only: bool = False,
        replace_tag: Optional[str] = None,
        **params: Any,
    ) -> Any:
        if not self._initial_seed_set and self.state is not None:
            self.set_initial_state_seed()

        result: Any = None
        if not record_only:
            result = self._execute_handler(name=name, params=params)

        self._append_record(name=name, params=params, tag=replace_tag)
        return result

    def record(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        replace_tag: Optional[str] = None,
    ) -> None:
        """Record an action without executing its handler."""
        self.execute(
            name,
            record_only=True,
            replace_tag=replace_tag,
            **(params or {}),
        )

    def remove_record_by_tag(self, tag: str) -> bool:
        """Remove the most recent history record with the provided tag."""
        for idx in range(len(self.history) - 1, -1, -1):
            if self.history[idx].tag == tag:
                del self.history[idx]
                if idx < self._cursor:
                    self._cursor -= 1
                return True
        return False

    def run_actions(self, actions: List[Dict[str, Any]]) -> List[Any]:
        results = []
        for entry in actions:
            name = entry.get("action") or entry.get("name")
            if not name:
                raise ValueError("Action entry missing 'action' or 'name' field.")
            params = entry.get("params", {})
            if params is None:
                params = {}
            results.append(self.execute(name, **params))
        return results

    def export_history(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.history]

    def save_history(self, path: str) -> str:
        payload = {
            "version": 1,
            "saved_at": _utc_timestamp(),
            "history": self.export_history(),
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(output_path)

    def load_history(
        self,
        path: str,
        *,
        replay: bool = False,
        replace: bool = True,
    ) -> List[ActionRecord]:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            entries = loaded.get("history", [])
        elif isinstance(loaded, list):
            entries = loaded
        else:
            raise ValueError("History file must be a list or an object with 'history'")

        records = [ActionRecord.from_dict(entry) for entry in entries]

        if not replay:
            if replace:
                self.history = list(records)
            else:
                if self._cursor < len(self.history):
                    del self.history[self._cursor :]
                self.history.extend(records)
            self._cursor = len(self.history)
            return records

        if replace:
            self.history = []
            self._cursor = 0
            if not self._initial_seed_set and self.state is not None:
                self.set_initial_state_seed()
            self._restore_initial_state()
        elif self._cursor < len(self.history):
            del self.history[self._cursor :]

        for record in records:
            self.execute(
                record.action,
                replace_tag=record.tag,
                **dict(record.params),
            )
            if self.history:
                self.history[-1].timestamp = record.timestamp

        self._cursor = len(self.history)
        return list(self.history)

    def set_initial_state_seed(self, state: Any = _USE_CURRENT_STATE) -> None:
        source = self.state if state is _USE_CURRENT_STATE else state
        self._initial_state_seed = _clone_state(source) if source is not None else None
        self._initial_seed_set = True

    def reset_to_initial(self, *, keep_history: bool = True) -> Optional[AppState]:
        self._restore_initial_state()
        self._cursor = 0
        if not keep_history:
            self.history = []
        return self.state

    @property
    def cursor(self) -> int:
        return self._cursor

    def can_undo(self) -> bool:
        return self._cursor > 0

    def can_redo(self) -> bool:
        return self._cursor < len(self.history)

    def undo(self, steps: int = 1) -> Optional[AppState]:
        steps_int = int(steps)
        if steps_int < 1:
            raise ValueError("undo steps must be >= 1")
        target_cursor = max(0, self._cursor - steps_int)
        return self._replay_to_cursor(target_cursor)

    def redo(self, steps: int = 1) -> Optional[AppState]:
        steps_int = int(steps)
        if steps_int < 1:
            raise ValueError("redo steps must be >= 1")
        target_cursor = min(len(self.history), self._cursor + steps_int)
        return self._replay_to_cursor(target_cursor)

    def _restore_initial_state(self) -> None:
        if not self._initial_seed_set:
            if self.state is not None:
                self.set_initial_state_seed()
            else:
                raise ValueError("Initial state seed is not set.")
        if self._initial_state_seed is None:
            self.state = None
            self.last_result = None
            return
        self.state = _clone_state(self._initial_state_seed)
        self.last_result = self.state

    def _replay_to_cursor(self, target_cursor: int) -> Optional[AppState]:
        if target_cursor < 0 or target_cursor > len(self.history):
            raise ValueError("Replay target is out of range.")
        self._restore_initial_state()
        for record in self.history[:target_cursor]:
            self._execute_handler(name=record.action, params=dict(record.params))
        self._cursor = target_cursor
        return self.state

    def _append_record(self, name: str, params: Dict[str, Any], tag: Optional[str]) -> None:
        if self._cursor < len(self.history):
            del self.history[self._cursor :]
        if tag:
            self.remove_record_by_tag(tag)
        self.history.append(
            ActionRecord(
                action=name,
                params=dict(params),
                timestamp=_utc_timestamp(),
                tag=tag,
            )
        )
        self._cursor = len(self.history)

    def _execute_handler(self, name: str, params: Dict[str, Any]) -> Any:
        action = self.registry.get_action(name)
        if not action:
            raise ValueError(f"Action '{name}' not found.")

        handler = action.handler
        accepts_state = _handler_accepts_state(handler)
        accepts_result = _handler_accepts_result(handler)
        call_params = _filter_handler_kwargs(handler, params)

        previous_result = self.last_result

        if accepts_state and accepts_result:
            if self.state is None:
                raise ValueError(f"Action '{name}' requires state, but no state is loaded.")
            if self.last_result is None:
                raise ValueError(f"Action '{name}' requires a previous result, but none is available.")
            result = handler(state=self.state, result=self.last_result, **call_params)
        elif accepts_state:
            if self.state is None:
                raise ValueError(f"Action '{name}' requires state, but no state is loaded.")
            result = handler(state=self.state, **call_params)
        elif accepts_result:
            if self.last_result is None:
                raise ValueError(f"Action '{name}' requires a previous result, but none is available.")
            result = handler(result=self.last_result, **call_params)
        else:
            result = handler(**call_params)

        if isinstance(result, AppState):
            self.state = result
        elif result is None and accepts_state:
            # In-place mutation pattern
            result = self.state

        if _should_preserve_last_result(name, result, previous_result):
            self.last_result = previous_result
        else:
            self.last_result = result
        return result


def _handler_accepts_state(handler: Any) -> bool:
    try:
        sig = _handler_signature(handler)
    except (TypeError, ValueError):
        return False
    return "state" in sig.parameters


def _handler_accepts_result(handler: Any) -> bool:
    try:
        sig = _handler_signature(handler)
    except (TypeError, ValueError):
        return False
    return "result" in sig.parameters


def _filter_handler_kwargs(handler: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only keyword arguments accepted by the handler.

    This lets history records carry UI-only metadata without breaking replay.
    """
    if not isinstance(params, dict):
        return {}
    try:
        sig = _handler_signature(handler)
    except (TypeError, ValueError):
        return dict(params)

    values = list(sig.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in values):
        return dict(params)

    allowed: set[str] = set()
    for param in values:
        if param.name in {"state", "result"}:
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(param.name)
    return {key: value for key, value in params.items() if key in allowed}


def _handler_signature(handler: Any) -> inspect.Signature:
    lazy_name = getattr(handler, "_lazy_usecase_name", None)
    if lazy_name:
        from takefits.core import usecases

        target = getattr(usecases, str(lazy_name))
        return inspect.signature(target)
    return inspect.signature(handler)


def _should_preserve_last_result(name: str, result: Any, previous_result: Any) -> bool:
    """Keep analysis results available after export actions return output paths."""
    return (
        previous_result is not None
        and name.startswith("export_")
        and isinstance(result, str)
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clone_state_field(value: Any) -> Any:
    """Clone a detached state field, falling back to the original on failure."""
    if value is None:
        return None

    deepcopier = getattr(value, "deepcopy", None)
    if callable(deepcopier):
        try:
            return deepcopier()
        except Exception:
            pass

    copier = getattr(value, "copy", None)
    if callable(copier):
        try:
            return copier()
        except Exception:
            pass

    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _clone_state(state: AppState) -> AppState:
    """Create a lightweight clone of *state*.

    The heavy data array is shared by reference because usecase handlers treat
    it as replace-on-write (for example ``state.data = smoothed_data``). FITS
    metadata objects, however, are cloned so in-place header/WCS edits do not
    mutate the initial seed used by undo/redo replay. Everything else is
    deep-copied so that the clone can be mutated independently.
    """
    # Temporarily detach heavy fields so deepcopy skips them.
    saved_data = state.data
    saved_header = state.header
    saved_wcs = state.wcs
    try:
        state.data = None
        state.header = None
        state.wcs = None
        cloned = copy.deepcopy(state)
    finally:
        # Restore originals on the source state.
        state.data = saved_data
        state.header = saved_header
        state.wcs = saved_wcs
    # Re-attach with shared data but isolated metadata objects.
    cloned.data = saved_data
    cloned.header = _clone_state_field(saved_header)
    cloned.wcs = _clone_state_field(saved_wcs)
    return cloned
