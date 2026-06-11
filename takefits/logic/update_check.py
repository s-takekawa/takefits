"""Qt-free helpers for the PyPI new-version check.

The GUI shows a non-modal "update available" dialog when a newer stable
release exists on PyPI.  Everything here is plain Python so the gating
logic (version comparison, throttling, skip-version handling) stays unit
testable without a GUI; the Qt worker/dialog live in
``takefits/ui/update_dialog.py``.

Design constraints (see docs/dev/release_prep_v0.3.0.md):

- The automatic check must be completely silent on any failure (offline,
  timeout, unexpected payload) -- it returns ``None`` and the GUI does
  nothing.
- Only stable ``X.Y.Z``-style versions are compared; pre-release/dev
  versions parse to ``None`` and never trigger a notification.
- State (opt-out flag, last-check timestamp, skipped version) lives in a
  small JSON file owned by this feature, so the shared ``config.yaml`` is
  never rewritten behind the user's back.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Optional, Tuple

PYPI_JSON_URL = "https://pypi.org/pypi/takefits/json"
STATE_FILENAME = "update_check.json"
DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_CHECK_INTERVAL_HOURS = 24.0

# Set (e.g. by conftest.py) to disable the automatic startup check.
DISABLE_ENV_VAR = "TAKEFITS_DISABLE_UPDATE_CHECK"


def parse_version(text) -> Optional[Tuple[int, ...]]:
    """Parse a stable dotted version into an int tuple, else ``None``.

    Anything that is not purely ``digits.digits...`` (pre-releases like
    ``0.3.1rc1``, dev/local builds, empty strings) returns ``None`` so the
    caller treats it as "not an upgrade candidate".
    """
    value = str(text or "").strip()
    if not value:
        return None
    parts = value.split(".")
    numbers = []
    for part in parts:
        if not part.isdigit():
            return None
        numbers.append(int(part))
    return tuple(numbers)


def is_newer(candidate, current) -> bool:
    """True when ``candidate`` is a stable version newer than ``current``."""
    cand = parse_version(candidate)
    cur = parse_version(current)
    if cand is None or cur is None:
        return False
    width = max(len(cand), len(cur))
    cand += (0,) * (width - len(cand))
    cur += (0,) * (width - len(cur))
    return cand > cur


def fetch_latest_version(
    url: str = PYPI_JSON_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Return the latest version string from PyPI, or ``None`` on any failure."""
    try:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "takefits-update-check",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        version = str(payload["info"]["version"]).strip()
        return version or None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Persistent state ({"enabled": bool, "last_check_at": iso, "skip_version": str})
def load_state(path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_state(path, state: dict) -> None:
    try:
        directory = os.path.dirname(str(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(dict(state), handle, indent=2, sort_keys=True)
    except Exception:
        pass


def update_state(path, **fields) -> dict:
    """Load-modify-save helper so partial writes never drop other fields."""
    state = load_state(path)
    state.update(fields)
    save_state(path, state)
    return state


def auto_check_enabled(state: dict) -> bool:
    return bool(state.get("enabled", True))


def should_auto_check(
    state: dict,
    now: Optional[datetime] = None,
    interval_hours: float = DEFAULT_CHECK_INTERVAL_HOURS,
) -> bool:
    """True when the silent startup check should run (opt-in + throttled)."""
    if os.environ.get(DISABLE_ENV_VAR):
        return False
    if not auto_check_enabled(state):
        return False
    last = state.get("last_check_at")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(str(last))
    except ValueError:
        return True
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - last_dt).total_seconds() >= interval_hours * 3600.0


def mark_checked(path, now: Optional[datetime] = None) -> dict:
    """Record a check attempt (success or failure) for throttling."""
    now = now or datetime.now(timezone.utc)
    return update_state(path, last_check_at=now.isoformat())


def should_notify(latest, current, state: dict) -> bool:
    """True when the auto check should surface the update dialog."""
    if not latest or not is_newer(latest, current):
        return False
    return str(state.get("skip_version") or "") != str(latest)
