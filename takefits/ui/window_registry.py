"""Application-level registry of open MainWindow instances.

Phase 1 of multi-window support. A single process can host several independent
MainWindows (each owning its own data, subwindows, and tool panels). The
registry tracks the live windows so that:

- closing a window quits the app only when it is the *last* one,
- a Window menu can list and switch between open windows,
- new windows can be positioned with a cascade offset.

The registry owns strong references to registered MainWindows until they
unregister during close. Qt top-level widgets can otherwise be garbage
collected immediately after menu-triggered creation if no Python caller keeps
the returned object. Single-window usage is unchanged: one window in, closing it
quits.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal


class WindowRegistry(QObject):
    """Tracks open MainWindow instances for the current process."""

    # Emitted whenever the set of open windows changes (register/unregister).
    windows_changed = Signal()

    _instance: Optional["WindowRegistry"] = None

    @classmethod
    def instance(cls) -> "WindowRegistry":
        if cls._instance is None:
            cls._instance = WindowRegistry()
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        # Insertion-ordered and strong-valued: the registry keeps menu-created
        # top-level MainWindows alive until closeEvent unregisters them.
        self._windows: Dict[int, object] = {}
        self._order: List[int] = []

    # ------------------------------------------------------------------
    def register(self, window) -> None:
        if window is None:
            return
        key = id(window)
        if key in self._windows:
            return
        self._windows[key] = window
        self._order.append(key)
        self.windows_changed.emit()

    def unregister(self, window) -> None:
        if window is None:
            return
        key = id(window)
        existed = key in self._windows
        if existed:
            try:
                del self._windows[key]
            except KeyError:
                pass
        if key in self._order:
            self._order.remove(key)
        if existed:
            self.windows_changed.emit()

    # ------------------------------------------------------------------
    def windows(self) -> List[object]:
        """Live windows in registration order (oldest first)."""
        result = []
        for key in list(self._order):
            window = self._windows.get(key)
            if window is None:
                # Stale key (window collected): prune lazily.
                self._order.remove(key)
                continue
            result.append(window)
        return result

    def count(self) -> int:
        return len(self.windows())

    def number(self, window) -> Optional[int]:
        """1-based position of ``window`` in registration order (oldest = 1).

        Numbers are contiguous: closing a window compacts the rest, so the
        live set is always ``FITS 1..N``. ``windows_changed`` fires on close,
        letting open windows refresh their displayed numbers.
        """
        if window is None:
            return None
        try:
            return self.windows().index(window) + 1
        except ValueError:
            return None

    def is_empty(self) -> bool:
        return self.count() == 0
