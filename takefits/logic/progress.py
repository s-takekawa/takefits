"""Qt-free progress reporting and cooperative cancellation primitives.

These helpers let the heavy logic-layer algorithms (clump finding, regrid, ...)
report progress and honour cancellation without importing any GUI toolkit.  A
Qt worker living on a background thread owns a :class:`CancellationToken`, flips
it from the UI thread, and passes a ``progress_callback`` that forwards updates
to Qt signals.  The algorithms only ever see the plain Python objects defined
here, so they stay unit-testable headlessly.
"""
from __future__ import annotations

from typing import Callable, Optional

# A progress callback receives an optional integer percentage (0-100, or None
# for "indeterminate/busy") and an optional human-readable status message.
ProgressCallback = Callable[[Optional[int], Optional[str]], None]


class OperationCancelled(Exception):
    """Raised inside a long-running operation once its token is tripped."""


class CancellationToken:
    """Cooperative cancellation flag shared between a UI thread and a worker.

    Writing and reading a single bool is atomic under CPython's GIL, so no lock
    is needed for the simple single-writer / single-reader pattern used here.
    """

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise OperationCancelled()


class ProgressReporter:
    """Thin wrapper around an optional callback and cancellation token.

    Every :meth:`update` doubles as a cancellation checkpoint: callers can drive
    progress and cooperative cancellation through a single call inside their hot
    loops.  Exceptions raised by the callback are swallowed (except
    :class:`OperationCancelled`) so a flaky UI slot can never crash a worker.
    """

    __slots__ = ("_callback", "_token")

    def __init__(
        self,
        callback: Optional[ProgressCallback] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> None:
        self._callback = callback
        self._token = cancel_token

    @property
    def cancel_token(self) -> Optional[CancellationToken]:
        return self._token

    def check_cancel(self) -> None:
        """Raise :class:`OperationCancelled` if cancellation was requested."""
        if self._token is not None and self._token.cancelled:
            raise OperationCancelled()

    def update(self, value: Optional[int] = None, message: Optional[str] = None) -> None:
        """Report progress and act as a cancellation checkpoint.

        Args:
            value: Percentage 0-100, or ``None`` for an indeterminate/busy state.
            message: Optional status text to surface in the UI.
        """
        self.check_cancel()
        if self._callback is None:
            return
        try:
            self._callback(value, message)
        except OperationCancelled:
            raise
        except Exception:
            # A failing progress sink must never take down the computation.
            pass


def as_reporter(
    progress_callback: Optional[ProgressCallback] = None,
    cancel_token: Optional[CancellationToken] = None,
    reporter: Optional[ProgressReporter] = None,
) -> ProgressReporter:
    """Return a usable reporter from whichever pieces a caller supplied."""
    if reporter is not None:
        return reporter
    return ProgressReporter(progress_callback, cancel_token)
