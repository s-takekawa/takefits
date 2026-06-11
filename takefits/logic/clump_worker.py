"""Qt worker that runs clump-finding usecases off the main thread.

Clump finding (Clumpfind / FellWalker / Dendrogram / SCIMES) used to run
synchronously inside the Run button handler, freezing the UI ("Not Responding")
for the entire computation on large cubes.  This worker mirrors the regrid
worker pattern: it lives on a ``QThread`` and reports progress / completion /
errors through Qt signals, while a :class:`CancellationToken` lets the UI thread
request cooperative cancellation.
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QObject, Signal as pyqtSignal

from takefits.core import usecases
from takefits.logic.progress import CancellationToken, OperationCancelled


class ClumpWorker(QObject):
    """Background worker that performs clump finding with Qt signals."""

    # value is 0-100, or -1 for an indeterminate/busy state.
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)  # ClumpResult
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    _ALGORITHMS = {
        "clumpfind": "run_clumpfind",
        "fellwalker": "run_fellwalker",
        "dendrogram": "run_dendrogram",
    }

    def __init__(
        self,
        state,
        algorithm: str,
        params: Dict,
        cancel_token: Optional[CancellationToken] = None,
    ):
        super().__init__()
        self._state = state
        self._algorithm = str(algorithm)
        self._params = dict(params or {})
        self._cancel = cancel_token or CancellationToken()

    @property
    def cancel_token(self) -> CancellationToken:
        return self._cancel

    def cancel(self) -> None:
        self._cancel.cancel()

    def _progress_callback(self, value: Optional[int], message: Optional[str]) -> None:
        if message is not None:
            self.status.emit(str(message))
        self.progress.emit(-1 if value is None else int(value))

    def run(self) -> None:
        """Entry point invoked once the owning QThread starts."""
        usecase_name = self._ALGORITHMS.get(self._algorithm)
        if usecase_name is None:
            self.error.emit(f"Unknown clump algorithm: {self._algorithm}")
            return

        usecase = getattr(usecases, usecase_name)
        try:
            result = usecase(
                self._state,
                progress_callback=self._progress_callback,
                cancel_token=self._cancel,
                **self._params,
            )
        except OperationCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            # A cancel request often manifests as a downstream exception; treat
            # anything raised after cancellation as a cancellation, not an error.
            if self._cancel.cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(str(exc))
            return

        if self._cancel.cancelled:
            self.cancelled.emit()
            return

        self.finished.emit(result)
