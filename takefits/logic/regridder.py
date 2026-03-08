"""Qt worker that runs the regrid usecase."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal as pyqtSignal

from takefits.core import usecases
from takefits.core.app_state import AppState, create_app_state


class Regridder(QObject):
    """Backend worker that performs FITS regridding operations with Qt signals."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(
        self,
        original_data=None,
        original_wcs=None,
        original_header=None,
        filename=None,
        state: AppState | None = None,
    ):
        super().__init__()
        if state is None:
            self._state = create_app_state(
                data=original_data,
                header=original_header,
                wcs=original_wcs,
                filepath=filename,
                spectral_metadata={},
            )
        else:
            self._state = state

    def perform_regrid(self, params: dict):
        try:
            result = usecases.compute_regrid(
                self._state,
                params,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return None

        self.finished.emit(result.data, result.header)
        return result.data, result.header
