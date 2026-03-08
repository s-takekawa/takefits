"""Qt worker wrapper for FITS loading.

This module intentionally keeps only Qt signal plumbing. FITS parsing and
normalization live in ``core/io/fits.py`` so GUI and headless paths share the
same implementation.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal as pyqtSignal

from takefits.core.io.fits import FITSLoadError, load_fits


class FITSWorker(QObject):
    """Background worker that loads a FITS file and emits Qt signals."""

    finished = pyqtSignal(np.ndarray, object, object, dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str, str)

    def __init__(self, filename: str):
        super().__init__()
        self.filename = str(filename)

    def run(self) -> None:
        self.progress.emit(f"Loading {self.filename}...")
        try:
            data, header, wcs, spectral_metadata = load_fits(self.filename)
        except FITSLoadError as err:
            path = err.filename if err.filename is not None else self.filename
            if err.kind == "not_found":
                message = f"File not found: {path}"
            elif err.kind == "not_file":
                message = f"Path is not a file: {path}"
            else:
                message = f"Failed to open FITS file: {path}"
            if err.detail:
                message = f"{message} ({err.detail})"
            self.error.emit("", message)
            return
        except Exception as err:  # noqa: BLE001
            self.error.emit("", f"An unexpected error occurred: {err}")
            return

        self.finished.emit(data, header, wcs, spectral_metadata)


__all__ = ["FITSWorker", "FITSLoadError", "load_fits"]
