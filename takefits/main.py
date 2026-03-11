#!/usr/bin/env python3

import ctypes
import ctypes.util
import json
import os
import platform
import sys

if platform.system() == "Darwin":
    try:
        _cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        _cf.CFBundleGetMainBundle.restype = ctypes.c_void_p
        _cf.CFBundleGetInfoDictionary.restype = ctypes.c_void_p
        _cf.CFBundleGetInfoDictionary.argtypes = [ctypes.c_void_p]
        _cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        _cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        _cf.CFDictionarySetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        _cf.CFRelease.argtypes = [ctypes.c_void_p]
        _bundle = _cf.CFBundleGetMainBundle()
        _info = _cf.CFBundleGetInfoDictionary(_bundle)
        _key = _cf.CFStringCreateWithCString(None, b"CFBundleName", 0x08000100)
        _val = _cf.CFStringCreateWithCString(None, b"Takefits", 0x08000100)
        _cf.CFDictionarySetValue(_info, _key, _val)
        _cf.CFRelease(_key)
        _cf.CFRelease(_val)
    except Exception:
        pass
from PySide6.QtCore import QEvent, QObject, QSettings, QThread, QTimer, Qt, Slot
from PySide6.QtWidgets import QApplication, QFileDialog, QAbstractItemView
import matplotlib as mpl
import matplotlib.style as mplstyle
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
from astropy.wcs import FITSFixedWarning

mplstyle.use('fast')
mpl.use('QtAgg')

from takefits.ui.main_window import MainWindow
from takefits.tools.color_scale import RegisterColor
from takefits.core.version import APP_NAME, APP_VERSION
from takefits.logic.fits_loader import FITSWorker  # Import the refactored worker
from takefits.app_paths import app_config_path


def is_workspace_file(path: str) -> bool:
    lower = str(path or "").lower()
    return lower.endswith(".workspace.json") or lower.endswith(".json")


def resolve_workspace_source_fits(workspace_path: str) -> tuple[str, str]:
    absolute_workspace = os.path.abspath(str(workspace_path))
    with open(absolute_workspace, "r", encoding="utf-8") as handle:
        payload = json.loads(handle.read())
    if not isinstance(payload, dict) or not isinstance(payload.get("history"), list):
        raise ValueError("Invalid workspace format.")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("Workspace is missing source metadata.")
    source_fits = str(source.get("filepath") or "").strip()
    if not source_fits:
        raise ValueError("Workspace does not include source FITS filepath.")
    if not os.path.isabs(source_fits):
        source_fits = os.path.join(os.path.dirname(absolute_workspace), source_fits)
    return absolute_workspace, os.path.abspath(source_fits)


def choose_fits_file():
    """Show a single file dialog with a visible hint and return the chosen file or None."""
    settings_path = app_config_path("takefits.ini")
    settings = QSettings(settings_path, QSettings.Format.IniFormat)
    dialog = QFileDialog(None, "Takefits - Open FITS File")
    dialog.setNameFilters(["FITS Files (*.fits *.FITS *.fit)", "All Files (*)"])
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    last_dir = settings.value("last_fits_dir", "", str)
    if last_dir and os.path.isdir(last_dir):
        dialog.setDirectory(last_dir)
    dialog.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    # Ensure Enter/Return activates Open instead of inline rename
    class EnterToAcceptFilter(QObject):
        def __init__(self, target_dialog):
            super().__init__(target_dialog)
            self.target_dialog = target_dialog

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self.target_dialog.selectedFiles():
                    self.target_dialog.accept()
                    return True
            return QObject.eventFilter(self, obj, event)

    enter_filter = EnterToAcceptFilter(dialog)
    dialog.installEventFilter(enter_filter)
    # Nudge focus after show so keyboard navigation works immediately
    def focus_file_view():
        if dialog.isVisible():
            dialog.activateWindow()
            dialog.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            for view in dialog.findChildren(QAbstractItemView):
                view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
                view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
                if view.isVisible():
                    view.installEventFilter(enter_filter)
                    view.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                    break
    QTimer.singleShot(0, focus_file_view)
    dialog.setLabelText(QFileDialog.DialogLabel.FileName, "FITS file to view:")
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return None
    selected_files = dialog.selectedFiles()
    filename = selected_files[0] if selected_files else None
    if filename:
        settings.setValue("last_fits_dir", os.path.dirname(filename) or ".")
        settings.setValue("last_fits_path", filename)
        settings.sync()
    return filename


class StartupController(QObject):
    """Handle worker completion on the GUI thread."""

    def __init__(self, app, thread, filename: str, workspace_path: str | None):
        super().__init__()
        self.app = app
        self.thread = thread
        self.filename = filename
        self.workspace_path = workspace_path
        self.main_win = None

    @Slot(object, object, object, dict)
    def on_finished(self, data, header, wcs, spectral_metadata):
        self.main_win = MainWindow(
            'xy',
            f"MainWindow: {self.filename}",
            data,
            header,
            wcs,
            self.filename,
            spectral_metadata,
        )
        print("*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*")
        self.main_win.show()
        if self.workspace_path:
            set_workspace_path = getattr(self.main_win, "set_workspace_save_path", None)
            if callable(set_workspace_path):
                set_workspace_path(self.workspace_path)
            restored = self.main_win.load_workspace_from_path(
                self.workspace_path,
                confirm_replace=False,
                show_result_dialog=False,
            )
            if not restored:
                print(f"[takefits] Failed to restore workspace: {self.workspace_path}", file=sys.stderr)
        self.thread.quit()
        self.thread.wait()

    @Slot(str, str)
    def on_error(self, title, details):
        self.thread.quit()
        self.thread.wait()
        message = details if details else title
        if message:
            print(f"[takefits] {message}", file=sys.stderr)
        self.app.exit(1)

def main():
    # Specify warnings to ignore
    warnings.simplefilter('ignore', VerifyWarning)
    warnings.simplefilter('ignore', FITSFixedWarning)
    warnings.simplefilter('ignore', AstropyWarning)
    warnings.simplefilter('ignore', AstropyUserWarning)
    
    print(f"{APP_NAME} version {APP_VERSION}")

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("takefits")
    
    RegisterColor()
    
    # Determine launch target (FITS file or workspace file)
    workspace_path = None
    if len(sys.argv) >= 2:
        launch_arg = sys.argv[1]
        if is_workspace_file(launch_arg):
            try:
                workspace_path, filename = resolve_workspace_source_fits(launch_arg)
            except Exception as exc:
                print(f"[takefits] Failed to open workspace: {exc}", file=sys.stderr)
                sys.exit(1)
        else:
            filename = launch_arg
    else:
        filename = choose_fits_file()
        if not filename:
            sys.exit(0)
    # Create a QThread and start the FITSWorker
    thread = QThread()
    worker = FITSWorker(filename)
    controller = StartupController(app, thread, filename, workspace_path)
    worker.moveToThread(thread)
    
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    worker.finished.connect(controller.on_finished)
    worker.progress.connect(lambda msg: print(msg))
    worker.error.connect(controller.on_error)
    
    thread.start()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
