#!/usr/bin/env python3

import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import sys
import warnings
from types import SimpleNamespace

if __package__ in {None, ""}:
    # Support direct script execution from a source checkout via a symlinked launcher.
    checkout_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if checkout_root not in sys.path:
        sys.path.insert(0, checkout_root)

from takefits.core.version import APP_DISPLAY_VERSION, APP_NAME, APP_VERSION_TEXT


def _resolve_prog(argv=None, prog: str | None = None) -> str:
    if prog:
        return prog
    if argv is not None:
        return "takefits"
    candidate = os.path.basename(sys.argv[0] or "").strip()
    if not candidate or candidate in {"__main__.py", "main.py"}:
        return "takefits"
    return candidate


def _build_argument_parser(*, prog: str = "takefits") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="GUI-based astronomical FITS viewer and analysis tool.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="FITS file or workspace file to open.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=APP_VERSION_TEXT,
    )
    return parser


def _configure_macos_bundle_name() -> None:
    if platform.system() != "Darwin":
        return
    try:
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        cf.CFBundleGetMainBundle.restype = ctypes.c_void_p
        cf.CFBundleGetInfoDictionary.restype = ctypes.c_void_p
        cf.CFBundleGetInfoDictionary.argtypes = [ctypes.c_void_p]
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFDictionarySetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        bundle = cf.CFBundleGetMainBundle()
        info = cf.CFBundleGetInfoDictionary(bundle)
        key = cf.CFStringCreateWithCString(None, b"CFBundleName", 0x08000100)
        val = cf.CFStringCreateWithCString(None, b"Takefits", 0x08000100)
        cf.CFDictionarySetValue(info, key, val)
        cf.CFRelease(key)
        cf.CFRelease(val)
    except Exception:
        pass


def _load_gui_runtime():
    _configure_macos_bundle_name()

    from PySide6.QtCore import QEvent, QObject, QSettings, QThread, QTimer, Qt
    from PySide6.QtWidgets import QApplication, QFileDialog, QAbstractItemView
    import matplotlib as mpl
    import matplotlib.style as mplstyle
    from astropy.io.fits.verify import VerifyWarning
    from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
    from astropy.wcs import FITSFixedWarning

    mplstyle.use("fast")
    mpl.use("QtAgg")

    warnings.simplefilter("ignore", VerifyWarning)
    warnings.simplefilter("ignore", FITSFixedWarning)
    warnings.simplefilter("ignore", AstropyWarning)
    warnings.simplefilter("ignore", AstropyUserWarning)

    from takefits.app_paths import app_config_path
    from takefits.logic.fits_loader import FITSWorker
    from takefits.tools.color_scale import RegisterColor
    from takefits.ui.main_window import MainWindow

    return SimpleNamespace(
        QAbstractItemView=QAbstractItemView,
        QApplication=QApplication,
        QEvent=QEvent,
        QFileDialog=QFileDialog,
        QObject=QObject,
        QSettings=QSettings,
        QThread=QThread,
        QTimer=QTimer,
        Qt=Qt,
        FITSWorker=FITSWorker,
        MainWindow=MainWindow,
        RegisterColor=RegisterColor,
        app_config_path=app_config_path,
    )


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


def choose_fits_file(runtime):
    """Show a single file dialog with a visible hint and return the chosen file or None."""
    settings_path = runtime.app_config_path("takefits.ini")
    settings = runtime.QSettings(settings_path, runtime.QSettings.Format.IniFormat)
    dialog = runtime.QFileDialog(None, "Takefits - Open FITS File")
    dialog.setNameFilters(["FITS Files (*.fits *.FITS *.fit)", "All Files (*)"])
    dialog.setFileMode(runtime.QFileDialog.FileMode.ExistingFile)
    dialog.setAcceptMode(runtime.QFileDialog.AcceptMode.AcceptOpen)
    dialog.setOption(runtime.QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setWindowModality(runtime.Qt.WindowModality.ApplicationModal)
    dialog.setWindowFlag(runtime.Qt.WindowType.WindowStaysOnTopHint, True)
    last_dir = settings.value("last_fits_dir", "", str)
    if last_dir and os.path.isdir(last_dir):
        dialog.setDirectory(last_dir)
    dialog.setFocusPolicy(runtime.Qt.FocusPolicy.StrongFocus)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

    class EnterToAcceptFilter(runtime.QObject):
        def __init__(self, target_dialog):
            super().__init__(target_dialog)
            self.target_dialog = target_dialog

        def eventFilter(self, obj, event):
            if event.type() == runtime.QEvent.Type.KeyPress and event.key() in (
                runtime.Qt.Key.Key_Return,
                runtime.Qt.Key.Key_Enter,
            ):
                if self.target_dialog.selectedFiles():
                    self.target_dialog.accept()
                    return True
            return super().eventFilter(obj, event)

    enter_filter = EnterToAcceptFilter(dialog)
    dialog.installEventFilter(enter_filter)

    def focus_file_view():
        if dialog.isVisible():
            dialog.activateWindow()
            dialog.setFocus(runtime.Qt.FocusReason.ActiveWindowFocusReason)
            for view in dialog.findChildren(runtime.QAbstractItemView):
                view.setEditTriggers(runtime.QAbstractItemView.EditTrigger.NoEditTriggers)
                view.setSelectionBehavior(runtime.QAbstractItemView.SelectionBehavior.SelectRows)
                if view.isVisible():
                    view.installEventFilter(enter_filter)
                    view.setFocus(runtime.Qt.FocusReason.ActiveWindowFocusReason)
                    break

    runtime.QTimer.singleShot(0, focus_file_view)
    dialog.setLabelText(runtime.QFileDialog.DialogLabel.FileName, "FITS file to view:")
    if dialog.exec() != runtime.QFileDialog.DialogCode.Accepted:
        return None
    selected_files = dialog.selectedFiles()
    filename = selected_files[0] if selected_files else None
    if filename:
        settings.setValue("last_fits_dir", os.path.dirname(filename) or ".")
        settings.setValue("last_fits_path", filename)
        settings.sync()
    return filename


def launch_gui(filename: str | None, workspace_path: str | None) -> int:
    runtime = _load_gui_runtime()

    print(APP_VERSION_TEXT)

    app = runtime.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_DISPLAY_VERSION)
    app.setOrganizationName("takefits")

    runtime.RegisterColor()

    if not filename:
        filename = choose_fits_file(runtime)
        if not filename:
            return 0

    class StartupController(runtime.QObject):
        """Handle worker completion on the GUI thread."""

        def __init__(self, app_obj, thread, launch_filename: str, launch_workspace_path: str | None):
            super().__init__()
            self.app = app_obj
            self.thread = thread
            self.filename = launch_filename
            self.workspace_path = launch_workspace_path
            self.main_win = None

        def on_finished(self, data, header, wcs, spectral_metadata):
            self.main_win = runtime.MainWindow(
                "xy",
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

        def on_error(self, title, details):
            self.thread.quit()
            self.thread.wait()
            message = details if details else title
            if message:
                print(f"[takefits] {message}", file=sys.stderr)
            self.app.exit(1)

    thread = runtime.QThread()
    worker = runtime.FITSWorker(filename)
    controller = StartupController(app, thread, filename, workspace_path)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    worker.finished.connect(controller.on_finished)
    worker.progress.connect(lambda msg: print(msg))
    worker.error.connect(controller.on_error)

    thread.start()
    return app.exec()


def main(argv=None, *, gui_launcher=None, prog: str | None = None) -> int:
    parser = _build_argument_parser(prog=_resolve_prog(argv=argv, prog=prog))
    args = parser.parse_args(argv)

    filename = None
    workspace_path = None
    if args.path:
        if is_workspace_file(args.path):
            try:
                workspace_path, filename = resolve_workspace_source_fits(args.path)
            except Exception as exc:
                print(f"[takefits] Failed to open workspace: {exc}", file=sys.stderr)
                return 1
        else:
            filename = args.path

    if gui_launcher is None:
        gui_launcher = launch_gui
    return int(gui_launcher(filename, workspace_path) or 0)


def main_dev(argv=None, *, gui_launcher=None) -> int:
    return main(argv, gui_launcher=gui_launcher, prog="takefits-dev")


if __name__ == "__main__":
    raise SystemExit(main())
