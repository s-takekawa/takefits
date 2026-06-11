"""Qt worker and non-modal dialog for the PyPI new-version check.

The gating logic lives in ``takefits/logic/update_check.py`` (Qt-free);
this module only fetches off the GUI thread and presents the result.
The app never updates itself -- the dialog shows the pip command and a
release-notes link, mirroring DS9/CARTA-style update notices.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal as pyqtSignal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from takefits.core.version import APP_VERSION
from takefits.logic import update_check

RELEASES_URL = "https://github.com/s-takekawa/takefits/releases"
UPDATE_COMMAND = "pip install -U takefits"


class UpdateCheckWorker(QObject):
    """Fetches the latest PyPI version on a background QThread."""

    # Emits the latest version string, or None when the check failed.
    finished = pyqtSignal(object)

    def __init__(
        self,
        url: str = update_check.PYPI_JSON_URL,
        timeout: float = update_check.DEFAULT_TIMEOUT_SECONDS,
    ):
        super().__init__()
        self._url = url
        self._timeout = timeout

    def run(self) -> None:
        self.finished.emit(
            update_check.fetch_latest_version(self._url, timeout=self._timeout)
        )


class ManualCheckMessageBox(QMessageBox):
    """Manual-check result message with the auto-check opt-in checkbox.

    The "up to date" and "could not check" outcomes must still offer the
    automatic-check toggle: without it, a user who opted out while on the
    latest version would have no way to opt back in (the update dialog --
    the only other place with the checkbox -- never appears for them).
    The choice persists on toggle, matching UpdateAvailableDialog.
    """

    def __init__(self, title: str, text: str, icon, state_path: str, parent=None):
        super().__init__(parent)
        self._state_path = str(state_path)
        self.setWindowTitle(title)
        self.setText(text)
        self.setIcon(icon)

        state = update_check.load_state(self._state_path)
        self.auto_check_checkbox = QCheckBox("Check for updates automatically", self)
        self.auto_check_checkbox.setChecked(update_check.auto_check_enabled(state))
        self.auto_check_checkbox.toggled.connect(self._on_auto_check_toggled)
        self.setCheckBox(self.auto_check_checkbox)

    def _on_auto_check_toggled(self, checked: bool) -> None:
        update_check.update_state(self._state_path, enabled=bool(checked))


class UpdateAvailableDialog(QDialog):
    """Non-modal "update available" notice with skip / opt-out controls."""

    def __init__(self, latest_version: str, state_path: str, parent=None):
        super().__init__(parent)
        self._latest_version = str(latest_version)
        self._state_path = str(state_path)

        self.setWindowTitle("Update Available")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setModal(False)

        layout = QVBoxLayout(self)

        message = QLabel(
            "A new version of Takefits is available.\n"
            f"{APP_VERSION}  →  {self._latest_version}"
        )
        layout.addWidget(message)

        layout.addWidget(QLabel("To update, run:"))
        command_row = QHBoxLayout()
        self.command_edit = QLineEdit(UPDATE_COMMAND)
        self.command_edit.setReadOnly(True)
        command_row.addWidget(self.command_edit)
        self.copy_button = QPushButton("Copy")
        self.copy_button.clicked.connect(self._copy_command)
        command_row.addWidget(self.copy_button)
        layout.addLayout(command_row)

        link = QLabel(f'<a href="{RELEASES_URL}">Release notes</a>')
        link.setOpenExternalLinks(True)
        layout.addWidget(link)

        state = update_check.load_state(self._state_path)
        self.auto_check_checkbox = QCheckBox("Check for updates automatically")
        self.auto_check_checkbox.setChecked(update_check.auto_check_enabled(state))
        self.auto_check_checkbox.toggled.connect(self._on_auto_check_toggled)
        layout.addWidget(self.auto_check_checkbox)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.skip_button = QPushButton("Skip This Version")
        self.skip_button.clicked.connect(self._skip_this_version)
        button_row.addWidget(self.skip_button)
        self.later_button = QPushButton("Remind Me Later")
        self.later_button.setDefault(True)
        self.later_button.clicked.connect(self.close)
        button_row.addWidget(self.later_button)
        layout.addLayout(button_row)

    def _copy_command(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.command_edit.text())

    def _on_auto_check_toggled(self, checked: bool) -> None:
        update_check.update_state(self._state_path, enabled=bool(checked))

    def _skip_this_version(self) -> None:
        update_check.update_state(self._state_path, skip_version=self._latest_version)
        self.close()
