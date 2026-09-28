"""GUI entry helpers for exporting a selected Moment result as a pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from takefits.app_paths import repository_cli_runner
from takefits.core.pipeline import (
    build_moment_export_manifest,
    check_moment_result_support,
)
from takefits.core.pipeline.diagnostics import PipelineDiagnostic
from takefits.ui.moment_render_snapshot import capture_moment_render_snapshot


@dataclass(frozen=True)
class MomentPipelineExportRequest:
    """Values selected in the Moment pipeline export dialog."""

    result_window: Any
    manifest_path: str
    fits_path: Optional[str]
    image_path: Optional[str]


@dataclass(frozen=True)
class MomentPipelineSaveResult:
    """Saved manifest, or diagnostics explaining why none was written."""

    path: Optional[str]
    manifest: Optional[Dict[str, Any]]
    diagnostics: Tuple[PipelineDiagnostic, ...]

    @property
    def ok(self) -> bool:
        return self.path is not None and self.manifest is not None and not any(
            item.severity == "error" for item in self.diagnostics
        )


def _error(code: str, message: str, location: str) -> PipelineDiagnostic:
    return PipelineDiagnostic("error", code, message, location)


def format_pipeline_diagnostics(
    diagnostics: Sequence[PipelineDiagnostic],
) -> str:
    """Return compact, user-facing diagnostic lines."""
    lines = []
    for item in diagnostics:
        location = f" ({item.location})" if item.location else ""
        lines.append(
            f"[{str(item.severity).upper()}] {item.code}{location}: {item.message}"
        )
    return "\n".join(lines)


def pipeline_cli_commands(manifest_path: str) -> tuple[str, str]:
    """Shell-quoted commands for the current source checkout and interpreter."""
    runner = repository_cli_runner()
    manifest = str(Path(manifest_path).expanduser().resolve())
    return tuple(
        shlex.join([sys.executable, str(runner), option, manifest])
        for option in ("--validate", "--actions")
    )


def show_pipeline_saved(owner: QWidget, path: str, message: str) -> None:
    """Show commands without executing the exported pipeline."""
    dialog = QDialog(owner)
    dialog.setWindowTitle("Moment Pipeline Saved")
    dialog.resize(860, 460)
    layout = QVBoxLayout(dialog)
    info = QPlainTextEdit(message, dialog)
    info.setReadOnly(True)
    layout.addWidget(info)
    layout.addWidget(QLabel("Only YAML was saved. Run the pipeline to create FITS / image outputs."))
    layout.addWidget(QLabel("Commands use this Python environment and source checkout."))
    for label, command in zip(("Validate", "Run"), pipeline_cli_commands(path)):
        layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        edit = QLineEdit(command, dialog)
        edit.setReadOnly(True)
        edit.setCursorPosition(0)
        button = QPushButton(f"Copy {label.lower()} command", dialog)
        button.clicked.connect(lambda _checked=False, text=command: QApplication.clipboard().setText(text))
        row.addWidget(edit, 1)
        row.addWidget(button)
        layout.addLayout(row)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def _pipeline_context(owner: Any) -> Tuple[Dict[str, Any], list, int]:
    """Source descriptor, serialized active history and cursor of ``owner``."""
    source = owner._dataset_descriptor()
    records, cursor, _total = owner._session_records_up_to_cursor()
    return source, owner._serialize_action_records(records), cursor


def save_moment_pipeline_manifest(
    owner: Any,
    result_window: Any,
    *,
    manifest_path: str,
    fits_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> MomentPipelineSaveResult:
    """Capture, validate, and write one selected GUI Moment pipeline."""
    manifest_text = str(manifest_path or "").strip()
    fits_text = str(fits_path or "").strip() or None
    image_text = str(image_path or "").strip() or None
    diagnostics: list[PipelineDiagnostic] = []
    if not manifest_text:
        diagnostics.append(
            _error(
                "missing_manifest_path",
                "Choose where to save the pipeline manifest.",
                "manifest_path",
            )
        )
    if fits_text is None and image_text is None:
        diagnostics.append(
            _error(
                "missing_export_target",
                "Choose at least one Moment FITS or rendered-image output.",
                "targets",
            )
        )
    if diagnostics:
        return MomentPipelineSaveResult(None, None, tuple(diagnostics))

    selected_tag = str(
        getattr(result_window, "_workspace_action_tag", "") or ""
    ).strip()
    if not selected_tag:
        diagnostics.append(
            _error(
                "missing_selected_result_tag",
                "The selected window is not linked to a recorded Moment action.",
                "target.result_tag",
            )
        )
        return MomentPipelineSaveResult(None, None, tuple(diagnostics))

    render_snapshot = None
    if image_text is not None:
        capture = capture_moment_render_snapshot(result_window)
        diagnostics.extend(capture.diagnostics)
        if not capture.ok or capture.snapshot is None:
            return MomentPipelineSaveResult(None, None, tuple(diagnostics))
        render_snapshot = capture.snapshot

    try:
        source, history, cursor = _pipeline_context(owner)
        registry = owner.action_session.registry
    except Exception as exc:
        diagnostics.append(
            _error(
                "pipeline_context_unavailable",
                f"The active FITS action history could not be collected: {exc}",
                "history",
            )
        )
        return MomentPipelineSaveResult(None, None, tuple(diagnostics))

    targets: Dict[str, Any] = {}
    if fits_text is not None:
        targets["fits_path"] = fits_text
    if image_text is not None:
        targets["image_path"] = image_text

    converted = build_moment_export_manifest(
        source,
        history,
        targets,
        manifest_path=manifest_text,
        registry=registry,
        selected_action_tag=selected_tag,
        cursor=cursor,
        render_snapshot=render_snapshot,
    )
    diagnostics.extend(converted.diagnostics)
    if not converted.ok or converted.manifest is None:
        return MomentPipelineSaveResult(None, None, tuple(diagnostics))

    output_path = Path(manifest_text).expanduser()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "# takefits pipeline manifest v1 (editable YAML)\n"
            "# Saving this file does not run the pipeline or create its outputs.\n"
            "# When editing Moment conditions, keep compute_moment and export_moment_*\n"
            "# parameters consistent (moment_type, axis, ranges, clip_threshold).\n"
            "# Validation checks structure, not scientific consistency.\n\n"
            + "\n".join(
                yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True)
                for key, value in converted.manifest.items()
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        diagnostics.append(
            _error(
                "manifest_write_failed",
                f"The pipeline manifest could not be written: {exc}",
                "manifest_path",
            )
        )
        return MomentPipelineSaveResult(None, None, tuple(diagnostics))

    return MomentPipelineSaveResult(
        str(output_path), converted.manifest, tuple(diagnostics)
    )


class MomentPipelineExportDialog(QDialog):
    """Select one live Moment result and its future CLI output paths."""

    def __init__(
        self,
        owner: QWidget,
        result_windows: Sequence[Any],
        *,
        source_path: str,
        save_handler: Optional[
            Callable[[MomentPipelineExportRequest], MomentPipelineSaveResult]
        ] = None,
    ) -> None:
        super().__init__(owner)
        self.setWindowTitle("Export Moment Pipeline")
        self.setModal(True)
        self.setMinimumWidth(600)
        # With a handler, Save writes the manifest before closing, so a
        # rejected export keeps the dialog and its edited paths open.
        self._save_handler = save_handler
        self.save_result: Optional[MomentPipelineSaveResult] = None

        source = Path(str(source_path or "")).expanduser()
        base_dir = source.parent if str(source_path or "").strip() else Path.cwd()
        stem = source.stem if source.name else "moment"

        self.source_label = QLineEdit(str(source) if source.name else "<no source>", self)
        self.source_label.setReadOnly(True)
        self.source_label.setToolTip(self.source_label.text())
        self.source_label.setCursorPosition(len(self.source_label.text()))
        self.result_combo = QComboBox(self)
        self.result_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.result_combo.setMinimumContentsLength(20)
        try:
            context_source, history, cursor = _pipeline_context(owner)
            context_error = None
        except Exception as exc:
            context_source, history, cursor = {}, [], None
            context_error = f"The active FITS action history could not be collected: {exc}"
        # Why each unsupported result cannot be exported, by combo index.
        self._blocked_results: Dict[int, str] = {}
        # The owner supplies creation order; show and select the newest first.
        for index, window in enumerate(reversed(result_windows)):
            try:
                title = str(window.windowTitle() or "").strip()
            except Exception:
                title = ""
            plane = str(getattr(window, "plane", "") or "").upper()
            label = title or f"Moment result {index + 1}"
            if plane:
                label = f"{label} [{plane}]"
            tag = getattr(window, "_workspace_action_tag", None)
            record = next((item for item in reversed(history) if tag and
                           (item.get("tag") or item.get("params", {}).get("_window_action_tag")) == tag), None)
            if record:
                params = record.get("params", {})
                details = [str(params.get("moment_type", "Moment")), plane,
                           f"axis={params.get('axis', '?')}"]
                for key in ("pixel_range", "world_range"):
                    if params.get(key) is not None:
                        details.append(f"{key}={params[key]}")
                details.append(f"clip={params.get('clip_threshold', 'none')}")
                label = " | ".join(details) + f" — {label}"
            if tag:
                label += f" (id: {tag})"
            kind = str(record.get("params", {}).get("moment_type", "Moment")) if record else "Moment"
            short_label = f"{index + 1}. {kind} / {plane or '?'}"
            blocker = self._export_blocker(tag, context_source, history, cursor, context_error)
            if blocker:
                self._blocked_results[index] = blocker
                short_label += " (not exportable)"
                label += f"\nNot exportable: {blocker}"
            self.result_combo.addItem(short_label, window)
            self.result_combo.setItemData(index, label, Qt.ItemDataRole.ToolTipRole)
            if blocker:
                # Visible with its reason, but not selectable.
                self.result_combo.model().item(index).setEnabled(False)
        exportable = [
            index
            for index in range(self.result_combo.count())
            if index not in self._blocked_results
        ]
        if exportable:
            self.result_combo.setCurrentIndex(exportable[0])
        self.result_details = QLabel(self)
        self.result_details.setWordWrap(True)
        self.result_details.setTextFormat(Qt.TextFormat.PlainText)
        def update_details(index: int) -> None:
            self.result_details.setText(
                self.result_combo.itemData(index, Qt.ItemDataRole.ToolTipRole) or ""
            )

        self.result_combo.currentIndexChanged.connect(update_details)
        update_details(self.result_combo.currentIndex())
        self.unsupported_note = QLabel(self)
        self.unsupported_note.setWordWrap(True)
        self.unsupported_note.setTextFormat(Qt.TextFormat.PlainText)
        self.unsupported_note.setText(self._blocked_summary())
        self.unsupported_note.setVisible(bool(self._blocked_results))

        self.manifest_edit = QLineEdit(
            str(base_dir / f"{stem}.moment.pipeline.yaml"), self
        )
        self.fits_check = QCheckBox("Moment FITS", self)
        self.fits_check.setChecked(True)
        self.fits_edit = QLineEdit(str(base_dir / f"{stem}.moment.fits"), self)
        self.image_check = QCheckBox("Rendered image", self)
        self.image_check.setChecked(True)
        self.image_edit = QLineEdit(str(base_dir / f"{stem}.moment.png"), self)

        manifest_row, self.manifest_browse = self._path_row(self.manifest_edit)
        fits_row, self.fits_browse = self._path_row(self.fits_edit)
        image_row, self.image_browse = self._path_row(self.image_edit)

        source_group = QGroupBox("1. Source and selected result", self)
        form = QFormLayout(source_group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.addRow(QLabel("Source FITS:"))
        form.addRow(self.source_label)
        form.addRow("Moment result:", self.result_combo)
        form.addRow(self.result_details)
        form.addRow(self.unsupported_note)
        # Align the source field with the compact selector's right edge.
        source_width = (
            form.itemAt(2, QFormLayout.ItemRole.LabelRole).widget().sizeHint().width()
            + form.horizontalSpacing()
            + self.result_combo.sizeHint().width()
        )
        self.source_label.setMaximumWidth(source_width)
        save_group = QGroupBox("2. Save pipeline manifest (YAML)", self)
        save_layout = QVBoxLayout(save_group)
        save_layout.addWidget(manifest_row)
        outputs_group = QGroupBox("3. Outputs created when the pipeline runs", self)
        output_layout = QVBoxLayout(outputs_group)
        output_layout.addWidget(self.fits_check)
        output_layout.addWidget(fits_row)
        output_layout.addWidget(self.image_check)
        output_layout.addWidget(image_row)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.accepted.connect(self._validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save manifest…")
        self.manifest_browse.clicked.connect(self._browse_manifest)
        self.fits_browse.clicked.connect(self._browse_fits)
        self.image_browse.clicked.connect(self._browse_image)
        self.fits_check.toggled.connect(self._sync_output_controls)
        self.image_check.toggled.connect(self._sync_output_controls)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for group_layout in (save_layout, output_layout):
            group_layout.setSpacing(4)
        layout.addWidget(source_group)
        layout.addWidget(save_group)
        layout.addWidget(outputs_group)
        hint = QLabel("Saves settings only — no processing. Supports Integration (Moment 0) / XY results.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.buttons)
        self._sync_output_controls()
        self.result_combo.currentIndexChanged.connect(self._sync_save_enabled)
        self._sync_save_enabled()
        # Fit the contents instead of distributing a fixed 580px height as gaps.
        self.ensurePolished()
        width = max(600, min(owner.width(), 760))
        height = layout.totalHeightForWidth(width)
        self.resize(width, height if height > 0 else self.sizeHint().height())

    @staticmethod
    def _export_blocker(
        tag: Any,
        source: Dict[str, Any],
        history: list,
        cursor: Optional[int],
        context_error: Optional[str],
    ) -> str:
        """Why a result cannot be exported, using the converter's own rules."""
        if context_error:
            return context_error
        if not str(tag or "").strip():
            return "The window is not linked to a recorded Moment action."
        errors = check_moment_result_support(
            source,
            history,
            selected_action_tag=str(tag).strip(),
            cursor=cursor,
        )
        return " ".join(item.message for item in errors)

    def _blocked_summary(self) -> str:
        if not self._blocked_results:
            return ""
        lines = ["Not exportable yet:"]
        for index, reason in sorted(self._blocked_results.items()):
            label = self.result_combo.itemText(index).removesuffix(" (not exportable)")
            lines.append(f"• {label}: {reason}")
        return "\n".join(lines)

    def _sync_save_enabled(self) -> None:
        index = self.result_combo.currentIndex()
        allowed = index >= 0 and index not in self._blocked_results
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(allowed)

    @staticmethod
    def _path_row(edit: QLineEdit) -> tuple[QWidget, QPushButton]:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse…", container)
        layout.addWidget(edit)
        layout.addWidget(browse)
        return container, browse

    def _sync_output_controls(self) -> None:
        fits_enabled = self.fits_check.isChecked()
        image_enabled = self.image_check.isChecked()
        self.fits_edit.setEnabled(fits_enabled)
        self.fits_browse.setEnabled(fits_enabled)
        self.image_edit.setEnabled(image_enabled)
        self.image_browse.setEnabled(image_enabled)

    def _browse_manifest(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Moment Pipeline",
            self.manifest_edit.text(),
            "Pipeline YAML (*.yaml *.yml);;All Files (*)",
        )
        if path:
            if not path.lower().endswith((".yaml", ".yml")):
                path += ".yaml"
            self.manifest_edit.setText(path)

    def _browse_fits(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose Moment FITS Output",
            self.fits_edit.text(),
            "FITS Files (*.fits *.fit);;All Files (*)",
        )
        if path:
            if not path.lower().endswith((".fits", ".fit")):
                path += ".fits"
            self.fits_edit.setText(path)

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose Rendered Image Output",
            self.image_edit.text(),
            "Images (*.png *.pdf);;All Files (*)",
        )
        if path:
            if not Path(path).suffix:
                path += ".png"
            self.image_edit.setText(path)

    def _validate_and_accept(self) -> None:
        if self.result_combo.currentData() is None:
            QMessageBox.warning(self, "Export Moment Pipeline", "No Moment result is available.")
            return
        blocker = self._blocked_results.get(self.result_combo.currentIndex())
        if blocker:
            QMessageBox.warning(self, "Export Moment Pipeline", blocker)
            return
        if not self.manifest_edit.text().strip():
            QMessageBox.warning(self, "Export Moment Pipeline", "Choose a manifest path.")
            return
        if not self.fits_check.isChecked() and not self.image_check.isChecked():
            QMessageBox.warning(self, "Export Moment Pipeline", "Choose at least one output.")
            return
        if self.fits_check.isChecked() and not self.fits_edit.text().strip():
            QMessageBox.warning(self, "Export Moment Pipeline", "Choose a Moment FITS path.")
            return
        if self.image_check.isChecked() and not self.image_edit.text().strip():
            QMessageBox.warning(self, "Export Moment Pipeline", "Choose an image path.")
            return
        if self._save_handler is not None:
            self.save_result = self._save_handler(self.request())
            if not self.save_result.ok:
                QMessageBox.warning(
                    self,
                    "Export Moment Pipeline",
                    format_pipeline_diagnostics(self.save_result.diagnostics)
                    or "The pipeline could not be exported.",
                )
                return
        self.accept()

    def request(self) -> MomentPipelineExportRequest:
        """Return the currently selected values after an accepted dialog."""
        return MomentPipelineExportRequest(
            result_window=self.result_combo.currentData(),
            manifest_path=self.manifest_edit.text().strip(),
            fits_path=(
                self.fits_edit.text().strip() if self.fits_check.isChecked() else None
            ),
            image_path=(
                self.image_edit.text().strip()
                if self.image_check.isChecked()
                else None
            ),
        )


__all__ = [
    "MomentPipelineExportDialog",
    "MomentPipelineExportRequest",
    "MomentPipelineSaveResult",
    "format_pipeline_diagnostics",
    "save_moment_pipeline_manifest",
]
