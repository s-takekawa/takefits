from PySide6.QtWidgets import (
    QVBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QMessageBox, QHBoxLayout
)
from datetime import datetime
from takefits.core.io.save_fits import update_datamin_datamax_if_present
from takefits.core.usecases import compute_scaled
from takefits.logic.data_tools import create_preview_snapshot
from takefits.tools.base_panel import (
    BaseToolPanel,
    capture_preferred_cursor_snapshot,
    clear_action_preview_record,
    has_action_record_tag,
    record_action_preview,
    replay_action_history_to_current_cursor,
)
from takefits.core.history_provenance import build_processing_history_lines


class ScalingPanel(BaseToolPanel):
    """
    A panel for applying a simple manual scaling factor (y = ax) to the FITS data.
    """
    
    def __init__(self, fits_viewer, subwindows):
        self.original_data = None
        self.scaling_reference_data = None
        self._has_pending_changes = False
        self._history_category = "scaling"
        self._history_entry = None
        self._action_record_tag = "panel:scaling"
        super().__init__(fits_viewer, subwindows)
        self.setFixedSize(self.sizeHint())
        self.resync_after_workspace_restore()



    def _init_ui_elements(self):
        """Initialize UI widgets."""
        self.scale_input = QLineEdit(self)
        self.scale_input.setFixedWidth(80)
        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")
        self.save_button = QPushButton("Save")
        self.save_button.setEnabled(False)

    def initUI(self):
        """Initialize the User Interface."""
        self._init_ui_elements()
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 16, 8, 8)
        main_layout.setSpacing(6)


        form_layout = QFormLayout()
        form_layout.addRow("Scaling Factor:", self.scale_input)
        main_layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.save_button)
        main_layout.addLayout(button_layout)

        

        self.setLayout(main_layout)
        self.setWindowTitle(f'Scaling: {self.fits_viewer.filename}')
        self.move_to_default_position()

        # Connect Signals
        self.apply_button.clicked.connect(self.apply_manual_scaling)
        self.apply_button.setDefault(True)
        self.scale_input.returnPressed.connect(self.apply_manual_scaling)
        self.reset_button.clicked.connect(self.reset_scaling)
        self.save_button.clicked.connect(self.save_fits)

    def move_to_default_position(self):
        """Move the panel next to the main control panel."""
        if hasattr(self.fits_viewer, 'control_panel'):
            control_panel_geo = self.fits_viewer.control_panel.geometry()
            self.move(control_panel_geo.x() + control_panel_geo.width(), control_panel_geo.y())

    def _ensure_original_data(self):
        """Cache the original data if it hasn't been already."""
        if self.original_data is None:
            self.original_data = create_preview_snapshot(
                self.fits_viewer.data,
                operation_name="Scaling",
            )

    def apply_manual_scaling(self):
        """Apply the y=ax manual scaling."""
        try:
            scale_factor = float(self.scale_input.text())
            self._ensure_original_data()
            
            if self.scaling_reference_data is None:
                # Both caches describe the same immutable pre-scaling cube.
                # Sharing avoids a second full-size copy on the first Apply.
                self.scaling_reference_data = self.original_data

            scaled_data = compute_scaled(self.scaling_reference_data, scale_factor)
            history_msg = f"Manual scaling: multiplied by {scale_factor}"
            
            self._apply_data_and_history(scaled_data, history_msg)
            self._record_preview_action(scale_factor)
            self._has_pending_changes = True
            self.save_button.setEnabled(True)

        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Invalid scaling factor value.')
        except Exception as e:
            QMessageBox.critical(self, 'Scaling Error', f'An error occurred: {e}')
            self.reset_scaling()

    def _apply_data_and_history(self, scaled_data, history_msg):
        """Apply data, update history, and refresh displays."""
        self._update_history(history_msg)
        self.fits_viewer.data = scaled_data
        self.fits_viewer.update_cube()
        for window in self.subwindows:
            if window:
                window.data = scaled_data
                window.update_cube()
        self.update_all_displays()

    def reset_scaling(self):
        """Resets the manual scaling to the original data state."""
        preferred_cursor = capture_preferred_cursor_snapshot(self.fits_viewer)
        removed_preview = self._clear_preview_action()
        restored_from_history = False
        if removed_preview:
            restored_from_history = replay_action_history_to_current_cursor(
                self.fits_viewer,
                preferred_cursor=preferred_cursor,
            )

        if not restored_from_history and self.original_data is not None:
            self.fits_viewer.data = self.original_data
            self.fits_viewer.update_cube()
            for window in self.subwindows:
                if window:
                    window.data = self.original_data
                    window.update_cube()

            self.update_all_displays()

        self.original_data = None
        self.scaling_reference_data = None
        self.scale_input.clear()
        self.save_button.setEnabled(False)
        self._clear_history()
        self._has_pending_changes = False

    def _record_preview_action(self, scale_factor: float) -> None:
        record_action_preview(
            self.fits_viewer,
            "apply_scaling",
            {"scale_factor": float(scale_factor)},
            replace_tag=self._action_record_tag,
        )

    def _clear_preview_action(self) -> bool:
        return bool(
            clear_action_preview_record(
                self.fits_viewer,
                self._action_record_tag,
                action_name="apply_scaling",
            )
        )

    def update_all_displays(self):
        all_windows = [self.fits_viewer] + self.subwindows
        for window in all_windows:
            if not window: continue
            current_channel = 0
            if hasattr(window, 'current_channel_index'):
                try:
                    current_channel = window.current_channel_index()
                except Exception:
                    current_channel = 0
            
            data_slice = None
            if window.data.ndim == 4:
                if window.plane == 'xy': data_slice = window.data[0, current_channel, :, :]
                elif window.plane == 'xz': data_slice = window.data[0, :, current_channel, :]
                elif window.plane == 'zy': data_slice = window.data[0, :, :, current_channel].T
            elif window.data.ndim == 3:
                if window.plane == 'xy': data_slice = window.data[current_channel, :, :]
                elif window.plane == 'xz': data_slice = window.data[:, current_channel, :]
                elif window.plane == 'zy': data_slice = window.data[:, :, current_channel].T
            elif window.data.ndim == 2:
                data_slice = window.data
            if data_slice is not None and hasattr(window, 'im'):
                window.im.set_data(data_slice)
                window.canvas.draw()

    def save_fits(self):
        """Save the scaled data to a new FITS file."""
        if self.original_data is None:
            QMessageBox.information(self, 'No Changes', 'No scaling has been applied to save.')
            return

        data_to_save = self.fits_viewer.data
        new_header = self.fits_viewer.header.copy()

        # Use the bounded central extrema scanner instead of allocating a
        # cube-sized finite mask and a second compacted data copy.
        previous_extrema = {
            key: new_header[key]
            for key in ("DATAMIN", "DATAMAX")
            if key in new_header
        }
        update_datamin_datamax_if_present(
            new_header,
            data_to_save,
            ensure=True,
            drop_if_all_invalid=True,
        )
        if "DATAMIN" not in new_header and "DATAMAX" not in new_header:
            # Historically an all-invalid result left pre-existing extrema
            # untouched. Preserve that edge-case header behaviour.
            for key, value in previous_extrema.items():
                new_header[key] = value

        from takefits.ui.save_fits_dialog import SaveFITS

        if self._history_entry:
            history_lines = list(new_header.get('HISTORY', []))
            filtered_lines = [line for line in history_lines if line not in self._history_entry]
            if 'HISTORY' in new_header:
                new_header.remove('HISTORY', remove_all=True, ignore_missing=True)
            for line in filtered_lines:
                new_header.add_history(line)

        for entry in build_processing_history_lines(self.fits_viewer):
            new_header.add_history(entry)
        save_fits = SaveFITS(data_to_save, new_header, self.fits_viewer.filename)
        save_fits.save(suffix="sc")

    def _update_history(self, operation_details: str):
        """Adds or updates a formatted HISTORY entry."""
        if self._history_entry:
            self._clear_history()

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        general_entry = f"Data scaled using takefits on {timestamp}"
        operation_entry = f"{operation_details}"
        
        self._history_entry = (general_entry, operation_entry)

        seen_headers = set()
        for header in self._get_all_headers():
            if id(header) in seen_headers:
                continue
            header.add_history(general_entry)
            header.add_history(operation_entry)
            seen_headers.add(id(header))

    def _clear_history(self):
        """Removes the specific HISTORY entries added by this panel."""
        if not self._history_entry:
            return
        
        seen_headers = set()
        for header in self._get_all_headers():
            if id(header) in seen_headers:
                continue
            
            if 'HISTORY' not in header:
                continue
            
            current_history = list(header['HISTORY'])
            # Filter out the exact lines added
            new_history = [line for line in current_history if line not in self._history_entry]
            
            del header['HISTORY']
            for line in new_history:
                header.add_history(line)
            
            seen_headers.add(id(header))

        self._history_entry = None

    def _get_all_headers(self):
        """Generator for all relevant headers (main viewer and subwindows)."""
        if self.fits_viewer and self.fits_viewer.header:
            yield self.fits_viewer.header
        for window in self.subwindows:
            if window and window.header:
                yield window.header

    def closeEvent(self, event):
        super().closeEvent(event)

    def has_pending_changes(self) -> bool:
        return self._has_pending_changes

    def pending_close_title(self) -> str:
        return "Close Scaling Panel"

    def pending_close_text(self) -> str:
        return "There are unapplied scaling changes."

    def discard_pending_changes(self) -> None:
        self.reset_scaling()

    def resync_after_workspace_restore(self) -> None:
        pending = bool(has_action_record_tag(self.fits_viewer, self._action_record_tag))
        self._has_pending_changes = pending
        self.save_button.setEnabled(pending)
