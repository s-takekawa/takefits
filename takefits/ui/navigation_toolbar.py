from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import logging
import os
import yaml

from takefits.app_paths import app_config_path
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics, QIcon, QCursor
from PySide6.QtWidgets import QFileDialog, QMessageBox

from takefits.tools.color_scale import ColorMode

logger = logging.getLogger(__name__)

class MyNavigationToolbar(NavigationToolbar2QT):
    _nav_mode_sync_in_progress = False

    def __init__(self, canvas, parent, plane,ax, color_mode = None, default_image_name="figure", default_image_ext="pdf"):
        # NavigationToolbar2QT.__init__ may call set_history_buttons(), so fields
        # referenced there must exist before super().__init__().
        self.canvas = canvas
        self.parent = parent
        self.plane = plane
        self.ax = ax
        self.color_mode = color_mode
        self.default_image_name = default_image_name
        self.default_image_ext = default_image_ext
        self._cursor_override_active = False
        self._external_can_back = False
        self._external_can_forward = False
        super().__init__(canvas, parent)
        loc_label = getattr(self, "locLabel", None)
        if loc_label is not None:
            # Keep multiline coordinate text readable on narrow windows.
            loc_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            loc_label.setTextFormat(Qt.TextFormat.PlainText)
            loc_label.setWordWrap(False)
            two_line_height = (loc_label.fontMetrics().lineSpacing() * 2) + 4
            loc_label.setMinimumHeight(two_line_height)
            loc_label.setMaximumHeight(two_line_height)
            loc_label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
        self._apply_cursor_for_mode()


    def configure_subplots(self, *args, **kwargs):
        """
        If a dialog is already open, just bring it to front and update window size.
        Otherwise, create a new one.
        """
        # 1) Check if a dialog is already open and visible
        if self._subplot_dialog is not None and self._subplot_dialog.isVisible():
            self._subplot_dialog.width_spin.setValue(int(self.parent.window().width()))
            self._subplot_dialog.height_spin.setValue(int(self.parent.window().height()))
            self._subplot_dialog.raise_()
            self._subplot_dialog.activateWindow()
            return self._subplot_dialog
        else:
            # 2) Otherwise, close the old one if it exists (but hidden), then recreate
            if self._subplot_dialog is not None:
                self._subplot_dialog.close()
                self._subplot_dialog = None
    
            self._subplot_dialog = MySubplotToolQt(self.canvas.figure, self.parent)
            self.canvas.mpl_connect("close_event", lambda e: self._subplot_dialog.reject())
    
            # When the parent window is destroyed, also close the dialog
            self.parent.window().destroyed.connect(self._subplot_dialog.close)
    
            # Update spinboxes with current window size
            self._subplot_dialog.update_from_current_subplotpars()
    
            # Show new dialog
            self._subplot_dialog.show()
            return self._subplot_dialog
            

    def zoom(self, *args):
        super().zoom(*args)
        self._apply_cursor_for_mode()
        self._sync_navigation_mode('zoom')

    def set_message(self, s):
        text = str(s or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = []
        for raw in text.split("\n"):
            line = " ".join(str(raw).split()).strip()
            if line:
                normalized.append(line)
        if not normalized:
            super().set_message("")
            return
        if len(normalized) > 2:
            normalized = normalized[:2]
        super().set_message("\n".join(normalized))

    def pan(self, *args):
        super().pan(*args)
        self._apply_cursor_for_mode()
        self._sync_navigation_mode('pan')

    def _iter_linked_toolbars(self):
        target = self._resolve_root_viewer()
        if target is None:
            return []
        viewers = [target] + [v for v in list(getattr(target, 'subwindows', []) or []) if v is not None]
        toolbars = []
        for viewer in viewers:
            toolbar = getattr(viewer, 'toolbar', None)
            if isinstance(toolbar, MyNavigationToolbar) and getattr(toolbar, 'color_mode', None) is None:
                toolbars.append(toolbar)
        return toolbars

    def _sync_navigation_mode(self, source_mode):
        if self.color_mode is not None:
            return
        if MyNavigationToolbar._nav_mode_sync_in_progress:
            return
        desired_mode = 'pan/zoom' if source_mode == 'pan' else 'zoom rect'
        mode_now = str(getattr(self, 'mode', '') or '').lower()
        should_enable = (mode_now == desired_mode)

        linked_toolbars = [tb for tb in self._iter_linked_toolbars() if tb is not self]
        if not linked_toolbars:
            return

        MyNavigationToolbar._nav_mode_sync_in_progress = True
        try:
            for toolbar in linked_toolbars:
                current_mode = str(getattr(toolbar, 'mode', '') or '').lower()
                if source_mode == 'pan':
                    if should_enable and current_mode != 'pan/zoom':
                        toolbar.pan()
                    elif (not should_enable) and current_mode == 'pan/zoom':
                        toolbar.pan()
                elif source_mode == 'zoom':
                    if should_enable and current_mode != 'zoom rect':
                        toolbar.zoom()
                    elif (not should_enable) and current_mode == 'zoom rect':
                        toolbar.zoom()
        finally:
            MyNavigationToolbar._nav_mode_sync_in_progress = False

    def sync_navigation_mode_from_linked(self):
        """Adopt current pan/zoom mode from already-open linked viewers."""
        if self.color_mode is not None:
            return
        if MyNavigationToolbar._nav_mode_sync_in_progress:
            return

        linked_toolbars = [tb for tb in self._iter_linked_toolbars() if tb is not self]
        desired = None
        for toolbar in linked_toolbars:
            mode = str(getattr(toolbar, 'mode', '') or '').lower()
            if mode == 'pan/zoom':
                desired = 'pan'
                break
            if mode == 'zoom rect' and desired is None:
                desired = 'zoom'

        current_mode = str(getattr(self, 'mode', '') or '').lower()
        if desired == 'pan' and current_mode != 'pan/zoom':
            self.pan()
        elif desired == 'zoom' and current_mode != 'zoom rect':
            self.zoom()


    def release_zoom(self, event):
        super().release_zoom(event)
        self.get_current_lim(event)
        self._record_shared_view_history("zoom")
        self._apply_cursor_for_mode()
        self._refresh_overlay_after_nav()
        self._notify_view_history_changed()

        #self.parent._background = self.canvas.copy_from_bbox(self.parent.overlay_ax.bbox)


    def release_pan(self, event):
        super().release_pan(event)
        self.get_current_lim(event)
        self._record_shared_view_history("pan")
        self._apply_cursor_for_mode()
        self._refresh_overlay_after_nav()
        self._notify_view_history_changed()

    def back(self, *args):
        if self.color_mode is None:
            target = self._resolve_root_viewer()
            callback = getattr(target, "view_back", None)
            if callable(callback):
                try:
                    callback()
                finally:
                    self._apply_cursor_for_mode()
                return
        else:
            callback = getattr(self.parent, "view_back", None)
            if callable(callback):
                try:
                    callback()
                finally:
                    self._apply_cursor_for_mode()
                return
        super().back(*args)
        self.get_current_lim(None)
        self._apply_cursor_for_mode()
        self._refresh_overlay_after_nav()
        self._notify_view_history_changed()

    def forward(self, *args):
        if self.color_mode is None:
            target = self._resolve_root_viewer()
            callback = getattr(target, "view_forward", None)
            if callable(callback):
                try:
                    callback()
                finally:
                    self._apply_cursor_for_mode()
                return
        else:
            callback = getattr(self.parent, "view_forward", None)
            if callable(callback):
                try:
                    callback()
                finally:
                    self._apply_cursor_for_mode()
                return
        super().forward(*args)
        self.get_current_lim(None)
        self._apply_cursor_for_mode()
        self._refresh_overlay_after_nav()
        self._notify_view_history_changed()

    def set_history_buttons(self):
        super().set_history_buttons()
        if getattr(self, "color_mode", None) is None:
            target = self._resolve_root_viewer()
            state_getter = getattr(target, "_shared_view_history_state", None)
            if callable(state_getter):
                try:
                    can_back, can_forward = state_getter()
                    self.set_external_history_state(can_back, can_forward)
                except Exception:
                    self._apply_external_history_state()
        else:
            state_getter = getattr(self.parent, "_view_history_state", None)
            if callable(state_getter):
                try:
                    can_back, can_forward = state_getter()
                    self.set_external_history_state(can_back, can_forward)
                except Exception:
                    self._apply_external_history_state()
        self._notify_view_history_changed()

    def press_pan(self, event):
        super().press_pan(event)
        self._apply_cursor_for_mode(during_pan=True)
        # optional? no draw on move

    def drag_pan(self, motion_event):
        super().drag_pan(motion_event)
        self._apply_cursor_for_mode(during_pan=True)

    def get_current_lim(self, event=None):
        if event is not None and event.inaxes and event.inaxes.get_gid() == "colorbar":
            return
        if self.ax is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self._apply_world_ranges_from_limits(xlim, ylim)

    def _resolve_root_viewer(self):
        viewer = getattr(self, 'parent', None)
        if viewer is None:
            return None
        main_viewer = getattr(viewer, 'main_viewer', None)
        if main_viewer is not None:
            return main_viewer
        return viewer

    def _apply_world_ranges_from_limits(self, xlim, ylim):
        if self.color_mode == None:
            target = self._resolve_root_viewer()
            if target is None:
                return
            begin_batch = getattr(target, "_begin_view_history_batch", None)
            end_batch = getattr(target, "_end_view_history_batch", None)
            if callable(begin_batch):
                begin_batch()
            range_panel = getattr(target, 'range_panel', None)
            try:
                if range_panel is not None and hasattr(range_panel, 'update_ranges'):
                    range_panel.update_ranges(self.plane, xlim, ylim)
                if hasattr(target, 'update_ranges'):
                    target.update_ranges(self.plane, xlim, ylim)
                if self.plane == 'xy':
                    if hasattr(target, 'set_x_range'):
                        target.set_x_range()
                    if hasattr(target, 'set_y_range'):
                        target.set_y_range()
                elif self.plane == 'xz':
                    if hasattr(target, 'set_x_range'):
                        target.set_x_range()
                    if hasattr(target, 'set_z_range'):
                        target.set_z_range()
                elif self.plane == 'zy':
                    if hasattr(target, 'set_y_range'):
                        target.set_y_range()
                    if hasattr(target, 'set_z_range'):
                        target.set_z_range()
            finally:
                if callable(end_batch):
                    end_batch()
        elif self.color_mode == ColorMode.INTEG or self.color_mode == ColorMode.CHANNEL:
            self.parent.update_ranges(self.plane, xlim, ylim)

    def _notify_view_history_changed(self):
        if self.color_mode is None:
            target = self._resolve_root_viewer()
        else:
            target = getattr(self, "parent", None)
        if target is None:
            return
        refresh_fn = getattr(target, "_refresh_view_navigation_actions", None)
        if callable(refresh_fn):
            try:
                refresh_fn()
            except Exception:
                pass

    def home(self, *args):
        if self.color_mode == None:
            target_viewer = self._resolve_root_viewer()

            if target_viewer is not None and hasattr(target_viewer, 'reset_all_ranges'):
                target_viewer.reset_all_ranges()
        elif self.color_mode == ColorMode.INTEG or self.color_mode == ColorMode.CHANNEL:
            self.parent.set_full_range()
        self._apply_cursor_for_mode()
        self._notify_view_history_changed()

    def set_external_history_state(self, can_back: bool, can_forward: bool):
        self._external_can_back = bool(can_back)
        self._external_can_forward = bool(can_forward)
        self._apply_external_history_state()

    def _apply_external_history_state(self):
        actions = getattr(self, "_actions", None)
        if not isinstance(actions, dict):
            return
        back_action = actions.get("back")
        forward_action = actions.get("forward")
        if back_action is not None:
            try:
                back_action.setEnabled(bool(self._external_can_back))
            except Exception:
                pass
        if forward_action is not None:
            try:
                forward_action.setEnabled(bool(self._external_can_forward))
            except Exception:
                pass

    def _record_shared_view_history(self, reason: str):
        if self.color_mode is not None:
            return
        target = self._resolve_root_viewer()
        recorder = getattr(target, "_record_shared_view_history", None)
        if callable(recorder):
            try:
                recorder(reason=reason)
            except Exception:
                pass

    def _apply_cursor_for_mode(self, *, during_pan=False):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        if self._cursor_override_active:
            self._clear_cursor_override(app)

        cursor = None
        if self.mode == 'zoom rect':
            cursor = QCursor(Qt.CursorShape.CrossCursor)
        elif self.mode == 'pan/zoom':
            cursor = QCursor(Qt.CursorShape.SizeAllCursor)

        if cursor is not None:
            app.setOverrideCursor(cursor)
            self._cursor_override_active = True

    def _refresh_overlay_after_nav(self):
        parent = getattr(self, 'parent', None)
        plane = getattr(parent, 'plane', None) or getattr(self, 'plane', None)

        refreshed = False
        if parent is not None:
            refresh_fn = getattr(parent, 'redraw_overlay_for_plane', None)
            if callable(refresh_fn) and plane is not None:
                try:
                    refresh_fn(plane)
                    refreshed = True
                except Exception:
                    refreshed = False

        marker_manager = getattr(parent, 'marker_manager', None)
        if marker_manager is None:
            marker_manager = getattr(self, 'marker_manager', None)

        if not refreshed and marker_manager is not None and plane is not None:
            try:
                marker_manager.draw_markers_for_blit(plane)
                marker_manager.redraw_plane(plane)
            except Exception:
                pass

    def _clear_cursor_override(self, app=None):
        if not self._cursor_override_active:
            return
        if app is None:
            app = QtWidgets.QApplication.instance()
        if app is None:
            self._cursor_override_active = False
            return
        try:
            app.restoreOverrideCursor()
        finally:
            self._cursor_override_active = False

    def __del__(self):
        self._clear_cursor_override()

    def onclick(self, event):
        if self.mode == 'zoom rect': pass
        
    def save_figure(self):
        current_dir = os.getcwd()
        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        initial_dir = desktop_dir if os.access(desktop_dir, os.W_OK) else current_dir
        if not self.default_image_name:
            if hasattr(self.parent, "filename") and self.parent.filename:
                self.default_image_name = self.parent.filename
            else:
                self.default_image_name = "figure"
        if self.default_image_name.endswith(".fits"):
            self.default_image_name = self.default_image_name[:-5]

        try:
            self.default_image_ext = f"{self.parent.integ_mode}.{self.default_image_ext}"
        except: pass
        default_filename = f"{self.default_image_name}.{self.default_image_ext}"

        path, _ = QFileDialog.getSaveFileName(
            self.canvas.parent(),
            "Save Figure",
            os.path.join(initial_dir, default_filename),
            "PDF Files (*.pdf);;EPS Files (*.eps);;SVG Files (*.svg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if path:
            visibility = []
            animated = []
            try:
                for patch in self.ax.patches:
                    visibility.append(patch.get_visible())
                    animated.append(getattr(patch, 'get_animated', lambda: False)())
                    patch.set_visible(True)
                    if hasattr(patch, 'set_animated'):
                        patch.set_animated(False)
                self.canvas.figure.savefig(path, transparent=True, dpi=300)
                filename = os.path.basename(path)
                self.show_save_success_message(path, filename)
            finally:
                for patch, vis, anim in zip(self.ax.patches, visibility, animated):
                    patch.set_visible(vis)
                    if hasattr(patch, 'set_animated'):
                        patch.set_animated(anim)
                self._refresh_overlays()
            # Restore a reasonable status message after saving, if available.
            # Use the Axes' format_coord (expects data coords), not the canvas.
            try:
                if self.ax is not None and hasattr(self.ax, "format_coord"):
                    xmid = sum(self.ax.get_xlim()) / 2.0
                    ymid = sum(self.ax.get_ylim()) / 2.0
                    message = self.ax.format_coord(xmid, ymid)
                    if message:
                        self.set_message(message)
            except Exception:
                # Silently ignore any issues restoring the message.
                pass

    def show_save_success_message(self, path, filename):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(f'"{filename}" was saved successfully at:\n{path}')
        #print(f'\n\n"{filename}" was saved successfully at:\n{path}')
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def set_axes(self, new_ax):
        self.ax = new_ax

    def _refresh_overlays(self):
        refresh = getattr(self.parent, 'redraw_main_overlay_and_blit', None)
        if callable(refresh):
            refresh()



class MySubplotToolQt(QtWidgets.QDialog):
    def __init__(self, targetfig, parent):
        super().__init__(parent)
        self.subplot_params_file = app_config_path('subplot_params.yaml')
        self._parent = parent
        self.setWindowIcon(QIcon("matplotlib.png"))
        self.setObjectName("SubplotTool")
        self._spinboxes = {}
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(10, 0, 10, 10)
        self.setLayout(main_layout)
        for group, spinboxes, buttons in [
                ("Borders",
                 ["top", "bottom", "left", "right"],
                 [("Export values", self._export_values)]),
                ("Spacings",
                 ["hspace", "wspace"],
                 [("Tight layout", self._tight_layout),
                  ("Reset", self._reset),
                  ("Close", self.close)])]:
            layout = QtWidgets.QVBoxLayout()
            main_layout.addLayout(layout)
            box = QtWidgets.QGroupBox(group)
            layout.addWidget(box)
            inner = QtWidgets.QFormLayout(box)
            for name in spinboxes:
                self._spinboxes[name] = spinbox = QtWidgets.QDoubleSpinBox()
                spinbox.setRange(0, 1)
                spinbox.setDecimals(3)
                spinbox.setSingleStep(0.005)
                spinbox.setKeyboardTracking(False)
                spinbox.valueChanged.connect(self._on_value_changed)
                inner.addRow(name, spinbox)
            layout.addStretch(1)
            for name, method in buttons:
                button = QtWidgets.QPushButton(name)
                # Don't trigger on <enter>, which is used to input values.
                button.setAutoDefault(False)
                button.clicked.connect(method)
                layout.addWidget(button)
                #if name == "Close":
                #    button.setFocus()
        self._figure = targetfig
        self._defaults = {}
        self._export_values_dialog = None
        
        
        window_container = QtWidgets.QWidget()
        window_container.setObjectName("CustomWindowSizeContainer")
        window_container_layout = QtWidgets.QVBoxLayout(window_container)
        window_container_layout.setContentsMargins(0, 0, 0, 0)
        window_container_layout.setSpacing(5)

        window_group_box = QtWidgets.QGroupBox("Window")
        window_group_box.setFixedHeight(100)
        window_layout = QtWidgets.QGridLayout()
        window_layout.setContentsMargins(10, 0, 10, 0)
        window_layout.setSpacing(5)
        window_group_box.setLayout(window_layout)

        width_label = QtWidgets.QLabel("width")
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(499, 9999)
        self.width_spin.setValue(int(parent.width()))
        self.width_spin.setObjectName("widthSpin")

        height_label = QtWidgets.QLabel("height")
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(115, 9999)
        self.height_spin.setValue(int(parent.height()))
        self.height_spin.setObjectName("heightSpin")

        window_layout.addWidget(height_label, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        window_layout.addWidget(self.height_spin, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        window_layout.addWidget(width_label, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        window_layout.addWidget(self.width_spin, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        def update_window_width():
            new_width = self.width_spin.value()
            parent.resize(new_width, int(parent.window().height()))

        def update_window_height():
            new_height = self.height_spin.value()
            parent.resize(int(parent.window().width()), new_height)

        self.width_spin.valueChanged.connect(update_window_width)
        self.height_spin.valueChanged.connect(update_window_height)

        window_container_layout.addWidget(window_group_box, alignment=Qt.AlignmentFlag.AlignBottom)

        import_button = QtWidgets.QPushButton("Import values")
        import_button.setAutoDefault(False)
        import_button.clicked.connect(lambda _=False: self._import_values(show_message=True))
        window_container_layout.addWidget(import_button, alignment=Qt.AlignmentFlag.AlignBottom)


        dlg_layout = self.layout()
        existing_container = self.findChild(QtWidgets.QWidget, "CustomWindowSizeContainer")
        if existing_container is not None:
            dlg_layout.removeWidget(existing_container)
            existing_container.deleteLater()

        dlg_layout.insertWidget(0, window_container)

        tmp_dict = {'height': self.height_spin, 'width': self.width_spin}
        self._spinboxes = {**tmp_dict, **self._spinboxes}
        self.update_from_current_subplotpars()
        
    def update_from_current_subplotpars(self):
        spinboxes = self._spinboxes
        allowed_keys = {"left", "right", "top", "bottom", "wspace", "hspace"}
    
        self._defaults = {} 
    
        for name, spinbox in spinboxes.items():
            if name in allowed_keys:
                self._defaults[spinbox] = getattr(self._figure.subplotpars, name)
            elif name == "width":
                current_width = self._parent.window().width()
                self._defaults[spinbox] = current_width
                spinbox.blockSignals(True)
                spinbox.setValue(current_width)
                spinbox.blockSignals(False)
            elif name == "height":
                current_height = self._parent.window().height()
                self._defaults[spinbox] = current_height
                spinbox.blockSignals(True)
                spinbox.setValue(current_height)
                spinbox.blockSignals(False)

        self._reset()

    def _export_values(self):
        # Explicitly round to 3 decimals (which is also the spinbox precision)
        # to avoid numbers of the form 0.100...001.
        self._export_values_dialog = QtWidgets.QDialog()
        layout = QtWidgets.QVBoxLayout()
        self._export_values_dialog.setLayout(layout)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        layout.addWidget(text)
        text.setPlainText(
            ",\n".join(
                f"{attr}={int(spinbox.value())}" if attr in {"height", "width"}
                else f"{attr}={spinbox.value():.3}"
                for attr, spinbox in self._spinboxes.items()
            )
        )
        # Adjust the height of the text widget to fit the whole text, plus
        # some padding.
        size = text.maximumSize()
        size.setHeight(
            QFontMetrics(text.document().defaultFont())
            .size(0, text.toPlainText()).height() + 20)
        text.setMaximumSize(size)
        self._export_values_dialog.show()
        
        param_dict = {}
        for attr, spinbox in self._spinboxes.items():
            if attr in {"height", "width"}:
                # Save as integer
                param_dict[attr] = int(spinbox.value())
            else:
                # Save as float with 3 decimal places
                param_dict[attr] = float(f"{spinbox.value():.3f}")
        
        # Fixed filename
        filename = self.subplot_params_file
        
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(param_dict, f, default_flow_style=False)
        
        print(f"\033[96mExported parameters to '{filename}'\033[0m")

    def _import_values(self, show_message: bool = True):
        """
        Import parameters from the same fixed YAML file ('myparams.yaml')
        and update each spinbox accordingly.
        """
        filename = self.subplot_params_file
        
        # Attempt to load the file
        try:
            with open(filename, "r", encoding="utf-8") as f:
                param_dict = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"\033[93m\033[1mFile '{filename}' not found.\033[0m")
            return
        except Exception as e:
            print(f"Failed to load '{filename}': {e}")
            return
        
        # Update spinboxes with loaded values
        for attr, value in param_dict.items():
            if attr in self._spinboxes:
                # Update the spinbox itself
                self._spinboxes[attr].blockSignals(True)
                self._spinboxes[attr].setValue(value)
                self._spinboxes[attr].blockSignals(False)
    
                # If it's width or height, also resize the parent window
                if attr == "width":
                    # if you have a separate QSpinBox for width:
                    self.width_spin.setValue(int(value))
                    #self.width_spin_dummy.setValue(float(value))
                    # Then resize parent
                    self._parent.resize(int(value), self._parent.height())
    
                elif attr == "height":
                    self.height_spin.setValue(int(value))
                    #self.height_spin_dummy.setValue(float(value))
                    self._parent.resize(self._parent.width(), int(value))
        
        # Optionally trigger any re-draw or logic after updating
        self._on_value_changed()
        if show_message:
            print(f"\033[96mImported parameters from '{filename}' successfully.\033[0m")
        else:
            logger.debug("Imported parameters from '%s' successfully.", filename)



    def _on_value_changed(self):
        spinboxes = self._spinboxes
        # Set all mins and maxes, so that this can also be used in _reset().
        for lower, higher in [("bottom", "top"), ("left", "right")]:
            spinboxes[higher].setMinimum(spinboxes[lower].value() + .001)
            spinboxes[lower].setMaximum(spinboxes[higher].value() - .001)
        
        allowed_keys = {"left", "right", "top", "bottom", "wspace", "hspace"}
        adjust_params = {attr: spinbox.value() for attr, spinbox in spinboxes.items() if attr in allowed_keys}
        self._figure.subplots_adjust(**adjust_params)
        #self._figure.subplots_adjust(
        #    **{attr: spinbox.value() for attr, spinbox in spinboxes.items()})
        self._figure.canvas.draw_idle()

    def _tight_layout(self):
        spinboxes = self._spinboxes
        self._figure.tight_layout()
        allowed_keys = {"left", "right", "top", "bottom", "wspace", "hspace"}
        for attr, spinbox in spinboxes.items():
            if attr in allowed_keys:
                spinbox.blockSignals(True)
                spinbox.setValue(vars(self._figure.subplotpars)[attr])
                spinbox.blockSignals(False)
        self._figure.canvas.draw_idle()

    def _reset(self):
        for spinbox, value in self._defaults.items():
            #if (spinbox is self._spinboxes.get('width')
            #        or spinbox is self._spinboxes.get('height')):
            #    spinbox.setRange(100, 3000)
            #else:
            #    spinbox.setRange(0, 1)
    
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
    
        # After setting spinboxes for width/height, resize the parent window too:
        w_val = self._spinboxes['width'].value()
        h_val = self._spinboxes['height'].value()
        self.width_spin.setValue(int(w_val))
        #self.width_spin_dummy.setValue(float(w_val))
        self.height_spin.setValue(int(h_val))
        #self.height_spin_dummy.setValue(float(h_val))
        self._parent.window().resize(int(w_val), int(h_val))
    
        self._on_value_changed()
        toggle_overlays = getattr(self.parent, 'set_overlay_updates_enabled', None)
        if callable(toggle_overlays):
            toggle_overlays(False)
