from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import os
import yaml
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics, QIcon, QCursor
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from tools.color_scale import ColorMode

class MyNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, plane,ax, color_mode = None, default_image_name="figure", default_image_ext="pdf"):
        super().__init__(canvas, parent)
        self.canvas = canvas
        self.parent = parent
        self.plane = plane
        self.ax = ax
        self.color_mode = color_mode
        self.default_image_name = default_image_name
        self.default_image_ext = default_image_ext
        self._cursor_override_active = False
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

    def pan(self, *args):
        super().pan(*args)
        self._apply_cursor_for_mode()


    def release_zoom(self, event):
        super().release_zoom(event)
        self.get_current_lim(event)
        self._apply_cursor_for_mode()
        self._refresh_overlay_after_nav()

        #self.parent._background = self.canvas.copy_from_bbox(self.parent.overlay_ax.bbox)


    def release_pan(self, event):
        super().release_pan(event)
        self.get_current_lim(event)
        self._apply_cursor_for_mode()
        self._refresh_overlay_after_nav()

    def press_pan(self, event):
        super().press_pan(event)
        self._apply_cursor_for_mode(during_pan=True)
        # optional? no draw on move

    def drag_pan(self, motion_event):
        super().drag_pan(motion_event)
        self._apply_cursor_for_mode(during_pan=True)

    def get_current_lim(self, event):
        if event.inaxes and event.inaxes.get_gid() == "colorbar": return
        if self.ax is None: return
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        if self.color_mode == None:
            try:
                from ui.subwindow import SubWindow
                if isinstance(self.parent, SubWindow):
                    main_window = self.parent.parent
                    main_window.range_panel.update_ranges(self.plane, xlim, ylim)
                    main_window.update_ranges(self.plane, xlim, ylim)
                    if self.plane == 'xy':
                        main_window.set_x_range()
                        main_window.set_y_range()
                    elif self.plane == 'xz':
                        main_window.set_x_range()
                        main_window.set_z_range()
                    elif self.plane == 'zy':
                        main_window.set_y_range()
                        main_window.set_z_range()
                else:
                    self.parent.range_panel.update_ranges(self.plane, xlim, ylim)
                    self.parent.update_ranges(self.plane, xlim, ylim)
                    if self.plane == 'xy':
                        self.parent.set_x_range()
                        self.parent.set_y_range()
                    elif self.plane == 'xz':
                        self.parent.set_x_range()
                        self.parent.set_z_range()
                    elif self.plane == 'zy':
                        self.parent.set_y_range()
                        self.parent.set_z_range()
            except ImportError:
                self.parent.range_panel.update_ranges(self.plane, xlim, ylim)
                self.parent.update_ranges(self.plane, xlim, ylim)
                if self.plane == 'xy':
                    self.parent.set_x_range()
                    self.parent.set_y_range()
                elif self.plane == 'xz':
                    self.parent.set_x_range()
                    self.parent.set_z_range()
                elif self.plane == 'zy':
                    self.parent.set_y_range()
                    self.parent.set_z_range()
        elif self.color_mode == ColorMode.INTEG or self.color_mode == ColorMode.CHANNEL:
            self.parent.update_ranges(self.plane, xlim, ylim)

    def home(self, *args):
        if self.color_mode == None:
            target_viewer = self.parent
            try:
                from ui.subwindow import SubWindow
                if isinstance(self.parent, SubWindow):
                    target_viewer = self.parent.parent
            except ImportError:
                pass

            if hasattr(target_viewer, 'reset_all_ranges'):
                target_viewer.reset_all_ranges()
                if hasattr(target_viewer, 'range_panel'):
                    target_viewer.range_panel.update_ranges('xy', None, None)
                    if target_viewer.data.ndim > 2:
                        target_viewer.range_panel.update_ranges('xz', None, None)
                        target_viewer.range_panel.update_ranges('zy', None, None)
        elif self.color_mode == ColorMode.INTEG or self.color_mode == ColorMode.CHANNEL:
            self.parent.set_full_range()
        self._apply_cursor_for_mode()

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
            f"PDF Files (*.pdf);;EPS Files (*.eps);;SVG Files (*.svg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
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
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
        os.makedirs(config_dir, exist_ok=True)
        self.subplot_params_file = os.path.join(config_dir, 'subplot_params.yaml')
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
        import_button.clicked.connect(self._import_values)
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

    def _import_values(self):
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
        print(f"\033[96mImported parameters from '{filename}' successfully.\033[0m")



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
        overlay_was_enabled = None
        if callable(toggle_overlays):
            overlay_was_enabled = getattr(self.parent, '_overlay_updates_enabled', True)
            toggle_overlays(False)
