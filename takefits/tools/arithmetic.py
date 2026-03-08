from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QMessageBox, QGroupBox, QListWidget
)
from PySide6.QtCore import Qt, Signal as pyqtSignal
from PySide6.QtGui import QFont
import numpy as np
import ast
from datetime import datetime
from takefits.ui.save_fits_dialog import SaveFITS
from takefits.core.history_provenance import build_processing_history_lines


from takefits.core.usecases import compute_arithmetic
from takefits.tools.base_panel import (
    capture_preferred_cursor_snapshot,
    clear_action_preview_record,
    confirm_pending_close,
    has_action_record_tag,
    record_action_preview,
    replay_action_history_to_current_cursor,
)

class UnitPreservationChecker(ast.NodeVisitor):
    """
    Checks if an arithmetic expression preserves the original unit (BUNIT).
    Returns True if the operation is unit-preserving (e.g., A + constant, A * constant),
    False if the unit changes (e.g., A * B, np.log10(A)).
    """
    # Functions that preserve units
    PRESERVING_FUNCTIONS = {
        'abs', 'absolute', 'fabs', 
        'max', 'min', 'amax', 'amin', 'nanmax', 'nanmin',
        'fmax', 'fmin', 'maximum', 'minimum',
        'mean', 'nanmean', 'median', 'nanmedian',
        'std', 'nanstd', 'ptp', 
        'sum', 'nansum', # Sum preserves unit string (e.g. K + K = K)
        'copysign', 'trunc', 'floor', 'ceil', 'rint'
    }

    def __init__(self, variable_names):
        self.variable_names = set(variable_names)
        self.preserves_unit = True
        self.uses_multiple_variables = False
        
    def visit_BinOp(self, node):
        # Check if both operands are variables (e.g., A * B)
        left_is_var = self._is_variable(node.left)
        right_is_var = self._is_variable(node.right)
        
        if left_is_var and right_is_var:
            # Two variables: unit changes (e.g., A * B, A / B)
            # Exception: Addition/Subtraction of same unit variables *should* preserve
            # But we don't know if units are same. Assuming they are for same-type operations?
            # Current logic conservative: A+B -> False.
            # Let's keep it conservative for multi-variable ops unless explicitly handled.
            self.preserves_unit = False
            self.uses_multiple_variables = True
        elif isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod)):
            # Multiplication/Division/Power with variable always changes unit 
            # (unless multiplying by unitless constant, but strictly unit string might imply scale?)
            # Usually: K * 2 -> K. K * K -> K^2.
            # Here scalar * var -> preserves.
            if not (self._is_constant(node.left) or self._is_constant(node.right)):
                # If neither is constant (i.e. both vars), already caught above.
                # If one is constant:
                # A * 2 -> Preserves.
                pass
        elif isinstance(node.op, (ast.Add, ast.Sub)):
            # Addition or subtraction: preserve unit if one operand is constant
            if not (self._is_constant(node.left) or self._is_constant(node.right)):
                # A + B caught above.
                pass
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Function calls
        func_name = None
        if isinstance(node.func, ast.Attribute):
            # np.max, np.abs
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            # abs, max
            func_name = node.func.id
            
        if func_name not in self.PRESERVING_FUNCTIONS:
             self.preserves_unit = False
             
        self.generic_visit(node)
    
    def visit_UnaryOp(self, node):
        # Unary operations like -A, +A preserve units
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            # +A, -A preserve unit
            pass
        else:
            # ~A (invert) changes unit or applies to mask
            self.preserves_unit = False
        self.generic_visit(node)
    
    def _is_variable(self, node):
        """Check if node is a variable (A, B, etc.)"""
        # Also handles Attributes if they are variables? No, variables are Names in this context.
        if isinstance(node, ast.Name):
            return node.id in self.variable_names
        return False
    
    def _is_constant(self, node):
        """Check if node is a constant (number)"""
        return isinstance(node, (ast.Constant, ast.Num)) or \
               (isinstance(node, ast.UnaryOp) and 
                isinstance(node.operand, (ast.Constant, ast.Num)))

class CubeArithmeticPanel(QWidget):
    """
    A tool for performing mathematical operations on FITS data cubes.
    Supports arithmetic, numpy functions, and masking.
    """
    
    destroyed = pyqtSignal()
    
    def __init__(self, fits_viewer, subwindows, parent=None):
        super().__init__(parent)
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows
        self._action_record_tag = "panel:arithmetic"
        
        self.variables = {} 
        self.original_data_map = {} # Store original data for all windows
        self.original_bunit_map = {} # Store original BUNIT state for all windows
        self._has_restored_preview = False
        
        self.initUI()
        self.refresh_sources()
        self.resync_after_workspace_restore()

    def initUI(self):
        self.setWindowTitle("Arithmetic")
        self.resize(360, 220)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5) # Tight margins
        main_layout.setSpacing(5)

        # 1. Variables List
        source_group = QGroupBox("Variables")
        source_layout = QVBoxLayout()
        source_layout.setContentsMargins(2, 5, 2, 2)
        source_layout.setSpacing(2)
        
        self.source_list = QListWidget()
        self.source_list.setFixedHeight(38) # Approx 2 lines, tighter
        self.source_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.source_list.setStyleSheet("font-size: 11px;")
        self.source_list.itemClicked.connect(self.handle_list_click)
        
        # Refresh Button (Right aligned, standard style)
        refresh_layout = QHBoxLayout()
        refresh_layout.addStretch()
        
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedSize(60, 24)
        refresh_btn.setStyleSheet("font-size: 10px; padding: 2px;")
        refresh_btn.clicked.connect(self.refresh_sources)
        
        refresh_layout.addWidget(refresh_btn)
        
        source_layout.addWidget(self.source_list)
        source_layout.addLayout(refresh_layout)
        source_group.setLayout(source_layout)
        main_layout.addWidget(source_group)

        # 2. Expression Input
        expr_group = QGroupBox("Formula (numpy syntax)")
        expr_layout = QVBoxLayout()
        expr_layout.setContentsMargins(2, 5, 2, 2)
        
        self.expr_input = QLineEdit()
        self.expr_input.setPlaceholderText("e.g. np.log10(A), np.where(B > 1, A/B, np.nan), etc.")
        self.expr_input.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.expr_input.setFont(QFont("Menlo", 12))
        self.expr_input.returnPressed.connect(self.compute_and_execute)
        
        expr_layout.addWidget(self.expr_input)
        expr_group.setLayout(expr_layout)
        main_layout.addWidget(expr_group)

        # 3. Actions (No grouping)
        
        # Preview and Reset Buttons
        preview_reset_layout = QHBoxLayout()
        preview_reset_layout.setSpacing(5)
        
        self.preview_btn = QPushButton("Preview Result")
        self.preview_btn.clicked.connect(self.apply_preview)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_preview)
        self.reset_btn.setFixedWidth(60)
        
        preview_reset_layout.addWidget(self.preview_btn)
        preview_reset_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(preview_reset_layout)
        
        self.save_btn = QPushButton("Save as FITS")
        self.save_btn.clicked.connect(self.save_result)
        
        main_layout.addWidget(self.save_btn)
        main_layout.addStretch()

    def show_message(self, message):
         QMessageBox.information(self, "Info", message)
         
    def _get_shape_str(self, data):
        """Returns string representation of shape (X, Y, Z)."""
        if data is None: return ""
        # numpy shape is (Z, Y, X) or (T, Z, Y, X)
        # We want to display (X, Y, Z)
        shape = data.shape
        if len(shape) >= 3:
            # Reverse the last 3 dimensions to get (X, Y, Z)
            dims = shape[-3:][::-1]
            return f"({', '.join(map(str, dims))})"
        else:
            # Fallback for 2D or 1D
            return str(shape[::-1])

    def refresh_sources(self):
        self.variables = {}
        self.source_list.clear()
        
        # A = Current Viewer Data
        if hasattr(self.fits_viewer, 'data'):
            data = self.fits_viewer.data
            shape_str = self._get_shape_str(data)
            self.variables["A"] = {
                "data": data, 
                "header": self.fits_viewer.header,
                "name": self.fits_viewer.filename
            }
            self.source_list.addItem(f"A: {self.fits_viewer.filename} {shape_str}")
            
        # B = Optional Second File
        if "B" not in self.variables:
             self.source_list.addItem("B: [Click to Load]")

    def handle_list_click(self, item):
        if item.text().startswith("B:"):
             self.load_file_b()

    def load_file_b(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Load FITS for Variable B", "", "FITS Files (*.fits)", options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            from astropy.io import fits
            import os
            try:
                with fits.open(path) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                
                self.variables["B"] = {"data": data, "header": header, "name": os.path.basename(path)}
                self.variables["B"]["path"] = path
                # Update list item
                self.source_list.clear()
                
                # Re-add A
                a_info = self.variables.get('A', {})
                a_name = a_info.get('name', 'N/A')
                a_shape = self._get_shape_str(a_info.get('data'))
                self.source_list.addItem(f"A: {a_name} {a_shape}")
                
                 # Add B
                b_name = self.variables['B']['name']
                b_shape = self._get_shape_str(data)
                self.source_list.addItem(f"B: {b_name} {b_shape}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def compute_and_execute(self):
        # Auto-activate preview on Enter
        self.apply_preview()

    def run_computation(self):
        expr_str = self.expr_input.text().strip()
        if not expr_str:
            return None, None

        # Prepare arguments for usecase
        data_a = self.variables.get("A", {}).get("data")
        header_a = self.variables.get("A", {}).get("header")
        
        if data_a is None:
            QMessageBox.warning(self, "Data Error", "Primary data (A) is not available.")
            return None, None
            
        data_b = self.variables.get("B", {}).get("data")

        try:
            # Delegate computation to core usecase
            result = compute_arithmetic(
                data_a=data_a,
                operation="expression", 
                data_b=data_b,
                expression=expr_str
            )
            return result, header_a
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Computation failed:\n{e}")
            return None, None

    def apply_preview(self):
        """Calculates and applies the result to headers/display."""
        data, header = self.run_computation()
        if data is not None:
            # Cache original data if not already cached
            if not self.original_data_map:
                self.original_data_map[self.fits_viewer] = self.fits_viewer.data
                for sub in self.subwindows:
                    if sub:
                        self.original_data_map[sub] = sub.data
            if not self.original_bunit_map:
                self._capture_original_bunit_map()

            # Update all windows
            viewer_data = self.original_data_map.get(self.fits_viewer)
            if viewer_data is not None and data.ndim != viewer_data.ndim:
                    QMessageBox.warning(self, "Warning", "Result dimension differs. Viewer update might fail.")

            self._apply_data_to_all(data)
            preserves_unit = self._expression_preserves_unit(self.expr_input.text().strip())
            self._apply_preview_bunit_state(preserves_unit)
            self._record_preview_action()
    
    def reset_preview(self):
        """Restores the original data from cache."""
        preferred_cursor = capture_preferred_cursor_snapshot(self.fits_viewer)
        removed_preview = self._clear_preview_action()
        restored_from_history = False
        if removed_preview:
            restored_from_history = replay_action_history_to_current_cursor(
                self.fits_viewer,
                preferred_cursor=preferred_cursor,
            )

        if not restored_from_history and self.original_data_map:
            # Restore each window
            for window, orig_data in self.original_data_map.items():
                if window: # Check if still alive
                        self._update_window_data(window, orig_data)
            
            self._sync_app_state_data()
            self.update_all_displays()
        self._restore_original_bunit_state()
        self._sync_bunit_to_viewers()
        self._sync_app_state_data()
        self._refresh_live_intensity_labels()
        self.original_data_map.clear()
        self.original_bunit_map.clear()
        self._has_restored_preview = False

    def _record_preview_action(self):
        expr_str = self.expr_input.text().strip()
        if not expr_str:
            return
        payload = {
            "operation": "expression",
            "expression": expr_str,
        }
        data_b_path = self.variables.get("B", {}).get("path")
        if data_b_path:
            payload["data_b_path"] = data_b_path
        record_action_preview(
            self.fits_viewer,
            "compute_arithmetic",
            payload,
            replace_tag=self._action_record_tag,
        )

    def _clear_preview_action(self) -> bool:
        return bool(
            clear_action_preview_record(
                self.fits_viewer,
                self._action_record_tag,
                action_name="compute_arithmetic",
            )
        )

    def _has_pending_preview_in_history(self) -> bool:
        return bool(has_action_record_tag(self.fits_viewer, self._action_record_tag))

    def resync_after_workspace_restore(self) -> None:
        self._has_restored_preview = self._has_pending_preview_in_history()

    def _apply_data_to_all(self, new_data):
        """Apply new data to all windows and refresh."""
        all_windows = [self.fits_viewer] + [s for s in self.subwindows if s]
        for window in all_windows:
            self._update_window_data(window, new_data)
        
        self._sync_app_state_data()
        self.update_all_displays()

    def _update_window_data(self, window, data):
        """Helper to update window.data and window.cube safely."""
        window.data = data
        if hasattr(window, 'update_cube'):
            window.update_cube()
            return
        if data.ndim > 2:
            if data.ndim == 3:
                window.cube = window.data
            elif data.ndim == 4:
                window.cube = window.data[0]

    def _sync_app_state_data(self):
        """Keep MainWindow.app_state data in sync with arithmetic preview/apply state."""
        main_window = getattr(self.fits_viewer, 'main_window', None) or self.fits_viewer
        if hasattr(main_window, 'sync_app_state_data'):
            main_window.sync_app_state_data(
                data=self.fits_viewer.data,
                header=self.fits_viewer.header,
                wcs=self.fits_viewer.wcs,
            )

    def _capture_original_bunit_map(self):
        """Capture original per-window BUNIT state so preview/reset can toggle units safely."""
        self.original_bunit_map = {}
        all_windows = [self.fits_viewer] + [s for s in self.subwindows if s]
        for window in all_windows:
            header = getattr(window, 'header', None)
            if header is None:
                continue
            has_bunit = False
            bunit_value = None
            try:
                has_bunit = 'BUNIT' in header
                if has_bunit:
                    bunit_value = header.get('BUNIT')
            except Exception:
                has_bunit = False
                bunit_value = None
            self.original_bunit_map[window] = (has_bunit, bunit_value)

    def _restore_original_bunit_state(self):
        """Restore original BUNIT state captured before arithmetic preview."""
        if not self.original_bunit_map:
            return
        for window, state in self.original_bunit_map.items():
            header = getattr(window, 'header', None)
            if header is None:
                continue
            had_bunit, bunit_value = state
            try:
                if had_bunit:
                    header['BUNIT'] = bunit_value
                else:
                    header.pop('BUNIT', None)
            except Exception:
                continue

    def _apply_preview_bunit_state(self, preserves_unit: bool):
        """Apply preview-time BUNIT policy and refresh live labels."""
        if not self.original_bunit_map:
            self._capture_original_bunit_map()

        for window, state in self.original_bunit_map.items():
            header = getattr(window, 'header', None)
            if header is None:
                continue
            had_bunit, bunit_value = state
            try:
                if preserves_unit:
                    if had_bunit:
                        header['BUNIT'] = bunit_value
                    else:
                        header.pop('BUNIT', None)
                else:
                    header.pop('BUNIT', None)
            except Exception:
                continue

        self._sync_bunit_to_viewers()
        self._sync_app_state_data()
        self._refresh_live_intensity_labels()

    def _sync_bunit_to_viewers(self):
        """Keep live intensity unit labels synchronized with current header BUNIT."""
        header = getattr(self.fits_viewer, 'header', None)
        bunit = ""
        if header is not None:
            try:
                bunit = str(header.get('BUNIT', '') or '').strip()
            except Exception:
                bunit = ""

        try:
            self.fits_viewer.bunit = bunit
        except Exception:
            pass

        for window in self.subwindows:
            if not window:
                continue
            try:
                window.bunit = bunit
            except Exception:
                continue

    def _refresh_live_intensity_labels(self):
        """Refresh lower-right coordinate/intensity labels without requiring a new click."""
        main_window = getattr(self.fits_viewer, 'main_window', None) or self.fits_viewer
        refresh_fn = getattr(main_window, '_refresh_wcs_display_strings', None)
        if callable(refresh_fn):
            try:
                refresh_fn()
                return
            except Exception:
                pass
        try:
            self.fits_viewer.update_clicked_pix(
                int(self.fits_viewer._get_shared_xpix()),
                int(self.fits_viewer._get_shared_ypix()),
                update_slices=False,
                fast_blit=True,
            )
        except Exception:
            return

    def _expression_preserves_unit(self, expr_str: str) -> bool:
        """Return whether expression keeps original BUNIT semantics."""
        variable_names = list(self.variables.keys())
        if not expr_str or not variable_names:
            return True
        try:
            tree = ast.parse(expr_str, mode='eval')
            checker = UnitPreservationChecker(variable_names)
            checker.visit(tree)
            return checker.preserves_unit
        except Exception:
            # Conservative fallback: unknown expression is treated as unit-changing.
            return False

    def update_all_displays(self):
        """Refreshes the display of the main viewer and all subwindows."""
        all_windows = [self.fits_viewer] + [s for s in self.subwindows if s]
        
        for window in all_windows:
            # Logic borrowed from ScalingPanel / FITSViewer update
            current_channel = 0
            if hasattr(window, 'current_channel_index'):
                try:
                    current_channel = window.current_channel_index()
                except Exception:
                    current_channel = 0
            
            # We use update_channel to properly refresh the slice and overlay
            if hasattr(window, 'update_channel'):
                try:
                    # If 2D (ndim < 3), update_channel is a no-op (due to recent fix).
                    # So we must update the image artist manually for 2D.
                    if window.data.ndim < 3:
                        if hasattr(window, 'im'):
                            window.im.set_data(window.data)                            
                            window.canvas.draw_idle()
                    else:
                        window.update_channel(window.plane, current_channel)
                        
                except Exception as e:
                    print(f"Error updating display for window: {e}")
            elif hasattr(window, 'im'):
                 window.im.set_data(window.data)
                 window.canvas.draw_idle()

    def save_result(self):
        data, header = self.run_computation()
        if data is not None:
            # Copy header to avoid modifying the original
            if header is not None:
                new_header = header.copy()
            else:
                from astropy.io import fits
                # Create a minimal header if none exists
                new_header = fits.Header()
            
            # Check if the operation preserves units
            expr_str = self.expr_input.text().strip()
            preserves_unit = self._expression_preserves_unit(expr_str)
            
            # Handle BUNIT based on operation type
            original_bunit = new_header.get('BUNIT', None)
            if not preserves_unit:
                # Remove BUNIT for operations that change units
                if 'BUNIT' in new_header:
                    del new_header['BUNIT']

            # Update data range if finite values exist; otherwise remove stale values
            finite_mask = np.isfinite(data)
            if np.any(finite_mask):
                finite_data = data[finite_mask]
                new_header['DATAMIN'] = float(finite_data.min())
                new_header['DATAMAX'] = float(finite_data.max())
            else:
                new_header.pop('DATAMIN', None)
                new_header.pop('DATAMAX', None)
            
            # Add HISTORY to header
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            new_header.add_history(f"Arithmetic operation by takefits on {timestamp}")
            new_header.add_history(f"Source file: {self.fits_viewer.filename}")
            new_header.add_history(f"Formula: {expr_str}")
            
            # Record Source Files
            name_a = self.variables.get("A", {}).get("name", "N/A")
            new_header.add_history(f"A: {name_a}")
            
            if "B" in self.variables:
                name_b = self.variables["B"].get("name", "N/A")
                new_header.add_history(f"B: {name_b}")
            
            # Record BUNIT information in HISTORY
            if original_bunit:
                if preserves_unit:
                    new_header.add_history(f"BUNIT preserved: {original_bunit}")
                else:
                    new_header.add_history(f"Original BUNIT removed: {original_bunit} (unit may have changed)")

            for entry in build_processing_history_lines(self.fits_viewer):
                new_header.add_history(entry)
            
            saver = SaveFITS(data, new_header, self.fits_viewer.filename)
            saver.save(suffix="calc")

    def closeEvent(self, event):
        has_pending_preview = bool(
            self.original_data_map
            or self._has_restored_preview
            or self._has_pending_preview_in_history()
        )
        if has_pending_preview:
            choice = confirm_pending_close(
                self,
                "Close Arithmetic Panel",
                "There are unapplied arithmetic preview changes.",
            )
            if choice == "cancel":
                event.ignore()
                return
            if choice == "discard":
                self.reset_preview()
            else:
                self.original_data_map.clear()
                self.original_bunit_map.clear()
                self._has_restored_preview = False
        self.destroyed.emit() 
        super().closeEvent(event)
