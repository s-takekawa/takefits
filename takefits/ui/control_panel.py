# ui/control_panel.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QDialog,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox)
from takefits.tools.color_scale import ColorSettingsPanel
from takefits.tools.unit_conversion_panel import UnitConversionPanel
from takefits.tools.scaling_panel import ScalingPanel
from takefits.tools.integration import IntegSettingsPanel
from takefits.tools.channel_map import ChannelMapSettingPanel
from takefits.tools.baseline import BaselinePanel
from takefits.tools.spectrum import SpecWindow
from takefits.tools.contour_panel import ContourPanel
from takefits.tools.clump_finding_panel import ClumpFindingPanel
from takefits.core.contour_manager import ContourManager

from takefits.tools.smoothing import SmoothSettingsPanel
from takefits.tools.pv_diagram import PVdiagram 
from takefits.tools.masking import MaskSettingsPanel
from takefits.tools.arithmetic import CubeArithmeticPanel

import os
#import json
#import re
#import traceback
try:
    from openai import OpenAI # type: ignore
except ImportError:
    #print("\033[91mError: 'openai' library not found. Please install it using 'pip install openai'.\033[0m")
    OpenAI = None
# --- AI Handler Import ---
from takefits.core.ai_handler import AIHandler
# --- API Key Setup ---
# IMPORTANT: Set your OpenAI API key securely via the OPENAI_API_KEY environment variable.
if OpenAI:
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if not client.api_key:
            raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    except Exception as e:
         print(f"\033[91mAPI Initialization Error: Failed to initialize OpenAI client: {e}\nAI features will be disabled.\033[0m")
         client = None
else:
    client = None

# --- AI Prompt Dialog Class ---
class AIPromptDialog(QDialog):
    """
    A simple dialog window to get natural language input for AI commands.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Command Prompt")
        self.setMinimumWidth(400) # Adjust width as needed

        layout = QVBoxLayout(self)

        # Instruction label
        self.label = QLabel("Enter your command (e.g., 'set color scale 0 to 10'):")
        layout.addWidget(self.label)

        # Input field (QLineEdit for single line)
        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText("Type command here...")
        layout.addWidget(self.prompt_input)

        # OK / Cancel buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch() # Align buttons to the right

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject) # Closes the dialog returning Rejected
        button_layout.addWidget(self.cancel_button)

        self.ok_button = QPushButton("Execute")
        self.ok_button.setDefault(True) # Allows Enter key to trigger
        self.ok_button.clicked.connect(self.accept) # Closes the dialog returning Accepted
        button_layout.addWidget(self.ok_button)

        layout.addLayout(button_layout)

        # Connect Enter key press in QLineEdit to the OK button click
        self.prompt_input.returnPressed.connect(self.ok_button.click)

    def get_prompt_text(self):
        """Return the text entered by the user, stripped of whitespace."""
        return self.prompt_input.text().strip()

# --- ControlPanel Widget Class ---
class ControlPanel(QWidget):
    """
    A widget providing buttons to open various tool panels (Color, Scaling, AI, etc.)
    for the FITS viewer.
    """
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows

        # Initialize references to tool panels (initially None)
        self.color_settings_panel = None
        self.scaling_panel = None
        self.unit_conversion_panel = None
        self.integ_settings_panel = None
        self.chmap_settings_panel = None
        self.smooth_settings_panel = None
        self.baseline_panel = None
        self.spec_window = None
        self.pvd_panel = None
        self.mask_settings_panel = None
        self.contour_panel = None
        self.arithmetic_panel = None
        self.clump_finding_panel = None

        # Instantiate the AI Handler (if OpenAI client is available)
        self.ai_handler = AIHandler(client, self) if client else None

        self.initUI()

    def initUI(self):
        """Initialize the user interface of the control panel."""
        layout = QVBoxLayout()
        layout.setSpacing(8) # Adjust spacing between buttons

        # --- Create and Connect Buttons ---
        # AI Command
        if OpenAI is not None:
            self.ai_command_button = QPushButton('AI Command', self)
            self.ai_command_button.clicked.connect(self._open_ai_prompt_dialog)
            layout.addWidget(self.ai_command_button)
            if self.ai_handler is None:
                self.ai_command_button.setEnabled(False)
                self.ai_command_button.setToolTip("AI features disabled. Check API key/installation.")

        # Arithmetic
        self.arithmetic_button = QPushButton("Arithmetic", self)
        self.arithmetic_button.clicked.connect(self.open_arithmetic_panel)
        layout.addWidget(self.arithmetic_button)

        # Baseline
        self.baseline_button = QPushButton("Baseline", self)
        self.baseline_button.clicked.connect(self.open_baseline_panel)
        layout.addWidget(self.baseline_button)
        if self.fits_viewer.data.ndim < 3:
            self.baseline_button.setEnabled(False)

        # Channel Map
        self.chmap_button = QPushButton('Channel Map', self)
        self.chmap_button.clicked.connect(self.open_chmap_settings)
        layout.addWidget(self.chmap_button)
        if self.fits_viewer.data.ndim < 3:
            self.chmap_button.setEnabled(False)

        # Clump Finding
        self.clump_finding_button = QPushButton('Clump Finding', self)
        self.clump_finding_button.clicked.connect(self.open_clump_finding_panel)
        layout.addWidget(self.clump_finding_button)

        # Color Settings
        self.color_button = QPushButton('Color Settings', self)
        self.color_button.clicked.connect(self.open_color_settings)
        layout.addWidget(self.color_button)

        # Contours
        self.contour_button = QPushButton("Contours", self)
        self.contour_button.clicked.connect(self.open_contour_panel)
        layout.addWidget(self.contour_button)

        # Cut Out
        self.cutout_button = QPushButton("Cut Out", self)
        self.cutout_button.clicked.connect(self.open_cutout_dialog)
        layout.addWidget(self.cutout_button)

        # Integration
        self.integ_button = QPushButton("Integration", self)
        self.integ_button.clicked.connect(self.open_integ_settings)
        layout.addWidget(self.integ_button)
        if self.fits_viewer.data.ndim < 3:
            self.integ_button.setEnabled(False)

        # Markers
        self.marker_button = QPushButton("Markers", self)
        self.marker_button.clicked.connect(self.fits_viewer.open_marker_panel)
        layout.addWidget(self.marker_button)

        # Mask
        self.mask_button = QPushButton("Mask", self)
        self.mask_button.clicked.connect(self.open_mask_settings)
        layout.addWidget(self.mask_button)

        # PV diagram
        self.pvd_button = QPushButton("PV diagram", self)
        self.pvd_button.clicked.connect(self.open_pvd_settings)
        layout.addWidget(self.pvd_button)
        if self.fits_viewer.data.ndim < 3:
            self.pvd_button.setEnabled(False)

        # Regrid
        self.regrid_button = QPushButton("Regrid", self)
        self.regrid_button.clicked.connect(self._open_regrid_panel)
        layout.addWidget(self.regrid_button)

        # Scaling
        self.scaling_button = QPushButton("Scaling", self)
        self.scaling_button.clicked.connect(self.open_scaling_panel)
        layout.addWidget(self.scaling_button)

        # Smoothing
        self.smooth_button = QPushButton("Smoothing", self)
        self.smooth_button.clicked.connect(self.open_smooth_settings)
        layout.addWidget(self.smooth_button)

        # Spectrum
        self.spec_button = QPushButton("Spectrum", self)
        self.spec_button.clicked.connect(self.open_spec_window)
        layout.addWidget(self.spec_button)
        if self.fits_viewer.data.ndim < 3:
            self.spec_button.setEnabled(False)

        # Unit Conversion
        self.unit_conversion_button = QPushButton("Unit Conversion", self)
        self.unit_conversion_button.clicked.connect(self.open_unit_conversion_panel)
        layout.addWidget(self.unit_conversion_button)
        # --- End Button Creation ---

        layout.addStretch() # Push buttons to the top

        self.setLayout(layout)
        self.setWindowTitle(f'Tools:{self.fits_viewer.filename}')

        self.adjustSize() # Adjust window size to fit content
        self.move_to_default_position() # Position window relative to main viewer
        self.show()

    def move_to_default_position(self):
        """Positions the control panel to the right of the main FITS viewer window."""
        try:
            mainwindow_geometry = self.fits_viewer.geometry()
            mainwindow_x = mainwindow_geometry.x()
            mainwindow_y = mainwindow_geometry.y()
            mainwindow_width = mainwindow_geometry.width()
            # Position slightly to the right and vertically aligned (adjust y offset as needed)
            self.move(mainwindow_x + mainwindow_width, mainwindow_y + 148)
        except Exception as e:
            print(f"Could not position control panel: {e}") # Handle cases where fits_viewer might not be ready

    # --- AI Command Handling ---
    def _open_ai_prompt_dialog(self):
        """Opens the AI prompt dialog and delegates processing to AIHandler."""
        if self.ai_handler is None:
            QMessageBox.warning(self, "AI Error", "AI features are currently disabled.")
            return

        dialog = AIPromptDialog(self) # Use ControlPanel as parent
        if dialog.exec() == QDialog.DialogCode.Accepted:
            prompt_text = dialog.get_prompt_text()
            if prompt_text:
                # Call AI Handler methods for interpretation and dispatching
                interpretation = self.ai_handler.get_command_interpretation(prompt_text)
                if interpretation:
                    # Pass the ControlPanel instance (self) for context if needed by handlers
                    self.ai_handler.dispatch_ai_command(interpretation)
            else:
                print("AI Prompt was empty.")
                QMessageBox.information(self, "Empty Command", "No command entered.")
        else:
            print("AI Prompt cancelled.")

    def _open_regrid_panel(self):
        """Open the regrid panel from the control panel."""
        if hasattr(self.fits_viewer, "open_regrid_panel"):
            self.fits_viewer.open_regrid_panel()

    # --- Methods to Open Tool Panels ---
    # These methods open the respective tool panels, ensuring only one instance exists.

    def open_chmap_settings(self):
        """Opens the Channel Map settings panel."""
        needs_new_panel = False
        if self.chmap_settings_panel is None:
            needs_new_panel = True
        else:
            try:
                needs_new_panel = not self.chmap_settings_panel.isVisible()
            except Exception:
                # Widget may already be deleted
                self.chmap_settings_panel = None
                needs_new_panel = True

        if needs_new_panel:
            self.chmap_settings_panel = ChannelMapSettingPanel(self.fits_viewer, self.subwindows)
            try:
                self.chmap_settings_panel.destroyed.disconnect(self.on_chmap_settings_closed)
            except Exception:
                pass
            self.chmap_settings_panel.destroyed.connect(self.on_chmap_settings_closed)

        self.chmap_settings_panel.show()
        self.chmap_settings_panel.raise_()
        self.chmap_settings_panel.activateWindow()

    def on_chmap_settings_closed(self):
        """Slot called when Channel Map panel is closed."""
        self.chmap_settings_panel = None

    def open_clump_finding_panel(self):
        """Opens the Clump Finding panel."""
        if self.clump_finding_panel is None:
            self.clump_finding_panel = ClumpFindingPanel(self.fits_viewer, self.subwindows)
            self.clump_finding_panel.show()
            try: self.clump_finding_panel.destroyed.disconnect(self.on_clump_finding_panel_closed)
            except TypeError: pass
            self.clump_finding_panel.destroyed.connect(self.on_clump_finding_panel_closed)
        else:
            self.clump_finding_panel.raise_()
            self.clump_finding_panel.activateWindow()

    def on_clump_finding_panel_closed(self):
        """Slot called when Clump Finding panel is closed."""
        self.clump_finding_panel = None

    def open_color_settings(self):
        """Opens the Color Settings panel."""
        if self.color_settings_panel is None:
            seed = getattr(self.fits_viewer, "_seed_main_color_panel_settings_from_current_image", None)
            if callable(seed):
                try:
                    seed()
                except Exception:
                    pass
            # Pass fits_viewer and subwindows to the panel
            self.color_settings_panel = ColorSettingsPanel(self.fits_viewer, self.subwindows)
            self.color_settings_panel.show()
            try: self.color_settings_panel.destroyed.disconnect(self.on_color_settings_closed)
            except TypeError: pass
            self.color_settings_panel.destroyed.connect(self.on_color_settings_closed)
        else:
            self.color_settings_panel.raise_()
            self.color_settings_panel.activateWindow()

    def on_color_settings_closed(self):
        """Slot called when Color Settings panel is closed."""
        updater = getattr(self.fits_viewer, "_update_main_color_panel_hint", None)
        if callable(updater):
            try:
                updater()
            except Exception:
                pass
        self.color_settings_panel = None

    def open_scaling_panel(self):
        """Opens the Scaling settings panel."""
        if self.scaling_panel is None:
            self.scaling_panel = ScalingPanel(self.fits_viewer, self.subwindows)
            self.scaling_panel.show()
            try: self.scaling_panel.destroyed.disconnect(self.on_scaling_panel_closed)
            except TypeError: pass
            self.scaling_panel.destroyed.connect(self.on_scaling_panel_closed)
        else:
            self.scaling_panel.raise_()
            self.scaling_panel.activateWindow()

    def on_scaling_panel_closed(self):
        """Slot called when Scaling panel is closed."""
        self.scaling_panel = None

    def open_unit_conversion_panel(self):
        """Opens the Unit Conversion settings panel."""
        if self.unit_conversion_panel is None:
            self.unit_conversion_panel = UnitConversionPanel(self.fits_viewer, self.subwindows)
            self.unit_conversion_panel.show()
            try: self.unit_conversion_panel.destroyed.disconnect(self.on_unit_conversion_panel_closed)
            except TypeError: pass
            self.unit_conversion_panel.destroyed.connect(self.on_unit_conversion_panel_closed)
        else:
            self.unit_conversion_panel.raise_()
            self.unit_conversion_panel.activateWindow()

    def on_unit_conversion_panel_closed(self):
        """Slot called when Unit Conversion panel is closed."""
        self.unit_conversion_panel = None

    def open_pvd_settings(self):
        """Opens the PV Diagram panel."""
        if self.pvd_panel is None:
            self.pvd_panel = PVdiagram(self.fits_viewer)
            self.pvd_panel.show()
            try: self.pvd_panel.destroyed.disconnect(self.on_pvd_closed)
            except TypeError: pass
            self.pvd_panel.destroyed.connect(self.on_pvd_closed)
        else:
            self.pvd_panel.raise_()
            self.pvd_panel.activateWindow()

    def on_pvd_closed(self):
        """Slot called when PV Diagram panel is closed."""
        self.pvd_panel = None

    def open_integ_settings(self):
        """Opens the Integration settings panel."""
        panel = self.integ_settings_panel
        if panel is not None:
            try:
                visible = panel.isVisible()
            except RuntimeError:
                panel = None
                self.integ_settings_panel = None
            else:
                if not visible:
                    panel.show()
                panel.raise_()
                panel.activateWindow()
                return
        self.integ_settings_panel = IntegSettingsPanel(self.fits_viewer, self.subwindows)
        self.integ_settings_panel.show()
        try:
            self.integ_settings_panel.destroyed.disconnect(self.on_integ_settings_closed)
        except TypeError:
            pass
        self.integ_settings_panel.destroyed.connect(self.on_integ_settings_closed)

    def on_integ_settings_closed(self):
        """Slot called when Integration panel is closed."""
        self.integ_settings_panel = None


    def open_cutout_dialog(self):
        self.fits_viewer.open_cutout_dialog(use_view_bounds=True)

    def open_mask_settings(self):
        if self.mask_settings_panel is None:
            self.mask_settings_panel = MaskSettingsPanel(self.fits_viewer, self.subwindows)
            self.mask_settings_panel.destroyed.connect(self.on_mask_settings_closed)
            self.mask_settings_panel.show()
        else:
            self.mask_settings_panel.raise_()
            self.mask_settings_panel.activateWindow()

    def on_mask_settings_closed(self):
        self.mask_settings_panel = None

    def open_spec_window(self):
        """Opens the Spectrum window."""
        data = getattr(self.fits_viewer, "data", None)
        if data is None or int(getattr(data, "ndim", 0)) < 3:
            QMessageBox.information(
                self,
                "Spectrum",
                "Spectrum is available only for 3D/4D cube data.",
            )
            return
        if self.spec_window is None:
            self.spec_window = SpecWindow(self.fits_viewer)
            
            # --- Signal Wiring (Refactoring Pilot) ---
            # Connect FITSViewer position update to SpecWindow
            if hasattr(self.fits_viewer, 'position_updated'):
                self.fits_viewer.position_updated.connect(self.spec_window.update_position)

            # Connect SpecWindow channel change to FITSViewer slider and subwindows cursor
            self.spec_window.channel_changed.connect(self.broadcast_channel_update)

            # Connect SubWindows position update to SpecWindow
            if self.subwindows:
                for sub in self.subwindows:
                    if sub and hasattr(sub, 'position_updated'):
                        sub.position_updated.connect(self.spec_window.update_position)

            self.spec_window.show()

            try: self.spec_window.destroyed.disconnect(self.on_spec_window_closed)
            except TypeError: pass
            self.spec_window.destroyed.connect(self.on_spec_window_closed)
        else:
            self.spec_window.raise_()
            self.spec_window.activateWindow()

    def broadcast_channel_update(self, channel):
        """
        Broadcasting channel changes to all viewers.
        Updates the main slider (which updates the XY slice) and 
        explicitly updates the crosshair cursor on XZ/ZY planes to match the new channel (Z-coordinate).
        """
        # 1. Update Main Window (XY Plane) Slider -> Changes Image Slice
        if hasattr(self.fits_viewer, 'slider'):
             self.fits_viewer.slider.setValue(channel)
        
        # 2. Update SubWindows (XZ / ZY Planes) Cursor Line
        if self.subwindows:
            for i, sub in enumerate(self.subwindows):
                if not sub:
                    continue

                plane = getattr(sub, 'plane', None)
                if plane == 'xz':
                     if hasattr(sub, '_update_plane_cursor'):
                         sub._update_plane_cursor('xz', y=channel)
                         # Explicitly update the artist
                         if hasattr(sub, 'hline'):
                             sub.hline.set_ydata([channel])
                             
                         if hasattr(sub, 'redraw_main_overlay_and_blit'):
                             sub.redraw_main_overlay_and_blit()
                
                elif plane == 'zy':
                     if hasattr(sub, '_update_plane_cursor'):
                         sub._update_plane_cursor('zy', x=channel)
                         # Explicitly update the artist
                         if hasattr(sub, 'vline'):
                             sub.vline.set_xdata([channel])
                             
                         if hasattr(sub, 'redraw_main_overlay_and_blit'):
                             sub.redraw_main_overlay_and_blit()

    def on_spec_window_closed(self):
        """Slot called when Spectrum window is closed."""
        # Cleanup connections is handled by Qt when widget is destroyed
        self.spec_window = None

    def open_baseline_panel(self):
        """Opens the Baseline panel."""
        data = getattr(self.fits_viewer, "data", None)
        if data is None or int(getattr(data, "ndim", 0)) < 3:
            QMessageBox.information(
                self,
                "Baseline",
                "Baseline is available only for 3D/4D cube data.",
            )
            return
        if self.baseline_panel is None:
            self.baseline_panel = BaselinePanel(self.fits_viewer, self.subwindows)

            if hasattr(self.fits_viewer, 'position_updated'):
                self.fits_viewer.position_updated.connect(self.baseline_panel.update_position)

            if self.subwindows:
                for sub in self.subwindows:
                    if sub and hasattr(sub, 'position_updated'):
                        sub.position_updated.connect(self.baseline_panel.update_position)

            self.baseline_panel.show()
            try:
                self.baseline_panel.destroyed.disconnect(self.on_baseline_panel_closed)
            except TypeError:
                pass
            self.baseline_panel.destroyed.connect(self.on_baseline_panel_closed)
        else:
            self.baseline_panel.raise_()
            self.baseline_panel.activateWindow()

    def on_baseline_panel_closed(self):
        """Slot called when Baseline panel is closed."""
        self.baseline_panel = None

    def open_smooth_settings(self):
        """Opens the Smoothing settings panel."""
        if self.smooth_settings_panel is None:
            self.smooth_settings_panel = SmoothSettingsPanel(self.fits_viewer, self.subwindows)
            self.smooth_settings_panel.show()
            try: self.smooth_settings_panel.destroyed.disconnect(self.on_smooth_settings_closed)
            except TypeError: pass
            self.smooth_settings_panel.destroyed.connect(self.on_smooth_settings_closed)
        else:
            self.smooth_settings_panel.raise_()
            self.smooth_settings_panel.activateWindow()

    def open_contour_panel(self):
        if self.contour_panel is None:
            # Ensure primary viewers are registered before inspecting manager targets.
            register = getattr(self.fits_viewer, "_register_contour_layer", None)
            if callable(register):
                register()
            for sub in self.subwindows or []:
                sub_register = getattr(sub, "_register_contour_layer", None)
                if callable(sub_register):
                    sub_register()

            default_targets = ContourManager.instance().layer_ids_for_owner(self.fits_viewer)
            if not default_targets:
                # Fallback to any layer if main viewer not registered yet.
                registered = ContourManager.instance().registered_layers()
                if registered:
                    default_targets = [next(iter(registered.keys()))]
            self.contour_panel = ContourPanel(self, default_targets=default_targets)
            self.contour_panel.destroyed.connect(self.on_contour_panel_closed)
            self.contour_panel.show()
        else:
            self.contour_panel.raise_()
            self.contour_panel.activateWindow()

    def on_contour_panel_closed(self):
        self.contour_panel = None

    def on_smooth_settings_closed(self):
        """Slot called when Smoothing panel is closed."""
        self.smooth_settings_panel = None

    def open_arithmetic_panel(self):
        """Opens the Cube Arithmetic panel."""
        if self.arithmetic_panel is None:
            self.arithmetic_panel = CubeArithmeticPanel(self.fits_viewer, self.subwindows)
            self.arithmetic_panel.show()
            try: self.arithmetic_panel.destroyed.disconnect(self.on_arithmetic_panel_closed)
            except TypeError: pass
            self.arithmetic_panel.destroyed.connect(self.on_arithmetic_panel_closed)
        else:
            self.arithmetic_panel.raise_()
            self.arithmetic_panel.activateWindow()
            
    def on_arithmetic_panel_closed(self):
        self.arithmetic_panel = None

    def closeEvent(self, event):
        """Handles the close event for the ControlPanel."""
        # Update the state of the corresponding action in the main window's menu bar
        if hasattr(self.fits_viewer, 'menu_bar') and self.fits_viewer.menu_bar:
            self.fits_viewer.menu_bar.control_panel_action.setChecked(False)
        # Ensure all associated tool panels are closed when this panel closes
        panels_to_close = [
            self.color_settings_panel, self.scaling_panel, self.unit_conversion_panel,
            self.integ_settings_panel, self.chmap_settings_panel,
            self.smooth_settings_panel, self.baseline_panel, self.spec_window, self.pvd_panel,
            self.contour_panel, self.arithmetic_panel, self.clump_finding_panel
        ]
        for panel in panels_to_close:
            if panel:
                try:
                    panel.close()
                except Exception as e:
                    print(f"Error closing panel {type(panel).__name__}: {e}")
        super().closeEvent(event)
