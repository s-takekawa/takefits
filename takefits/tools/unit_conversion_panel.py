import numpy as np
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel,
    QHBoxLayout, QMessageBox, QComboBox, QFormLayout, QGroupBox,
)
from PySide6.QtCore import Qt
from takefits.ui.save_fits_dialog import SaveFITS
from datetime import datetime

# Import constants for Planck's law
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from takefits.core.io.save_fits import update_datamin_datamax_if_present
from takefits.core.usecases import convert_intensity_unit
from takefits.core.history_provenance import build_processing_history_lines
from takefits.logic.data_tools import create_preview_snapshot, is_lazy_scaled
from takefits.tools.base_panel import (
    capture_preferred_cursor_snapshot,
    clear_action_preview_record,
    confirm_pending_close,
    has_action_record_tag,
    record_action_preview,
    replay_action_history_to_current_cursor,
)

class UnitConversionPanel(QWidget):
    """
    A panel for performing unit conversions
    (e.g., Jy/beam <-> K) for radio astronomy data cubes.
    
    It also allows editing key FITS header values (Rest Freq, BMAJ, BMIN, BPA).
    UI is based on QGroupBoxes, similar to masking.py.
    Size is fixed to prevent unwanted resizing.
    """
    
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows
        self.original_data = None  # Defer copying data
        self.original_header = self.fits_viewer.header.copy() # Store original header for reset
        self.original_bunit = self._get_normalized_bunit(self.original_header)
        self._conversion_baseline_bunit = None
        self._conversion_baseline_has_bunit = False
        self.scaling_reference_data = None  # Baseline for manual scaling
        self._history_categories = {}  # Track latest history entries per category
        self._history_dedup_keys = set()
        self.spectral_metadata = getattr(self.fits_viewer, 'spectral_metadata', {})
        self._spectral_axis_index = self.spectral_metadata.get('axis_index')
        self._conversion_action_record_tag = "panel:unit-conversion"

        self._disk_header = self._load_disk_header()
        disk_axis_info = self._extract_disk_spectral_axis_info(self._disk_header)
        self._disk_spectral_axis_index = disk_axis_info.get('index')
        self._disk_spectral_ctype = disk_axis_info.get('ctype')
        self._disk_spectral_cunit = disk_axis_info.get('cunit')
        original_axis_type_meta = str(self.spectral_metadata.get('original_axis_type', '')).lower()
        if original_axis_type_meta in ('frequency', 'velocity'):
            self._disk_axis_was_frequency = (original_axis_type_meta == 'frequency')
        else:
            self._disk_axis_was_frequency = disk_axis_info.get('is_frequency', False)

        #current_axis_type_meta = str(self.spectral_metadata.get('current_axis_type', '')).lower()

        if getattr(self.fits_viewer.data, 'ndim', 0) <= 2:
            self.spectral_axis_mode = 'velocity'
        #elif current_axis_type_meta in ('frequency', 'velocity'):
        #    self.spectral_axis_mode = 'frequency' if current_axis_type_meta == 'frequency' else 'velocity'
        else:
            # 3D/4D cube: Default to the axis type of the *original* FITS file on disk
            self.spectral_axis_mode = 'frequency' if self._disk_axis_was_frequency else 'velocity'

        self._update_spectral_metadata_from_header()

        # --- Internal state for UI and calculations ---
        self.current_bunit = self.original_bunit
        self.header_modified = False # Track if header values have been changed
        
        # --- Check header for conversion capabilities ---
        self.header_params = self._check_header_keys()
        self.header_info_strings = self._get_header_info_strings() # Get strings for UI
        
        # --- Constants for calculations ---
        self.beam_area_sr = self._calculate_beam_area_sr()
        self.pixel_area_sr = self._calculate_pixel_area_sr()

        # Initialize UI elements
        self._init_ui_elements()
        self.initUI()
        
        # Set a fixed size based on the minimal initial layout
        self.setFixedSize(self.sizeHint())
        self.resync_after_workspace_restore()

    def _load_disk_header(self):
        filename_path = getattr(self.fits_viewer, 'filename_path', None)
        if not filename_path:
            return None
        try:
            return fits.getheader(filename_path)
        except Exception as err:
            print(f"Warning: Could not read original FITS header from disk: {err}")
            return None

    def _extract_disk_spectral_axis_info(self, header):
        info = {'index': None, 'ctype': None, 'cunit': None, 'is_frequency': False}
        if header is None:
            return info

        try:
            naxis = int(header.get('NAXIS', 0))
        except (TypeError, ValueError):
            naxis = 0

        for axis in range(1, naxis + 1):
            ctype = str(header.get(f'CTYPE{axis}', '')).strip()
            if not ctype:
                continue
            ctype_upper = ctype.upper()
            if 'FREQ' in ctype_upper:
                cunit = str(header.get(f'CUNIT{axis}', '')).strip()
                info.update(index=axis, ctype=ctype, cunit=cunit or 'Hz', is_frequency=True)
                return info

        for axis in range(1, naxis + 1):
            ctype = str(header.get(f'CTYPE{axis}', '')).strip()
            if not ctype:
                continue
            ctype_upper = ctype.upper()
            if any(tag in ctype_upper for tag in ('VRAD', 'VELO', 'VOPT')):
                cunit = str(header.get(f'CUNIT{axis}', '')).strip()
                info.update(index=axis, ctype=ctype, cunit=cunit or 'km/s', is_frequency=False)
                return info

        return info

    def _find_velocity_axis_in_header(self, header):
        """Return (axis_index, ctype, cunit) for a velocity-like axis if present."""
        info = (None, None, None)
        if header is None:
            return info

        try:
            naxis = int(header.get('NAXIS', 0))
        except (TypeError, ValueError):
            naxis = 0

        for axis in range(1, naxis + 1):
            ctype = str(header.get(f'CTYPE{axis}', '')).strip()
            if not ctype:
                continue
            ctype_upper = ctype.upper()
            if any(tag in ctype_upper for tag in ('VRAD', 'VELO', 'VOPT')):
                cunit = str(header.get(f'CUNIT{axis}', '')).strip()
                return axis, ctype, cunit

        return info

    @staticmethod
    def _classify_axis_mode(ctype):
        """Classify an axis CTYPE into 'frequency', 'velocity', or 'unknown'."""
        if not ctype:
            return 'unknown'
        ctype_upper = str(ctype).upper()
        if 'FREQ' in ctype_upper:
            return 'frequency'
        if any(tag in ctype_upper for tag in ('VRAD', 'VELO', 'VOPT')):
            return 'velocity'
        return 'unknown'

    def _update_spectral_metadata_from_header(self, header=None):
        """Synchronize shared spectral metadata with the latest header state."""
        if not isinstance(self.spectral_metadata, dict):
            return

        header = header or self.fits_viewer.header

        axis_index = (
            self._spectral_axis_index
            or self.spectral_metadata.get('axis_index')
            or self._disk_spectral_axis_index
        )

        if axis_index is not None:
            ctype = header.get(f'CTYPE{axis_index}', '')
            unit = header.get(f'CUNIT{axis_index}', '')
            mode = self._classify_axis_mode(ctype)
            self._spectral_axis_index = axis_index
            self.spectral_metadata['axis_index'] = axis_index
            self.spectral_metadata['current_axis_ctype'] = ctype
            self.spectral_metadata['current_axis_type'] = mode
            if isinstance(unit, str) and unit.strip():
                self.spectral_metadata['current_axis_unit'] = unit.strip()

        restfreq = header.get('RESTFRQ', header.get('RESTFREQ'))
        try:
            restfreq_value = float(restfreq)
        except (TypeError, ValueError):
            restfreq_value = None

        self.spectral_metadata['restfreq_hz'] = restfreq_value
        if self.spectral_metadata.get('restfreq_original_hz') is None:
            self.spectral_metadata['restfreq_original_hz'] = restfreq_value

    def _init_ui_elements(self):
        """Initialize UI widgets to allow access from multiple methods."""
        self.header_group = QGroupBox("Header Information (Freq/Beam/3rd Axis)")
        self.rf_input = QLineEdit()
        
        # Group for Beam Properties
        self.beam_group = QGroupBox("Beam Properties")
        self.bmaj_input = QLineEdit()
        self.bmin_input = QLineEdit()
        self.bpa_input = QLineEdit()

        self.set_button = QPushButton("Set")
        self.reset_header_button = QPushButton("Reset Header")
        self.spectral_axis_combo = QComboBox()
        self.spectral_axis_combo.addItem("Velocity", userData='velocity')
        self.spectral_axis_combo.addItem("Frequency", userData='frequency')

        # Group 3: Unit Conversion
        self.conversion_group = QGroupBox("Intensity Unit Conversion")
        self.current_unit_label = QLabel(self.current_bunit.upper())
        self.target_unit_combo = QComboBox(self)
        self.target_unit_combo.setFixedWidth(150)
        self.method_label = QLabel("Method:")
        self.method_combo = QComboBox(self)
        self.apply_conversion_button = QPushButton("Apply Unit Conversion")
        
        # Common buttons
        self.reset_scaling_button = QPushButton('Reset', self)
        self.reset_scaling_button.setEnabled(False)
        self.save_button = QPushButton('Save as FITS') 

        self.save_button.setEnabled(False) # Initially disabled

    def initUI(self):
        """
        Initialize the User Interface with QGroupBoxes for scaling modes.
        """
        main_layout = QVBoxLayout()

        # --- Build Group 3: Unit Conversion (Moved up)
        conversion_layout = QFormLayout(self.conversion_group)
        conversion_layout.setContentsMargins(5, 10, 5, 10)
        conversion_layout.setVerticalSpacing(5)
        conversion_layout.addRow(QLabel("Current Unit:"), self.current_unit_label)
        conversion_layout.addRow(QLabel("Target Unit:"), self.target_unit_combo)
        
        self.method_combo.addItems(["Rayleigh-Jeans", "Planck"])
        conversion_layout.addRow(self.method_label, self.method_combo)

        conversion_button_layout = QHBoxLayout()
        conversion_button_layout.addWidget(self.apply_conversion_button)
        conversion_button_layout.addStretch()
        conversion_button_layout.addWidget(self.reset_scaling_button)
        conversion_layout.addRow(conversion_button_layout)
        
        self.method_label.setEnabled(False)
        self.method_combo.setEnabled(False)
        
        main_layout.addWidget(self.conversion_group)

        # --- Build Group 2: Header Information ---
        header_layout = QFormLayout(self.header_group)
        header_layout.setContentsMargins(10, 10, 10, 10)
        header_layout.setVerticalSpacing(5) # Reduce vertical spacing
        header_layout.addRow(QLabel("RestFreq (GHz):"), self.rf_input)
        header_layout.addRow(QLabel("BMAJ (arcsec):"), self.bmaj_input)
        header_layout.addRow(QLabel("BMIN (arcsec):"), self.bmin_input)
        header_layout.addRow(QLabel("BPA (deg):"), self.bpa_input)
        header_layout.addRow(QLabel("Spectral Axis:"), self.spectral_axis_combo)

        header_button_layout = QHBoxLayout()
        header_button_layout.addWidget(self.set_button)
        header_button_layout.addStretch()
        header_button_layout.addWidget(self.reset_header_button)
        header_layout.addRow(header_button_layout)

        main_layout.addWidget(self.header_group)
        
        self._update_header_fields()
        self._populate_conversion_options()
        self._toggle_method_combo(self.target_unit_combo.currentText())
        self._sync_spectral_axis_ui()

        save_button_layout = QHBoxLayout()
        save_button_layout.addStretch(1)
        save_button_layout.addWidget(self.save_button)
        main_layout.addLayout(save_button_layout)

        main_layout.addStretch(1)


        self.setLayout(main_layout)
        self.setWindowTitle(f'Unit Convert: {self.fits_viewer.filename}')
        self.move_to_default_position()

        # Connect Signals
        self.apply_conversion_button.clicked.connect(self.apply_unit_conversion)
        self.reset_scaling_button.clicked.connect(self.reset_conversion_operations)
        self.save_button.clicked.connect(self.save_fits)
        self.target_unit_combo.currentTextChanged.connect(self._toggle_method_combo)
        
        self.set_button.clicked.connect(self.update_header_values)
        self.reset_header_button.clicked.connect(self.reset_header_values)
        self.rf_input.returnPressed.connect(self.update_header_values)
        self.bmaj_input.returnPressed.connect(self.update_header_values)
        self.bmin_input.returnPressed.connect(self.update_header_values)
        self.bpa_input.returnPressed.connect(self.update_header_values)
        self.spectral_axis_combo.currentIndexChanged.connect(self._on_spectral_axis_mode_changed)

    def move_to_default_position(self):
        """Move the panel next to the main control panel."""
        if hasattr(self.fits_viewer, 'control_panel'):
            control_panel_geometry = self.fits_viewer.control_panel.geometry()
            self.move(control_panel_geometry.x() + control_panel_geometry.width(), control_panel_geometry.y())

    # ------------------------------------------------------------------
    # Header and Capability Checks
    # ------------------------------------------------------------------

    def _get_normalized_bunit(self, header):
        """Get and normalize the BUNIT from the header."""
        bunit = header.get('BUNIT', 'unknown').strip().lower()
        if bunit == 'jy/beam':
            return 'jy/beam'
        elif bunit == 'k':
            return 'k'
        elif bunit in ('jy/pixel', 'jy/pix'):
            return 'jy/pix'
        return 'unknown'

    def _sync_bunit_to_viewers(self):
        """Keep live intensity unit labels synchronized with header BUNIT."""
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

    def _check_header_keys(self):
        """
        Check the header for all keys required for conversions.
        Returns a dictionary of boolean flags.
        """
        header = self.fits_viewer.header # Always use the current header
        
        def _check_freq_axis(header):
            """Robust check for a valid frequency axis."""
            if 'RESTFRQ' in header:
                return True
            naxis = header.get('NAXIS', 0)
            for i in range(1, naxis + 1):
                ctype = header.get(f'CTYPE{i}', '')
                if 'FREQ' in ctype:
                    return True
                if 'VELO' in ctype or 'VRAD' in ctype:
                    break
            if naxis < 3 and 'CRVAL3' in header and 'FREQ' in header.get('CTYPE3', ''):
                 return True
            return False

        params = {
            'has_bmaj_bmin': 'BMAJ' in header and 'BMIN' in header,
            'has_cdelt': 'CDELT1' in header and 'CDELT2' in header,
            'has_freq_axis': _check_freq_axis(header)
        }
        
        self.can_convert_k = params['has_bmaj_bmin'] and params['has_freq_axis']
        self.can_convert_pix = params['has_bmaj_bmin'] and params['has_cdelt']
        
        return params
        
    def _get_header_info_strings(self):
        """Get header values as strings for the UI labels."""
        header = self.fits_viewer.header # Always use the current header
        info = {}
        
        try:
            info['bmaj'] = f"{header['BMAJ'] * 3600.0:.3f}"
            info['bmin'] = f"{header['BMIN'] * 3600.0:.3f}"
            info['bpa'] = f"{header['BPA']:.3f}"
        except Exception:
            info['bmaj'] = info.get('bmaj', "N/A")
            info['bmin'] = info.get('bmin', "N/A")
            info['bpa'] = info.get('bpa', "N/A")
            
        try:
            freq_hz = None
            if 'RESTFRQ' in header:
                freq_hz = header['RESTFRQ']
            else:
                naxis = header.get('NAXIS', 0)
                for i in range(1, naxis + 1):
                    ctype = header.get(f'CTYPE{i}', '')
                    if 'FREQ' in ctype:
                        freq_hz = header.get(f'CRVAL{i}')
                        break
                    if 'VELO' in ctype or 'VRAD' in ctype:
                        info['restfreq'] = "N/A"
                        break
            
            if freq_hz is not None:
                info['restfreq'] = f"{freq_hz / 1e9:.6f}"
            elif 'restfreq' not in info:
                info['restfreq'] = "N/A"
                
        except Exception:
            info['restfreq'] = "N/A (Error)"
            
        return info

    def _calculate_beam_area_sr(self):
        """Calculate beam solid angle in steradians. Returns 0 if not possible."""
        if not self.header_params['has_bmaj_bmin']:
            return 0.0
        try:
            bmaj_deg = self.fits_viewer.header['BMAJ']
            bmin_deg = self.fits_viewer.header['BMIN']
            bmaj_rad = np.deg2rad(bmaj_deg)
            bmin_rad = np.deg2rad(bmin_deg)
            return (np.pi * bmaj_rad * bmin_rad) / (4.0 * np.log(2.0))
        except Exception:
            return 0.0

    def _calculate_pixel_area_sr(self):
        """Calculate pixel solid angle in steradians. Returns 0 if not possible."""
        if not self.header_params['has_cdelt']:
            return 0.0
        try:
            cdelt1_deg = self.fits_viewer.header['CDELT1']
            cdelt2_deg = self.fits_viewer.header['CDELT2']
            cdelt1_rad = np.deg2rad(cdelt1_deg)
            cdelt2_rad = np.deg2rad(cdelt2_deg)
            return np.abs(cdelt1_rad * cdelt2_rad)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # UI Logic
    # ------------------------------------------------------------------

    def _update_header_fields(self):
        """Populate header input fields from current info strings."""
        self.header_info_strings = self._get_header_info_strings()
        self.rf_input.setText(self.header_info_strings.get('restfreq', 'N/A'))
        self.bmaj_input.setText(self.header_info_strings.get('bmaj', 'N/A'))
        self.bmin_input.setText(self.header_info_strings.get('bmin', 'N/A'))
        self.bpa_input.setText(self.header_info_strings.get('bpa', 'N/A'))
        self._sync_spectral_axis_ui()

    def _set_field_set_style(self, field: QLineEdit, is_set: bool):
        """Update the style and read-only state of a field to show if it's set."""
        field.setReadOnly(is_set)
        if is_set:
            field.setStyleSheet("background-color: #e6ffed; color: #555;") # Light green, grey text
        else:
            field.setStyleSheet("") # Default style

    def _populate_conversion_options(self):
        """
        Populate the target unit combo box based on the current unit,
        disable options if header keys are missing, and set a default target.
        """
        self.target_unit_combo.clear() 
        
        default_target_userdata = None
        
        if self.current_bunit == 'jy/beam':
            self.target_unit_combo.addItem("K (Brightness Temp)", userData='k')
            self.target_unit_combo.addItem("Jy/pixel", userData='jy/pix')
            self._set_option_enabled('k', self.can_convert_k)
            self._set_option_enabled('jy/pix', self.can_convert_pix)
            if self.can_convert_k: default_target_userdata = 'k'
            elif self.can_convert_pix: default_target_userdata = 'jy/pix'
                
        elif self.current_bunit == 'k':
            self.target_unit_combo.addItem("Jy/beam", userData='jy/beam')
            self.target_unit_combo.addItem("Jy/pixel", userData='jy/pix')
            self._set_option_enabled('jy/beam', self.can_convert_k)
            self._set_option_enabled('jy/pix', self.can_convert_k and self.can_convert_pix)
            if self.can_convert_k: default_target_userdata = 'jy/beam'
            elif self.can_convert_k and self.can_convert_pix: default_target_userdata = 'jy/pix'

        elif self.current_bunit == 'jy/pix':
            self.target_unit_combo.addItem("Jy/beam", userData='jy/beam')
            self.target_unit_combo.addItem("K (Brightness Temp)", userData='k')
            self._set_option_enabled('jy/beam', self.can_convert_pix)
            self._set_option_enabled('k', self.can_convert_pix and self.can_convert_k)
            if self.can_convert_pix and self.can_convert_k: default_target_userdata = 'k'
            elif self.can_convert_pix: default_target_userdata = 'jy/beam'

        if default_target_userdata:
            index_to_select = self.target_unit_combo.findData(default_target_userdata)
            if index_to_select != -1:
                self.target_unit_combo.setCurrentIndex(index_to_select)

        has_enabled_option = any(self.target_unit_combo.model().item(i).isEnabled() for i in range(self.target_unit_combo.count()))

        # Overall check for enabling the conversion group
        can_convert = has_enabled_option and self.current_bunit != 'unknown'
        self.target_unit_combo.setEnabled(can_convert)
        self.apply_conversion_button.setEnabled(can_convert)
        # The method combo's enabled state is handled by _toggle_method_combo

        if not can_convert:
            tooltip = "Current unit (BUNIT) is unknown or unsupported." if self.current_bunit == 'unknown' else "Required FITS header keys are missing or not set."
            self.conversion_group.setToolTip(tooltip)
            self.target_unit_combo.setCurrentIndex(-1)
            self.method_label.setEnabled(False)
            self.method_combo.setEnabled(False)
        else:
            self.conversion_group.setToolTip("")
            # After enabling, specifically check if method combo should be shown
            self._toggle_method_combo(self.target_unit_combo.currentText())

    def _sync_spectral_axis_ui(self):
        """Update spectral axis combo state based on data dimensionality and mode."""
        if not hasattr(self, 'spectral_axis_combo'):
            return

        if getattr(self.fits_viewer.data, 'ndim', 0) <= 2:
            self.spectral_axis_combo.setEnabled(False)
            self.spectral_axis_combo.setToolTip("Spectral axis options require 3D/4D data.")
            self.spectral_axis_combo.blockSignals(True)
            self.spectral_axis_combo.setCurrentIndex(0)
            self.spectral_axis_combo.blockSignals(False)
            return

        self.spectral_axis_combo.setEnabled(True)
        self.spectral_axis_combo.setToolTip("")

        desired_mode = getattr(self, 'spectral_axis_mode', 'velocity')
        index = self.spectral_axis_combo.findData(desired_mode)
        if index != -1 and index != self.spectral_axis_combo.currentIndex():
            self.spectral_axis_combo.blockSignals(True)
            self.spectral_axis_combo.setCurrentIndex(index)
            self.spectral_axis_combo.blockSignals(False)

    def _on_spectral_axis_mode_changed(self, _index: int):
        """Handle spectral axis selection changes."""
        if not self.spectral_axis_combo.isEnabled():
            return

        mode = self.spectral_axis_combo.currentData()
        if mode not in ('velocity', 'frequency'):
            return

        if mode == getattr(self, 'spectral_axis_mode', None):
            return

        self.spectral_axis_mode = mode
        mode_label = "Velocity (VRAD)" if mode == 'velocity' else "Frequency (FREQ)"
        self._update_history(f"Spectral axis preference set to {mode_label}.", category='spectral-axis')
        self.header_modified = True
        self.save_button.setEnabled(True)

    def _set_option_enabled(self, userData, enabled):
        """Finds item by userData and enables/disables it."""
        for i in range(self.target_unit_combo.count()):
            if self.target_unit_combo.itemData(i) == userData:
                item = self.target_unit_combo.model().item(i)
                if item:
                    item.setEnabled(enabled)
                return

    def _toggle_method_combo(self, text):
        """Enable/Disable the method combo box based on selection."""
        # First, check if the parent group is enabled at all
        if not self.target_unit_combo.isEnabled():
            self.method_label.setEnabled(False)
            self.method_combo.setEnabled(False)
            return

        target_unit = self.target_unit_combo.currentData()
        is_k_conversion = (target_unit == 'k' or self.current_bunit == 'k')
        show_method = is_k_conversion and self.can_convert_k

        self.method_label.setEnabled(show_method)
        self.method_combo.setEnabled(show_method)
        
        if show_method:
            is_3d_plus = self.fits_viewer.header.get('NAXIS', 2) >= 3
            planck_item = self.method_combo.model().item(1)
            if planck_item:
                planck_item.setEnabled(is_3d_plus)
            self.method_combo.setToolTip("" if is_3d_plus else "Planck conversion requires a 3D or 4D cube.")
            if not is_3d_plus:
                self.method_combo.setCurrentIndex(0)
        else:
            self.method_combo.setToolTip("")

    # ------------------------------------------------------------------
    # Main Scaling and Header Update Logic
    # ------------------------------------------------------------------

    def _ensure_original_data(self):
        """Check if original_data is cached, if not, cache it."""
        if self.original_data is None:
            self.original_data = create_preview_snapshot(
                self.fits_viewer.data,
                operation_name="Unit conversion",
            )
            header = getattr(self.fits_viewer, 'header', None)
            self._conversion_baseline_has_bunit = bool(header is not None and 'BUNIT' in header)
            self._conversion_baseline_bunit = header.get('BUNIT') if header is not None else None


    def update_header_values(self):
        """
        Validate and apply new header values from the UI.
        For each changed field, lock it and change its style upon success.
        """
        updated_keys = []
        error_messages = []
        ignore_values = ['n/a', 'nan', 'none', 'non', '']
        detail_history_messages = []

        def process_field(field, key, conversion_factor, is_degree=False):
            # Skip if the field is already read-only (already set)
            if field.isReadOnly():
                return

            text = field.text().strip()
            original_text = self.header_info_strings.get(key.lower() if not is_degree else 'bpa', 'N/A')
            
            if text.lower() in ignore_values:
                if original_text.lower() not in ignore_values:
                     # User deleted a valid value, treat as reset for this key
                    if key in self.original_header:
                        self.fits_viewer.header[key] = self.original_header[key]
                    elif key in self.fits_viewer.header:
                        del self.fits_viewer.header[key]
                    updated_keys.append(key)
                return # No change or reset to original, don't lock

            if text == original_text:
                return # No change

            try:
                value = float(text)
                # Special handling for RESTFRQ to log original and adjust velocity axis headers
                if key == 'RESTFRQ':
                    old_has = 'RESTFRQ' in self.fits_viewer.header
                    old_val_hz = self.fits_viewer.header.get('RESTFRQ', None)
                    new_val_hz = value * conversion_factor # This is a float

                    # --- BUG FIX (from previous turn) ---
                    # Attempt to convert old_val_hz to a float for comparison
                    old_val_hz_float = None
                    if old_val_hz is not None:
                        try:
                            old_val_hz_float = float(old_val_hz)
                        except (TypeError, ValueError):
                            pass # old_val_hz_float remains None

                    # Check if the value is *actually* changing
                    is_changed = (old_val_hz_float is None) or (not np.isclose(old_val_hz_float, new_val_hz))
                    # --- BUG FIX (from previous turn) END ---

                    self.fits_viewer.header['RESTFRQ'] = new_val_hz
                    if isinstance(self.spectral_metadata, dict):
                        try:
                            old_val_hz_numeric = float(old_val_hz)
                        except (TypeError, ValueError):
                            old_val_hz_numeric = None
                        self.spectral_metadata['restfreq_hz'] = new_val_hz
                        if self.spectral_metadata.get('restfreq_original_hz') is None:
                            original_rest = old_val_hz_numeric if old_has and old_val_hz_numeric is not None else new_val_hz
                            self.spectral_metadata['restfreq_original_hz'] = original_rest
                    
                    # HISTORY: record precise change
                    if old_has:
                        try:
                            old_val_hz_fmt = f"{float(old_val_hz):.6f}"
                        except (TypeError, ValueError):
                            old_val_hz_fmt = str(old_val_hz)
                        
                        # *** APPLY BUG FIX CHECK HERE ***
                        if is_changed:
                            detail_history_messages.append((
                                'restfreq-change',
                                f"RESTFRQ changed: {old_val_hz_fmt} Hz -> {new_val_hz:.6f} Hz"
                            ))
                    else:
                        # If it didn't exist before, it's definitely a change
                        detail_history_messages.append((
                            'restfreq-change',
                            f"RESTFRQ set to {new_val_hz:.6f} Hz (previously undefined)"
                        ))
                        is_changed = True # Ensure 'is_changed' is true if key was new

                    # If spectral axis is velocity-like, keep type and ensure units
                    naxis = self.fits_viewer.header.get('NAXIS', 0)
                    for i in range(1, naxis + 1):
                        ctype_key = f'CTYPE{i}'
                        ctype_i = self.fits_viewer.header.get(ctype_key, '')
                        ctype_upper = ctype_i.upper()
                    
                        current_spectral_axis = self._spectral_axis_index
                        if current_spectral_axis is None:
                            break
                        if i != current_spectral_axis:
                            continue
                        
                        if any(tag in ctype_upper for tag in ('VELO', 'VRAD', 'VOPT')):
                            normalized_ctype = ctype_i if ctype_i else 'VRAD'
                            self.fits_viewer.header[ctype_key] = normalized_ctype
                            cunit_key = f'CUNIT{i}'
                            cunit_val = str(self.fits_viewer.header.get(cunit_key, '')).strip()
                            if not cunit_val:
                                self.fits_viewer.header[cunit_key] = 'km/s'
                                cunit_val = 'km/s'

                            # Only run CRVAL update logic if the frequency *actually* changed
                            if old_has and old_val_hz and is_changed and old_val_hz_float is not None:
                                try:
                                    velocity_value = float(self.fits_viewer.header.get(f'CRVAL{i}', 0.0))
                                except (TypeError, ValueError):
                                    velocity_value = None

                                if velocity_value is not None:
                                    c_speed = const.c.to('m/s').value
                                    vel_unit = cunit_val.lower()
                                    vel_mps = velocity_value * 1000.0 if 'km/s' in vel_unit else velocity_value
                                    if np.isfinite(vel_mps):
                                        freq_obs = None
                                        if 'VOPT' in ctype_upper:
                                            denom = 1.0 + vel_mps / c_speed
                                            if np.isfinite(denom) and denom > 0:
                                                freq_obs = old_val_hz_float / denom
                                        else:
                                            freq_obs = old_val_hz_float * (1.0 - vel_mps / c_speed)

                                        if freq_obs and np.isfinite(freq_obs) and freq_obs > 0:
                                            if 'VOPT' in ctype_upper:
                                                new_vel_mps = c_speed * ((new_val_hz / freq_obs) - 1.0)
                                            else:
                                                # Use new_val_hz (which is guaranteed > 0 if freq_obs > 0)
                                                new_vel_mps = c_speed * (new_val_hz - freq_obs) / new_val_hz

                                            if np.isfinite(new_vel_mps):
                                                new_vel = new_vel_mps / 1000.0 if 'km/s' in vel_unit else new_vel_mps
                                                self.fits_viewer.header[f'CRVAL{i}'] = new_vel
                                                crval_key = f'CRVAL{i}'
                                                if crval_key not in updated_keys:
                                                    updated_keys.append(crval_key)
                            break
                    
                    # Only add RESTFRQ to updated_keys and lock the field IF the value actually changed.
                    if is_changed:
                        if key not in updated_keys:
                            updated_keys.append(key)
                        self._set_field_set_style(field, True) # Lock and style the field

                else:
                    self.fits_viewer.header[key] = value * conversion_factor
                    
                    # Moved these lines inside the 'else' block
                    if key not in updated_keys:
                        updated_keys.append(key)
                    self._set_field_set_style(field, True) # Lock and style the field

            except ValueError:
                error_messages.append(f"{key} is not a valid number.")
            except Exception as e:
                error_messages.append(f"Error setting {key}: {e}")

        process_field(self.rf_input, 'RESTFRQ', 1e9)
        process_field(self.bmaj_input, 'BMAJ', 1/3600.0)
        process_field(self.bmin_input, 'BMIN', 1/3600.0)
        process_field(self.bpa_input, 'BPA', 1.0, is_degree=True)

        # --- Finalize ---
        if error_messages:
            QMessageBox.warning(self, 'Input Error', '\n'.join(error_messages))
            # Revert fields that failed validation
            self.reset_header_values(show_message=False)
            return

        if not updated_keys:
            QMessageBox.information(self, 'No Changes', 'No valid header values were changed.')
        else:
            self.header_modified = True
            self.save_button.setEnabled(True)
            
            for key in updated_keys:
                for window in self.subwindows:
                    if window:
                        window.header[key] = self.fits_viewer.header[key]
            
            history_msg = f"Header updated: {', '.join(updated_keys)} modified by user."
            self._update_history(history_msg, category='header-update')
            for category, message in detail_history_messages:
                self._update_history(message, category=category, include_general=False)

            # Recalculate parameters and update UI state
            self._reinitialize_parameters()
            # Immediately propagate changes to viewers
            self.update_all_displays()
            self._update_spectral_metadata_from_header()
            
            QMessageBox.information(self, 'Success', f"FITS header has been updated for: {', '.join(updated_keys)}.")


    def _reinitialize_parameters(self):
        """Recalculate all internal parameters after a header change."""
        self.header_params = self._check_header_keys()
        # Do not update text fields here, only the underlying capabilities
        self.beam_area_sr = self._calculate_beam_area_sr()
        self.pixel_area_sr = self._calculate_pixel_area_sr()
        self._populate_conversion_options()

    def apply_unit_conversion(self):
        """Apply the selected unit conversion."""
        self._ensure_original_data()

        try:
            target_unit = self.target_unit_combo.currentData()
            if target_unit is None:
                QMessageBox.warning(self, 'Input Error', 'Please select a valid target unit.')
                return

            method_str = 'Rayleigh-Jeans' if 'Rayleigh-Jeans' in self.method_combo.currentText() else 'Planck'
            
            # Map method name to usecase expectation ("rayleigh-jeans", "planck")
            uc_method = method_str.lower()
            from_unit = self._get_normalized_bunit(self.fits_viewer.header)
            self.current_bunit = from_unit
            
            scaled_data, new_bunit_header = convert_intensity_unit(
                data=self.fits_viewer.data,
                header=self.fits_viewer.header,
                from_unit=from_unit,
                to_unit=target_unit,
                method=uc_method
            )
            
            # Record History (Simplified compared to old implementation)
            history_msg = f"Unit conversion: {self.current_bunit.upper()} -> {new_bunit_header} (Method:{method_str})"
            
            # We can't easily get the min/max factor without re-doing the math, so we skip detail logging of the factor.
            # This is acceptable for Tier 2 migration.
            
            self._apply_data_and_history(scaled_data, new_bunit_header, history_msg, history_category='conversion')
            self._record_conversion_preview_action(from_unit, target_unit, uc_method)

            self.current_bunit = new_bunit_header.lower()
            self.current_unit_label.setText(new_bunit_header)
            self.conversion_group.setToolTip("Reset panel to perform another conversion.")
            for widget in [self.target_unit_combo, self.apply_conversion_button, self.method_label, self.method_combo]:
                widget.setEnabled(False)
            self.reset_scaling_button.setEnabled(True)
            self.save_button.setEnabled(True)
            # Reset manual scaling baseline after conversions
            self.scaling_reference_data = None

        except Exception as e:
            QMessageBox.critical(self, 'Conversion Error', f'An error occurred: {e}')
            self.reset_conversion_operations()



    def _apply_data_and_history(self, scaled_data, new_bunit_header, history_msg, history_category=None):
        """Helper function to apply data, header, history, and update displays."""
        self._update_history(history_msg, category=history_category)

        self.fits_viewer.header['BUNIT'] = new_bunit_header
        self.fits_viewer.data = scaled_data
        self.fits_viewer.update_cube()
        
        for window in self.subwindows:
            if window:
                window.header['BUNIT'] = new_bunit_header
                window.data = scaled_data
                window.update_cube()
        self._sync_bunit_to_viewers()
        self.update_all_displays()

    def reset_header_values(self, show_message=True):
        """Resets header values to their original state, making fields editable again."""
        if self.header_modified and show_message:
            QMessageBox.information(self, 'Header Reset', 'Header values have been reset to their original state.')

        self.fits_viewer.header = self.original_header.copy()
        for window in self.subwindows:
            if window:
                window.header = self.original_header.copy()
        
        # Unlock all fields and reset their styles
        for field in [self.rf_input, self.bmaj_input, self.bmin_input, self.bpa_input]:
            self._set_field_set_style(field, False)

        # Repopulate fields with original values and re-check capabilities
        self._update_header_fields()
        self._reinitialize_parameters()
        if getattr(self.fits_viewer.data, 'ndim', 0) > 2:
            self.spectral_axis_mode = 'frequency' if self._disk_axis_was_frequency else 'velocity'
        self._sync_spectral_axis_ui()
        for category in ('spectral-axis', 'restfreq-change', 'header-update'):
            self._clear_history_category(category)
        self._history_dedup_keys.clear()
        
        self.header_modified = False
        
        # Disable save button ONLY if no data scaling has been applied
        if self.original_data is None:
            self.save_button.setEnabled(False)

        # Ensure viewers redraw with restored metadata
        self._sync_bunit_to_viewers()
        self.update_all_displays()
        self._update_spectral_metadata_from_header()

    def reset_conversion_operations(self):
        """Resets only the unit conversion UI to its original state."""
        preferred_cursor = capture_preferred_cursor_snapshot(self.fits_viewer)
        removed_preview = self._clear_conversion_preview_action()
        restored_from_history = False
        can_restore_from_history = removed_preview and (not self.header_modified)
        if can_restore_from_history:
            restored_from_history = replay_action_history_to_current_cursor(
                self.fits_viewer,
                preferred_cursor=preferred_cursor,
            )

        # Restore data if it was changed
        if not restored_from_history and self.original_data is not None:
            self.fits_viewer.data = self.original_data
            self.fits_viewer.update_cube()
            if self._conversion_baseline_has_bunit:
                self.fits_viewer.header['BUNIT'] = self._conversion_baseline_bunit
            else:
                self.fits_viewer.header.pop('BUNIT', None)
            for window in self.subwindows:
                if window:
                    window.data = self.original_data
                    if self._conversion_baseline_has_bunit:
                        window.header['BUNIT'] = self._conversion_baseline_bunit
                    else:
                        window.header.pop('BUNIT', None)
                    window.update_cube()
        self.original_data = None  # Nullify cache
        self._conversion_baseline_bunit = None
        self._conversion_baseline_has_bunit = False

        # Restore internal state and UI for conversion
        self._sync_bunit_to_viewers()
        self.current_bunit = self._get_normalized_bunit(self.fits_viewer.header)
        self.original_bunit = self.current_bunit
        self.current_unit_label.setText(self.current_bunit.upper().replace('JY','Jy').replace('PIX','pixel'))
        
        # Only disable save if header hasn't also been modified
        if not self.header_modified:
            self.save_button.setEnabled(False)

        # Re-initialize conversion UI components
        self.conversion_group.setToolTip("")
        for widget in [self.target_unit_combo, self.apply_conversion_button, self.method_label, self.method_combo]:
            widget.setEnabled(True)
        self.reset_scaling_button.setEnabled(False)
        self._populate_conversion_options() # This re-runs the enable/disable logic

        # Update displays
        self.update_all_displays()
        self._clear_history_category('conversion')
        self._update_spectral_metadata_from_header()

    def _record_conversion_preview_action(self, from_unit: str, to_unit: str, method: str) -> None:
        record_action_preview(
            self.fits_viewer,
            "convert_intensity_unit",
            {
                "from_unit": str(from_unit),
                "to_unit": str(to_unit),
                "method": str(method),
            },
            replace_tag=self._conversion_action_record_tag,
        )

    def _clear_conversion_preview_action(self) -> bool:
        return bool(
            clear_action_preview_record(
                self.fits_viewer,
                self._conversion_action_record_tag,
                action_name="convert_intensity_unit",
            )
        )

    def update_all_displays(self):
        """Force-redraw all viewer and subwindow canvases with updated data."""
        self._sync_bunit_to_viewers()
        all_windows = [self.fits_viewer] + self.subwindows
        for window in all_windows:
            if not window:
                continue

            window.update_cube()

            data_slice = None
            data = getattr(window, 'data', None)
            current_channel = 0
            if hasattr(window, 'current_channel_index'):
                try:
                    current_channel = window.current_channel_index()
                except Exception:
                    current_channel = 0
            plane = getattr(window, 'plane', 'xy')
            if data is None:
                continue

            try:
                if data.ndim == 4:
                    if plane == 'xy':
                        data_slice = data[0, current_channel, :, :]
                    elif plane == 'xz':
                        data_slice = data[0, :, current_channel, :]
                    elif plane == 'zy':
                        data_slice = data[0, :, :, current_channel].T
                elif data.ndim == 3:
                    if plane == 'xy':
                        data_slice = data[current_channel, :, :]
                    elif plane == 'xz':
                        data_slice = data[:, current_channel, :]
                    elif plane == 'zy':
                        data_slice = data[:, :, current_channel].T
                elif data.ndim == 2:
                    data_slice = data
            except Exception:
                data_slice = None

            if data_slice is not None and hasattr(window, 'im'):
                window.im.set_data(data_slice)
            if hasattr(window, 'canvas'):
                window.canvas.draw()

        main_window = getattr(self.fits_viewer, 'main_window', None) or self.fits_viewer
        refresh_wcs_strings = getattr(main_window, '_refresh_wcs_display_strings', None)
        if callable(refresh_wcs_strings):
            try:
                refresh_wcs_strings()
            except Exception:
                pass
                
    # ------------------------------------------------------------------
    # FITS Save and HISTORY
    # ------------------------------------------------------------------
  
    def _prepare_header_for_save(self, *, is_data_being_flipped_back: bool = False):
        """Return a header copy adjusted for FITS output."""
        header_copy = self.fits_viewer.header.copy()

        convert_to_frequency = (
            getattr(self.fits_viewer.data, 'ndim', 0) > 2
            and getattr(self, 'spectral_axis_mode', 'velocity') == 'frequency'
        )
        
        axis = self._spectral_axis_index or self._disk_spectral_axis_index
        if axis is None:
             axis, _, _ = self._find_velocity_axis_in_header(header_copy)

        if convert_to_frequency:
            
            desired_unit = None
            desired_ctype = None

            if axis is not None:
                restfreq = header_copy.get('RESTFRQ', header_copy.get('RESTFREQ'))
                try:
                    restfreq = float(restfreq)
                except (TypeError, ValueError):
                    restfreq = None

                crval_key = f'CRVAL{axis}'
                cdelt_key = f'CDELT{axis}'
                cunit_key = f'CUNIT{axis}'
                ctype_key = f'CTYPE{axis}'

                # Use the *original* axis type (from disk) to determine the scenario
                original_was_frequency = self._disk_axis_was_frequency
                
                # (Scenario 1: Freq -> Freq)
                if original_was_frequency:
                    
                    # (Freq -> Freq) means we restore the *original* spectral axis
                    # from the disk header, ignoring the current (Velo) axis.
                    
                    original_disk_header = self._load_disk_header()
                    if original_disk_header:
                         # Restore the *spectral axis* keys from disk.
                         for key_prefix in ('CTYPE', 'CUNIT', 'CRVAL', 'CRPIX', 'CDELT'):
                             key = f"{key_prefix}{axis}"
                             if key in original_disk_header:
                                 header_copy[key] = original_disk_header[key]
                             elif key in header_copy:
                                 # If key existed in viewer but not disk, remove it
                                 if key_prefix not in ('CRPIX'): # Keep CRPIX if needed?
                                     del header_copy[key]
                    
                    # Ensure CTYPE/CUNIT are valid if disk_header failed
                    current_ctype = header_copy.get(ctype_key, '')
                    if not current_ctype or 'VELO' in current_ctype.upper():
                        header_copy[ctype_key] = self._disk_spectral_ctype or 'FREQ'
                        
                    current_cunit = header_copy.get(cunit_key, '')
                    if not current_cunit or 'm/s' in current_cunit.lower():
                        header_copy[cunit_key] = self._disk_spectral_cunit or 'Hz'
                    
                    # Ensure the *new* RESTFRQ (from the UI) is in the header
                    if restfreq is not None:
                        header_copy['RESTFRQ'] = restfreq
                    
                    return header_copy
                
                # (Scenario 3: Velo -> Freq)
                # If we reach here, it means:
                # convert_to_frequency=True
                # AND original_was_frequency=False
                
                # Load the *target* format.
                if self.spectral_metadata.get('original_axis_type') == 'frequency':
                    desired_unit = self.spectral_metadata.get('original_axis_unit')
                    desired_ctype = self.spectral_metadata.get('original_axis_ctype')

                if desired_unit is None and self._disk_spectral_cunit:
                    desired_unit = self._disk_spectral_cunit
                if desired_ctype is None and self._disk_spectral_ctype and 'FREQ' in self._disk_spectral_ctype.upper():
                    desired_ctype = self._disk_spectral_ctype

                # Ensure desired_unit is a valid frequency unit
                desired_unit = (desired_unit or '').strip()
                try:
                    unit_obj = u.Unit(desired_unit) if desired_unit else None
                    if unit_obj is None or not unit_obj.is_equivalent(u.Hz):
                        desired_unit = 'Hz'
                except Exception:
                    desired_unit = 'Hz'

                desired_ctype = desired_ctype or 'FREQ'


                if restfreq and np.isfinite(restfreq):
                    vel_crval = header_copy.get(crval_key, 0.0)
                    vel_cdelt = header_copy.get(cdelt_key, 0.0)
                    vel_cunit = str(header_copy.get(cunit_key, '') or '').strip().lower()
                    if not vel_cunit:
                        vel_cunit = str(self.spectral_metadata.get('current_axis_unit', '') or '').strip().lower()

                    try:
                        vel_crval = float(vel_crval)
                    except (TypeError, ValueError):
                        vel_crval = None

                    try:
                        vel_cdelt = float(vel_cdelt)
                    except (TypeError, ValueError):
                        vel_cdelt = None

                    c_mps = const.c.to('m/s').value
                    vel_unit_factor = 1000.0 if vel_cunit == 'km/s' else 1.0

                    freq_crval = None
                    freq_cdelt = None
                    
                    current_ctype_upper = str(header_copy.get(ctype_key, '')).upper()

                    if vel_crval is not None:
                        vel_crval_mps = vel_crval * vel_unit_factor
                        if 'VOPT' in current_ctype_upper:
                            denom = 1.0 + vel_crval_mps / c_mps
                            if np.isfinite(denom) and denom != 0.0:
                                freq_crval = restfreq / denom
                        else:
                            freq_crval = restfreq * (1.0 - vel_crval_mps / c_mps)

                    if vel_cdelt is not None:
                        vel_cdelt_mps = vel_cdelt * vel_unit_factor
                        freq_cdelt = - (vel_cdelt_mps * restfreq) / c_mps

                    desired_unit = desired_unit or 'Hz'
                    try:
                        target_unit = u.Unit(desired_unit)
                        if not target_unit.is_equivalent(u.Hz):
                            target_unit = u.Hz
                            desired_unit = 'Hz'
                    except Exception:
                        target_unit = u.Hz
                        desired_unit = 'Hz'

                    if freq_crval is not None:
                        freq_crval = (freq_crval * u.Hz).to(target_unit).value
                        header_copy[crval_key] = freq_crval
                    if freq_cdelt is not None:
                        freq_cdelt = (freq_cdelt * u.Hz).to(target_unit).value
                        header_copy[cdelt_key] = freq_cdelt

                    if desired_unit:
                        header_copy[cunit_key] = desired_unit

                    header_copy[ctype_key] = desired_ctype or 'FREQ'
        
        elif axis is not None: # Target is Velocity (convert_to_frequency is False)
            
            # Scenarios (Velo -> Velo) and (Freq -> Velo) are handled here.
            # In both cases, the loader and UI have already ensured the in-memory
            # header (which we have in header_copy) correctly describes the data
            # with a velocity axis, including any rest frequency changes.
            # We just need to ensure the CTYPE and CUNIT are present and valid.

            ctype_key = f'CTYPE{axis}'
            cunit_key = f'CUNIT{axis}'

            desired_ctype = 'VRAD' 
            desired_unit = 'km/s'
            
            # Try to get more specific types from metadata if available
            if self.spectral_metadata.get('current_axis_type') == 'velocity':
                desired_ctype = self.spectral_metadata.get('current_axis_ctype') or desired_ctype
                desired_unit = self.spectral_metadata.get('current_axis_unit') or desired_unit
            elif self.spectral_metadata.get('original_axis_type') == 'velocity':
                # Fallback to original if current is not available
                desired_ctype = self.spectral_metadata.get('original_axis_ctype') or desired_ctype
                desired_unit = self.spectral_metadata.get('original_axis_unit') or desired_unit

            header_copy[ctype_key] = header_copy.get(ctype_key) or desired_ctype
            header_copy[cunit_key] = header_copy.get(cunit_key) or desired_unit
            
            # The complex re-calculation logic that was here for the Freq->Velo
            # case was buggy. It incorrectly mixed original header values with the new
            # rest frequency, ignoring the already-correct in-memory header state.
            # By simply trusting the in-memory header, we fix the bug.

        return header_copy

    def save_fits(self):
        """Save the currently displayed (scaled) data to a new FITS file."""
        if self.original_data is None and not self.header_modified:
            QMessageBox.information(self, 'No Changes', 'No scaling or header changes have been applied to save.')
            return

        # Determine suffix based on changes
        suffix_parts = []
        
        # 1. Check for intensity unit conversion
        bunit_changed = self.original_data is not None
        if bunit_changed:
            if self.current_bunit == 'k':
                suffix_parts.append("Tb")
            elif self.current_bunit == 'jy/beam':
                suffix_parts.append("Jybeam")
            elif self.current_bunit == 'jy/pix':
                suffix_parts.append("Jypix")

        # 2. Check for spectral axis conversion
        axis_mode_changed = (
            getattr(self.fits_viewer.data, 'ndim', 0) > 2 and
            self.header_modified and 
            self.spectral_axis_combo.isEnabled()
        )
        if axis_mode_changed:
            current_mode = self.spectral_axis_mode
            original_mode_is_freq = self._disk_axis_was_frequency
            
            if current_mode == 'frequency' and not original_mode_is_freq:
                suffix_parts.append("freq")
            elif current_mode == 'velocity' and original_mode_is_freq:
                suffix_parts.append("vel")

        # 3. Construct final suffix
        if suffix_parts:
            suffix = "_".join(suffix_parts)
        else: # This branch is taken if header was modified but no specific suffix applies
            suffix = "hdr_edit"

        # Saving does not mutate the displayed array. Keep the existing array
        # (or lazy memmap wrapper) instead of making a second full-cube copy.
        data_to_save = self.fits_viewer.data

        # Check if the data should be reverted to its original order
        was_flipped_on_load = self.spectral_metadata.get('axis_flipped', False)
        was_originally_freq = self._disk_axis_was_frequency
        is_frequency_mode = (getattr(self, 'spectral_axis_mode', 'velocity') == 'frequency')
        revert_flip = was_flipped_on_load and was_originally_freq and is_frequency_mode

        if revert_flip:
            # If the data was flipped on load (for internal consistency) and originated
            # from a frequency axis, revert it to its original order for saving.
            if self._spectral_axis_index is not None and data_to_save.ndim >= 3:
                spectral_axis_dim = data_to_save.ndim - self._spectral_axis_index
                
                if 0 <= spectral_axis_dim < data_to_save.ndim:
                    if is_lazy_scaled(data_to_save):
                        data_to_save = data_to_save._raw_view_op(
                            np.flip,
                            axis=spectral_axis_dim,
                        )
                    else:
                        data_to_save = np.flip(
                            data_to_save,
                            axis=spectral_axis_dim,
                        )
                    print("\033[96mData array reverted to original FITS file order for saving.\033[0m")

        new_header = self._prepare_header_for_save(is_data_being_flipped_back=revert_flip)
        update_datamin_datamax_if_present(
            new_header,
            data_to_save,
            ensure=True,
        )
        # Preserve this panel's historical all-invalid convention.
        if not (
            np.isfinite(new_header.get('DATAMIN', np.nan))
            and np.isfinite(new_header.get('DATAMAX', np.nan))
        ):
            new_header['DATAMIN'] = 0.0
            new_header['DATAMAX'] = 0.0

        stale_history_lines = []
        for lines in self._history_categories.values():
            stale_history_lines.extend(str(line) for line in lines if line)
        self._remove_history_lines(new_header, stale_history_lines)

        for entry in build_processing_history_lines(self.fits_viewer):
            new_header.add_history(entry)

        original_header_for_save = self._disk_header or self.fits_viewer.header
        save_fits = SaveFITS(data_to_save, new_header, self.fits_viewer.filename, original_header=original_header_for_save)
        save_fits.save(suffix=suffix)

    def _remove_history_lines(self, header, lines):
        """Remove specific HISTORY lines from a header."""
        if not header or not lines:
            return
        try:
            existing_entries = list(header.get('HISTORY', []))
        except KeyError:
            existing_entries = []

        if not existing_entries:
            return

        targets = {line for line in lines if line}
        if not targets:
            return

        filtered_entries = [entry for entry in existing_entries if entry not in targets]

        # Remove all HISTORY cards and re-add filtered ones to preserve order
        if 'HISTORY' in header:
            header.remove('HISTORY', remove_all=True, ignore_missing=True)

        for entry in filtered_entries:
            header.add_history(entry)

    def _clear_history_category(self, category: str):
        """Remove the stored HISTORY entries for a given category."""
        previous_lines = self._history_categories.pop(category, None)
        if not previous_lines:
            return
        self._remove_history_lines(self.fits_viewer.header, previous_lines)
        for window in self.subwindows:
            if window and window.header:
                self._remove_history_lines(window.header, previous_lines)
        self._history_dedup_keys = {key for key in self._history_dedup_keys if key[0] != category}

    @staticmethod
    def _split_lines(text, limit=70):
        """Split text into chunks that fit into FITS HISTORY cards."""
        if not text:
            return []
        return [text[i:i+limit] for i in range(0, len(text), limit)]

    def _update_history(self, operation_details: str, category: Optional[str] = None, include_general: bool = True):
        """Adds a formatted HISTORY entry to the FITS header.

        When a category is provided, only the latest entry per category is kept.
        An entry with the same category and details will not be added if it already exists.
        """
        if not (self.fits_viewer and self.fits_viewer.header):
            return

        dedup_key = (category, operation_details)
        if dedup_key in self._history_dedup_keys:
            return

        if category:
            self._clear_history_category(category)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare full strings
        raw_operation_entry = f"Operation details: {operation_details}"
        raw_general_entry = (
            f"Data scaled/header updated using takefits on {timestamp}"
            if include_general else None
        )

        # Split into safe chunks (list of strings)
        operation_chunks = self._split_lines(raw_operation_entry)
        general_chunks = self._split_lines(raw_general_entry) if raw_general_entry else []

        headers_to_update = []
        main_header = getattr(self.fits_viewer, 'header', None)
        if main_header:
            headers_to_update.append(main_header)
        for window in self.subwindows:
            header = getattr(window, 'header', None) if window else None
            if header:
                headers_to_update.append(header)

        seen_headers = set()
        for header in headers_to_update:
            header_id = id(header)
            if header_id in seen_headers:
                continue
            seen_headers.add(header_id)
            self._insert_history_entries(header, general_chunks, operation_chunks, category)

        self._history_dedup_keys.add(dedup_key)

        if category:
            entries_to_store = tuple(general_chunks + operation_chunks)
            if entries_to_store:
                self._history_categories[category] = entries_to_store

    def _insert_history_entries(self, header, general_chunks: list, operation_chunks: list, category):
        """Insert history entries with controlled ordering and de-duplication."""
        if not header:
            return

        try:
            entries = list(header.get('HISTORY', []))
        except KeyError:
            entries = []

        modified = False

        # Add General Entry
        if general_chunks:
            if general_chunks[0] not in entries:
                entries.extend(general_chunks)
                modified = True

        # Add Operation Entry
        if operation_chunks:
            if operation_chunks[0] not in entries:
                insert_index = self._determine_history_insert_index(entries, category)
                
                if insert_index >= len(entries):
                    entries.extend(operation_chunks)
                else:
                    for chunk in reversed(operation_chunks):
                        entries.insert(insert_index, chunk)
                modified = True

        if modified:
            if 'HISTORY' in header:
                header.remove('HISTORY', remove_all=True, ignore_missing=True)
            for entry in entries:
                header.add_history(entry)

    def _determine_history_insert_index(self, entries, category: Optional[str]) -> int:
        """Determine where to insert a new operation entry."""
        # Find index of the most recent general entry
        general_idx = self._find_history_index(entries, "Data scaled/header updated using takefits on ")

        if category == 'header-update':
            return general_idx + 1 if general_idx is not None else len(entries)

        if category == 'restfreq-change':
            # Look for the START of a "Header updated" entry. 
            header_idx = self._find_history_index(entries, "Operation details: Header updated:")
            if header_idx is not None:
                return header_idx + 1
            return general_idx + 1 if general_idx is not None else len(entries)

        if general_idx is not None:
            return general_idx + 1
        return len(entries)

    @staticmethod
    def _find_history_index(entries, prefix: str) -> Optional[int]:
        """Return the index of the last entry starting with the given prefix."""
        for idx in range(len(entries) - 1, -1, -1):
            if entries[idx].startswith(prefix):
                return idx
        return None

    # ------------------------------------------------------------------
    # Close Event
    # ------------------------------------------------------------------

    def _has_pending_close_changes(self) -> bool:
        """Return True if this panel has unapplied conversion/header changes."""
        current_bunit = self._get_normalized_bunit(self.fits_viewer.header)
        has_preview = bool(has_action_record_tag(self.fits_viewer, self._conversion_action_record_tag))
        return (
            self.original_data is not None
            or self.header_modified
            or current_bunit != self.original_bunit
            or has_preview
        )

    def _discard_pending_close_changes(self) -> None:
        """Discard panel changes and restore original data/header state."""
        if self.original_data is not None or has_action_record_tag(self.fits_viewer, self._conversion_action_record_tag):
            self.reset_conversion_operations()
        current_bunit = self._get_normalized_bunit(self.fits_viewer.header)
        if self.header_modified or current_bunit != self.original_bunit:
            self.reset_header_values(show_message=False)

    def resync_after_workspace_restore(self) -> None:
        if self._has_pending_close_changes():
            self.save_button.setEnabled(True)

    def closeEvent(self, event):
        """Handle window close event."""
        if self._has_pending_close_changes():
            choice = confirm_pending_close(
                self,
                "Close Unit Conversion Panel",
                "There are unapplied unit conversion/header changes.",
            )
            if choice == "cancel":
                event.ignore()
                return
            if choice == "discard":
                self._discard_pending_close_changes()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Conversion Calculation Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Conversion Calculation Helpers (Deprecated - Moved to Core/UseCases)
    # ------------------------------------------------------------------
    # Methods _get_freq_axis_hz, _reshape_freqs_for_broadcast, and unit conversions
    # have been migrated to core.usecases.convert_intensity_unit.
