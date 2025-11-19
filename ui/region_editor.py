import math
from copy import deepcopy
import astropy.units as u
from .moment_results_dialog import MomentResultsDialog
from astropy.wcs.utils import proj_plane_pixel_scales

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGridLayout,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QComboBox,
    QWidget,
)
from PyQt6.QtCore import Qt

from core.region import CircleRegion, RectangleRegion, EllipseRegion, CubeRegion
from core.common import Common



class RegionEditorDialog(QDialog):
    """Editable dialog for inspecting and tweaking region geometry."""

    def __init__(self, viewer, region, region_manager):
        super().__init__(viewer)
        self.viewer = viewer
        self.region = region
        self.region_manager = region_manager
        self._updating_fields = False
        self.offset_index = None
        self._field_units = {'width': 'pix', 'height': 'pix', 'radius': 'pix'}
        self._fields = {}
        self._axis_scales = {}
        self._pixel_limit = 1000.0
        self.moment_dialogs = []

        self.setWindowTitle(self._build_window_title())
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._build_ui()
        self._configure_spin_ranges()
        self.update_from_region()

    # ------------------------------------------------------------------
    # UI setup
    def _build_ui(self):
        layout = QVBoxLayout()

        self.shape_value_label = QLabel()
        layout.addWidget(self.shape_value_label)

        form = QFormLayout()
        form.setVerticalSpacing(4)  # Reduce spacing between form rows

        self.nameEdit = QLineEdit()
        self.nameEdit.setPlaceholderText("Region label")
        form.addRow("Label", self.nameEdit)
        
        self.center_x_spin = self._make_spinbox()
        self.center_y_spin = self._make_spinbox()
        self.centerWorldXEdit = QLineEdit()
        self.centerWorldYEdit = QLineEdit()
        self.centerWorldXEdit.setPlaceholderText("World X")
        self.centerWorldYEdit.setPlaceholderText("World Y")
        center_x_row = QHBoxLayout()
        center_x_row.addWidget(self.center_x_spin)
        center_x_row.addWidget(self.centerWorldXEdit)
        center_x_row.setContentsMargins(0, 0, 0, 0)
        center_x_container = QWidget()
        center_x_container.setLayout(center_x_row)
        center_y_row = QHBoxLayout()
        center_y_row.addWidget(self.center_y_spin)
        center_y_row.addWidget(self.centerWorldYEdit)
        center_y_row.setContentsMargins(0, 0, 0, 0)
        center_y_container = QWidget()
        center_y_container.setLayout(center_y_row)
        form.addRow("Center X (pix)", center_x_container)
        form.addRow("Center Y (pix)", center_y_container)

        self.radius_label = QLabel("Radius")
        self.radius_spin = self._make_spinbox()
        self.radius_unit_combo = self._make_unit_combo()
        self.radius_container = self._wrap_with_unit_combo(self.radius_spin, self.radius_unit_combo)
        form.addRow(self.radius_label, self.radius_container)


        self.size_container = QWidget()
        size_layout = QGridLayout(self.size_container)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setVerticalSpacing(2)

        self.width_label = QLabel("          Width")
        self.width_spin = self._make_spinbox()
        self.height_label = QLabel("          Height")
        self.height_spin = self._make_spinbox()
        self.size_unit_combo = self._make_unit_combo()

        size_layout.addWidget(self.width_label, 0, 0) 
        size_layout.addWidget(self.width_spin, 0, 1) 
        size_layout.addWidget(self.height_label, 1, 0) 
        size_layout.addWidget(self.height_spin, 1, 1)
        
        size_layout.addWidget(self.size_unit_combo, 0, 2, 2, 1)
        
        form.addRow(self.size_container)

        self.angle_label = QLabel("Angle (deg)")
        self.angle_spin = self._make_spinbox()
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setSingleStep(1.0)
        form.addRow(self.angle_label, self.angle_spin)

        self.z_min_label = QLabel("Z Min (pix)")
        self.z_min_spin = self._make_spinbox()
        self.z_min_spin.setDecimals(1)
        self.z_min_spin.setSingleStep(1.0)
        self.z_min_world_edit = QLineEdit()
        self.z_min_world_edit.setPlaceholderText("World Z")
        z_min_row = QHBoxLayout()
        z_min_row.setContentsMargins(0, 2, 0, 0) 
        z_min_row.addWidget(self.z_min_spin)
        z_min_row.addWidget(self.z_min_world_edit)
        z_min_container = QWidget()
        z_min_container.setLayout(z_min_row)
        form.addRow(self.z_min_label, z_min_container)


        self.z_max_label = QLabel("Z Max (pix)")
        self.z_max_spin = self._make_spinbox()
        self.z_max_spin.setDecimals(1)
        self.z_max_spin.setSingleStep(1.0)
        self.z_max_world_edit = QLineEdit()
        self.z_max_world_edit.setPlaceholderText("World Z")
        z_max_row = QHBoxLayout()
        z_max_row.setContentsMargins(0, 0, 0, 2) 
        z_max_row.addWidget(self.z_max_spin)
        z_max_row.addWidget(self.z_max_world_edit)
        z_max_container = QWidget()
        z_max_container.setLayout(z_max_row)
        form.addRow(self.z_max_label, z_max_container) 

        self._fields.update({
            'radius': {
                'axis': 'radius',
                'spin': self.radius_spin,
                'combo': self.radius_unit_combo,
                'label': self.radius_label,
                'container': self.radius_container,
            },
            'width': {
                'axis': 'x',
                'spin': self.width_spin,
                'combo': self.size_unit_combo,
                'label': self.width_label,
                'container': self.width_spin,
            },
            'height': {
                'axis': 'y',
                'spin': self.height_spin,
                'combo': self.size_unit_combo,
                'label': self.height_label,
                'container': self.height_spin,
            },
        })

        layout.addLayout(form)

        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)


        action_button_layout = QHBoxLayout()
        self.fill_view_button = QPushButton("Fill the View")
        action_button_layout.addWidget(self.fill_view_button)
        self.cutout_button = QPushButton("Cut Out", self)
        action_button_layout.addWidget(self.cutout_button)
        action_button_layout.addStretch(1)
        layout.addLayout(action_button_layout)

        self.moments_button = QPushButton("Calculate Moments")
        self.moments_label = QLabel()
        self.moments_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.moments_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.moments_label.setWordWrap(True)
        action_button_layout.addWidget(self.moments_button)
        action_button_layout.addWidget(self.moments_label)

        controls_layout = QHBoxLayout()
        controls_layout.addStretch(1)
        self.auto_apply_checkbox = QCheckBox("Apply changes automatically")
        self.auto_apply_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_apply_checkbox)
        layout.addLayout(controls_layout)
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setDefault(True)
        self.delete_button = QPushButton("Delete")
        self.close_button = QPushButton("Close")
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
        layout.addStretch(1)
        self.setLayout(layout)

        self.center_x_spin.valueChanged.connect(self._on_value_changed)
        self.center_y_spin.valueChanged.connect(self._on_value_changed)
        self.radius_spin.valueChanged.connect(self._on_value_changed)
        self.width_spin.valueChanged.connect(self._on_value_changed)
        self.height_spin.valueChanged.connect(self._on_value_changed)
        self.radius_unit_combo.currentTextChanged.connect(lambda unit: self._on_unit_changed('radius', unit))

        self.size_unit_combo.currentTextChanged.connect(lambda unit: self._on_shared_unit_changed('width', 'height', unit))
        self.angle_spin.valueChanged.connect(self._on_value_changed)
        self.auto_apply_checkbox.stateChanged.connect(self._on_auto_apply_changed)
        self.apply_button.clicked.connect(self.apply_changes)
        self.delete_button.clicked.connect(self._delete_region)
        self.close_button.clicked.connect(self.close)
        self.nameEdit.editingFinished.connect(self._on_label_changed)
        self.centerWorldXEdit.editingFinished.connect(self._on_world_edit_finished)
        self.centerWorldYEdit.editingFinished.connect(self._on_world_edit_finished)

        self.z_min_spin.valueChanged.connect(self._on_value_changed)
        self.z_max_spin.valueChanged.connect(self._on_value_changed)

        self.z_min_world_edit.editingFinished.connect(self._on_z_world_edit_finished)
        self.z_max_world_edit.editingFinished.connect(self._on_z_world_edit_finished)

        self.fill_view_button.clicked.connect(self._on_fill_the_view_clicked)
        self.cutout_button.clicked.connect(self._open_cutout_dialog)
        self.moments_button.clicked.connect(self._show_moment_results)

    def _make_spinbox(self):
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(0.5)
        return spin

    def _make_unit_combo(self):
        combo = QComboBox()
        combo.addItem('pix')
        combo.setEditable(False)
        return combo

    def _wrap_with_unit_combo(self, spin, combo):
        row = QHBoxLayout()
        row.addWidget(spin)
        row.addWidget(combo)
        row.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setLayout(row)
        return container

    def _configure_spin_ranges(self):
        array = getattr(self.viewer.im, 'get_array', lambda: None)()
        if array is None:
            width_limit = 1_000.0
            height_limit = 1_000.0
        else:
            height_limit, width_limit = array.shape[:2]

        self._pixel_limit = max(width_limit, height_limit) * 2.0
        for spin in (self.center_x_spin, self.center_y_spin):
            spin.setRange(-self._pixel_limit, self._pixel_limit)



    def _refresh_unit_controls(self):
        self._compute_axis_scales()
        region = self.region

        x_axis_type = self._axis_scales.get('x', {}).get('axis_type', '').upper()
        y_axis_type = self._axis_scales.get('y', {}).get('axis_type', '').upper()
        
        is_spatial_plane = ('RA' in x_axis_type or 'LON' in x_axis_type) and \
                           ('DEC' in y_axis_type or 'LAT' in y_axis_type)

        for name, meta in self._fields.items():
            combo = meta['combo']
            
            allowed_units = ['pix']
            # If the plane is spatial, allow angular unit conversions.
            if is_spatial_plane:
                allowed_units.extend(['deg', 'arcmin', 'arcsec'])
            
            # If the plane contains a velocity axis on either x or y,
            # you might want to add velocity units. This example sticks to the request.
            # x_kms = self._axis_scales.get('x', {}).get('kms_per_pix')
            # y_kms = self._axis_scales.get('y', {}).get('kms_per_pix')
            # if x_kms or y_kms:
            #     allowed_units.append('km/s')

            preferred = self._field_units.get(name, 'pix')
            combo.blockSignals(True)
            combo.clear()
            for unit in allowed_units:
                combo.addItem(unit)

            if preferred in allowed_units:
                combo.setCurrentText(preferred)
                selected = preferred
            else:
                combo.setCurrentText('pix')
                selected = 'pix'

            combo.setEnabled(len(allowed_units) > 1)
            combo.blockSignals(False)

            self._field_units[name] = selected
            self._configure_spin_for_unit(name, selected)

            pixel_value = self._get_region_pixel_value(region, name)
            if pixel_value is None:
                continue
            self._set_field_display_value(name, pixel_value)

    def _compute_axis_scales(self):
        converter = getattr(self.viewer, 'converter', None)
        wcs = getattr(converter, 'wcs', None)
        if wcs is None:
            wcs = getattr(self.viewer, 'wcs', None)
        self._axis_scales = {
            'x': {'deg_per_pix': None, 'kms_per_pix': None, 'unit': None, 'axis_type': None},
            'y': {'deg_per_pix': None, 'kms_per_pix': None, 'unit': None, 'axis_type': None},
            'radius': {'deg_per_pix': None, 'kms_per_pix': None},
        }

        if wcs is None:
            return

        slices = self._get_display_slices()
        if not slices:
            return

        x_pix = self.center_x_spin.value()
        y_pix = self.center_y_spin.value()

        pixel_base = self._build_pixel_vector(x_pix, y_pix, slices, wcs)
        try:
            world_base = wcs.wcs_pix2world([pixel_base], 0)[0]
        except Exception:
            return

        projected_scales = self._projected_deg_scales(wcs, slices, pixel_base, world_base)

        for axis_name in ('x', 'y'):
            if axis_name not in slices:
                continue
            axis_index = slices.index(axis_name)
            pixel_plus = pixel_base.copy()
            pixel_plus[axis_index] += 1.0
            try:
                world_plus = wcs.wcs_pix2world([pixel_plus], 0)[0]
            except Exception:
                continue

            delta = world_plus[axis_index] - world_base[axis_index]
            unit_obj = None
            if wcs.wcs.cunit is not None and axis_index < len(wcs.wcs.cunit):
                unit_obj = wcs.wcs.cunit[axis_index]
            unit_str = str(unit_obj) if unit_obj not in (None, '') else None

            axis_type = ''
            try:
                axis_type = wcs.axis_type_names[axis_index]
            except Exception:
                axis_type = ''

            axis_type_upper = axis_type.upper() if isinstance(axis_type, str) else ''
            unit_norm = unit_str.lower().replace(' ', '') if unit_str else ''
            deg_per_pix = projected_scales.get(axis_index)
            kms_per_pix = None

            if axis_type_upper.startswith('RA') or 'LON' in axis_type_upper:
                delta = ((delta + 180.0) % 360.0) - 180.0

            if deg_per_pix is None:
                if unit_norm in ('deg', 'degree', 'degrees'):
                    deg_per_pix = abs(delta)
                elif unit_norm == 'rad':
                    deg_per_pix = abs(delta) * (180.0 / math.pi)

            if 'km/s' in unit_norm:
                kms_per_pix = abs(delta)
            elif unit_norm == 'm/s':
                kms_per_pix = abs(delta) / 1000.0


            self._axis_scales[axis_name].update({
                'deg_per_pix': deg_per_pix,
                'kms_per_pix': kms_per_pix,
                'unit': unit_str,
                'axis_type': axis_type,
            })
            
        def _is_lat_axis(axis_type_str):
            if not axis_type_str:
                return False
            axis_upper = axis_type_str.upper()
            return 'DEC' in axis_upper or 'LAT' in axis_upper

        x_deg = self._axis_scales['x']['deg_per_pix']
        y_deg = self._axis_scales['y']['deg_per_pix']
        x_type = self._axis_scales['x']['axis_type']
        y_type = self._axis_scales['y']['axis_type']

        radius_deg_per_pix = None
        if _is_lat_axis(y_type) and y_deg:
            radius_deg_per_pix = y_deg
        elif _is_lat_axis(x_type) and x_deg:
            radius_deg_per_pix = x_deg
        elif y_deg and x_deg:
            radius_deg_per_pix = (x_deg + y_deg) / 2.0
        elif x_deg or y_deg:
            radius_deg_per_pix = x_deg if x_deg else y_deg

        if radius_deg_per_pix is not None:
            self._axis_scales['radius']['deg_per_pix'] = radius_deg_per_pix

        kms_scales = [value for value in (
            self._axis_scales['x']['kms_per_pix'],
            self._axis_scales['y']['kms_per_pix']
        ) if value]
        
        if len(kms_scales) == 2:
            self._axis_scales['radius']['kms_per_pix'] = sum(kms_scales) / len(kms_scales)
        elif len(kms_scales) == 1:
            self._axis_scales['radius']['kms_per_pix'] = kms_scales[0]

    def _projected_deg_scales(self, wcs, slices, pixel_base, world_base):
        try:
            wcs_copy = wcs.deepcopy() if hasattr(wcs, 'deepcopy') else deepcopy(wcs)
        except Exception:
            return {}

        try:
            for idx in range(min(len(pixel_base), wcs_copy.wcs.naxis)):
                wcs_copy.wcs.crpix[idx] = pixel_base[idx] + 1.0
                if idx < len(world_base):
                    wcs_copy.wcs.crval[idx] = world_base[idx]
            scales = proj_plane_pixel_scales(wcs_copy)
        except Exception:
            return {}

        scale_map = {}
        for idx, value in enumerate(scales):
            try:
                if isinstance(value, u.Quantity):
                    try:
                        deg_value = value.to(u.deg).value
                    except Exception:
                        deg_value = value.to(u.rad).value * (180.0 / math.pi)
                else:
                    deg_value = float(value)
                scale_map[idx] = abs(deg_value)
            except Exception:
                continue
        return scale_map

    def _get_display_slices(self):
        displaymap = getattr(self.viewer, 'displaymap', None)
        if displaymap is not None:
            slices = getattr(displaymap, 'slices', None)
            if slices:
                return slices

        for attr in ('integ_slice', 'projection_slices', 'slice'):  # integration windows, channel maps, etc.
            slices = getattr(self.viewer, attr, None)
            if slices and not callable(slices):
                return slices

        format_pix = getattr(self.viewer, 'format_pix', None)
        if format_pix is not None:
            return getattr(format_pix, 'slices', None)

        return None

    def _build_pixel_vector(self, x_pix, y_pix, slices, wcs):
        axis_types = []
        try:
            axis_types = [atype.upper() if atype else '' for atype in wcs.axis_type_names]
        except Exception:
            axis_types = []

        vector = []
        for idx, entry in enumerate(slices):
            if entry == 'x':
                vector.append(float(x_pix))
                continue
            if entry == 'y':
                vector.append(float(y_pix))
                continue

            axis_type = axis_types[idx] if idx < len(axis_types) else ''
            default_value = float(entry) if isinstance(entry, (int, float)) else 0.0

            axis_type_upper = axis_type.upper() if axis_type else ''
            if any(key in axis_type_upper for key in ('VELO', 'VRAD', 'VOPT', 'FREQ')):
                default_value = float(getattr(Common, 'zpix', 0.0))
            elif axis_type_upper.startswith('RA') or 'LON' in axis_type_upper:
                default_value = float(getattr(Common, 'xpix', 0.0))
            elif 'DEC' in axis_type_upper or 'LAT' in axis_type_upper:
                default_value = float(getattr(Common, 'ypix', 0.0))

            vector.append(default_value)

        while len(vector) < getattr(wcs, 'naxis', len(vector)):
            vector.append(0.0)

        return vector

    def _set_field_display_value(self, field_name, pixel_value):
        field = self._fields.get(field_name)
        if field is None:
            return
        unit = self._field_units.get(field_name, 'pix')
        converted = self._convert_from_pix(field_name, pixel_value, unit)
        spin = field['spin']
        spin.blockSignals(True)
        spin.setValue(converted)
        spin.blockSignals(False)

    def _configure_spin_for_unit(self, field_name, unit):
        spin = self._fields[field_name]['spin']
        max_val = self._pixel_limit
        axis = self._fields[field_name]['axis']
        scale = self._axis_scales.get(axis, {})
        deg_per_pix = scale.get('deg_per_pix')
        kms_per_pix = scale.get('kms_per_pix')

        if unit == 'deg' and deg_per_pix:
            max_val = self._pixel_limit * deg_per_pix
        elif unit == 'arcmin' and deg_per_pix:
            max_val = self._pixel_limit * deg_per_pix * 60.0
        elif unit == 'arcsec' and deg_per_pix:
            max_val = self._pixel_limit * deg_per_pix * 3600.0
        elif unit == 'km/s' and kms_per_pix:
            max_val = self._pixel_limit * kms_per_pix

        spin.setRange(-max_val, max_val)

        if unit == 'pix':
            spin.setDecimals(3)
            spin.setSingleStep(0.5)
        elif unit == 'deg':
            spin.setDecimals(6)
            spin.setSingleStep(0.0001)
        elif unit == 'arcmin':
            spin.setDecimals(4)
            spin.setSingleStep(0.001)
        elif unit == 'arcsec':
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
        elif unit == 'km/s':
            spin.setDecimals(3)
            spin.setSingleStep(0.1)

    def _convert_from_pix(self, field_name, pixel_value, unit):
        if unit == 'pix':
            return pixel_value
        axis = self._fields[field_name]['axis']
        scale = self._axis_scales.get(axis, {})
        deg_per_pix = scale.get('deg_per_pix')
        kms_per_pix = scale.get('kms_per_pix')

        if unit == 'deg' and deg_per_pix:
            return float(pixel_value) * deg_per_pix
        if unit == 'arcmin' and deg_per_pix:
            return (float(pixel_value) * deg_per_pix) * 60.0
        if unit == 'arcsec' and deg_per_pix:
            return (float(pixel_value) * deg_per_pix) * 3600.0
        if unit == 'km/s' and kms_per_pix:
            return pixel_value * kms_per_pix 
        return pixel_value

    def _convert_to_pix(self, field_name, value, unit):
        if unit == 'pix':
            return value
        axis = self._fields[field_name]['axis']
        scale = self._axis_scales.get(axis, {})
        deg_per_pix = scale.get('deg_per_pix')
        kms_per_pix = scale.get('kms_per_pix')

        if unit == 'deg' and deg_per_pix:
            return float(value) / deg_per_pix
        if unit == 'arcmin' and deg_per_pix:
            return (float(value) / 60.0) / deg_per_pix
        if unit == 'arcsec' and deg_per_pix:
            return (float(value) / 3600.0) / deg_per_pix
        if unit == 'km/s' and kms_per_pix:
            return value / kms_per_pix
        return value

    def _update_z_world_fields(self):
        """Updates the Z-axis world coordinate QLineEdits using the viewer's formatter."""
        if not hasattr(self.viewer, 'format_pix'):
            self.z_min_world_edit.setText("N/A")
            self.z_max_world_edit.setText("N/A")
            return

        try:
            ref_x = self.center_x_spin.value()
            ref_y = self.center_y_spin.value()
            
            z_min_pix = self.z_min_spin.value()
            world_val_min = self.viewer.format_pix.convert_chpix_to_world('xy', ref_x, ref_y, z_min_pix)
            world_str_min = self.viewer.format_pix.convert_chval_to_world_str('xy', world_val_min)
            self.z_min_world_edit.setText(world_str_min)

            z_max_pix = self.z_max_spin.value()
            world_val_max = self.viewer.format_pix.convert_chpix_to_world('xy', ref_x, ref_y, z_max_pix)
            world_str_max = self.viewer.format_pix.convert_chval_to_world_str('xy', world_val_max)
            self.z_max_world_edit.setText(world_str_max)

        except Exception:
            self.z_min_world_edit.setText("Conv. Error")
            self.z_max_world_edit.setText("Conv. Error")

    def _on_z_world_edit_finished(self):
        """Updates the Z-axis pixel spin boxes from the world coordinate QLineEdits."""
        converter = getattr(self.viewer, 'converter', None)
        if converter is None or self._updating_fields: return

        sender = self.sender()
        target_spin = self.z_min_spin if sender is self.z_min_world_edit else self.z_max_spin
        
        try:
            ref_world_x = Common.world_x
            ref_world_y = Common.world_y
            z_world_val = float(sender.text())
            
            if self.viewer.data.ndim == 3:
                pix_coords = converter.world_to_pix(ref_world_x, ref_world_y, z_world_val)
            else:
                pix_coords = converter.world_to_pix(ref_world_x, ref_world_y, z_world_val, 0)
            
            z_pix = pix_coords[2]

            self._updating_fields = True
            target_spin.setValue(z_pix)
            self._updating_fields = False
            
            if self.auto_apply_checkbox.isChecked():
                self.apply_changes()
            else:
                self.apply_button.setEnabled(True)

        except (ValueError, TypeError, IndexError):
            self._update_z_world_fields()

    def _value_in_pixels(self, field_name):
        self._compute_axis_scales()
        field = self._fields.get(field_name)
        if field is None:
            return 0.0
        unit = self._field_units.get(field_name, 'pix')
        value = field['spin'].value()
        return self._convert_to_pix(field_name, value, unit)

    def _get_region_pixel_value(self, region, field_name):
        if isinstance(region, CircleRegion):
            if field_name == 'radius':
                return region.radius
            return None
        if isinstance(region, (RectangleRegion, EllipseRegion)):
            if field_name == 'width':
                return getattr(region, 'width', None)
            if field_name == 'height':
                return getattr(region, 'height', None)
            return None
        return None

    def _on_unit_changed(self, field_name, new_unit):
        old_unit = self._field_units.get(field_name, 'pix')
        if old_unit == new_unit:
            return

        self._compute_axis_scales()
        field = self._fields.get(field_name)
        if field is None:
            self._field_units[field_name] = new_unit
            return

        spin = field['spin']
        spin.blockSignals(True)
        previous_value = spin.value()
        pixel_value = self._convert_to_pix(field_name, previous_value, old_unit)
        self._field_units[field_name] = new_unit
        self._configure_spin_for_unit(field_name, new_unit)
        converted = self._convert_from_pix(field_name, pixel_value, new_unit)

        spin.blockSignals(True)
        spin.setValue(converted)
        spin.blockSignals(False)


    def _on_shared_unit_changed(self, field1, field2, new_unit):
        """Handle unit changes for fields that share a unit combo box."""
        self._on_unit_changed(field1, new_unit)
        self._on_unit_changed(field2, new_unit)


    def update_from_region(self):
        if self.region_manager is None or self.region not in self.region_manager.regions:
            self.close()
            return

        self._updating_fields = True
        self.setWindowTitle(self._build_window_title())

        self._toggle_row(self.radius_label, self.radius_container, False)
        self._toggle_row(self.width_label, self.size_container, False)
        self._toggle_row(self.angle_label, self.angle_spin, False)
        self._toggle_row(self.z_min_label, self.z_min_spin.parentWidget(), False)
        self._toggle_row(self.z_max_label, self.z_max_spin.parentWidget(), False)

        if isinstance(self.region, CircleRegion):
            shape_name = "Circle"
            cx, cy = self.region.center
            self.center_x_spin.setValue(cx)
            self.center_y_spin.setValue(cy)
            
            self.radius_spin.setValue(self.region.radius)
            self._toggle_row(self.radius_label, self.radius_container, True)
            self.size_container.setVisible(False)

        elif isinstance(self.region, CubeRegion): # Check for Cube BEFORE Rectangle
            shape_name = "Cube"
            cx, cy = self.region.center
            self.center_x_spin.setValue(cx)
            self.center_y_spin.setValue(cy)
            self.width_spin.setValue(self.region.width)
            self.height_spin.setValue(self.region.height)
            self._update_z_world_fields()
            self.angle_spin.setValue(self._normalize_angle(self.region.angle))

            if self.viewer.data is not None and self.viewer.data.ndim >= 3:
                z_axis_size = self.viewer.data.shape[0]
                self.z_min_spin.setRange(0, z_axis_size - 1)
                self.z_max_spin.setRange(0, z_axis_size - 1)
            self.z_min_spin.setValue(self.region.z_min)
            self.z_max_spin.setValue(self.region.z_max)

            self.size_container.setVisible(True)

            self._toggle_row(self.width_label, self.size_container, True)
            self._toggle_row(self.angle_label, self.angle_spin, True)
            self._toggle_row(self.z_min_label, self.z_min_spin.parentWidget(), True)
            self._toggle_row(self.z_max_label, self.z_max_spin.parentWidget(), True)

        elif isinstance(self.region, (RectangleRegion, EllipseRegion)):
            shape_name = self.region.__class__.__name__.replace('Region', '')
            cx, cy = self.region.center
            self.center_x_spin.setValue(cx)
            self.center_y_spin.setValue(cy)
            self.width_spin.setValue(self.region.width)
            self.height_spin.setValue(self.region.height)
            self.angle_spin.setValue(self._normalize_angle(self.region.angle))
            
            self.size_container.setVisible(True)
            self._toggle_row(self.width_label, self.size_container, True)
            self._toggle_row(self.angle_label, self.angle_spin, True)

        else:
            shape_name = "Region"

        self.shape_value_label.setText(f"Shape: {shape_name}")
        self._update_label_field()
        self._refresh_unit_controls()
        self._update_world_fields()
        self._updating_fields = False

        self.moments_button.setVisible(True)
        self.moments_label.setText("") 

        self._update_stats()
        self._on_auto_apply_changed()
        if not self.auto_apply_checkbox.isChecked():
            self.apply_button.setEnabled(False)
        

    def closeEvent(self, event):
        if self.region_manager is not None:
            self.region_manager.on_editor_closed(self.region, self)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    def _toggle_row(self, label_widget, value_widget, visible):
        label_widget.setVisible(visible)
        value_widget.setVisible(visible)
        if isinstance(value_widget, QWidget) and value_widget.layout() is not None:
            for i in range(value_widget.layout().count()):
                item = value_widget.layout().itemAt(i).widget()
                if item is not None:
                    item.setVisible(visible)

    def _normalize_angle(self, angle):
        return ((angle + 180.0) % 360.0) - 180.0

    def _on_value_changed(self, _):
        if self._updating_fields:
            return
        if self.auto_apply_checkbox.isChecked():
            self.apply_changes()
        else:
            self.apply_button.setEnabled(True)

    def _on_auto_apply_changed(self, *_args):
        auto = self.auto_apply_checkbox.isChecked()
        self.apply_button.setEnabled(not auto)

    def apply_changes(self):
        """
        Applies the current editor values to the region object and selectively
        updates linked UI elements without a full refresh.
        """
        if self.region_manager is None:
            return
            
        params = {
            'center': (self.center_x_spin.value(), self.center_y_spin.value()),
            'label': self.nameEdit.text().strip()
        }
        if isinstance(self.region, CircleRegion):
            params['radius'] = max(self._value_in_pixels('radius'), 0.0)
        elif isinstance(self.region, (RectangleRegion, EllipseRegion, CubeRegion)):
            params['width'] = max(self._value_in_pixels('width'), 0.0)
            params['height'] = max(self._value_in_pixels('height'), 0.0)
            params['angle'] = self.angle_spin.value()
            if isinstance(self.region, CubeRegion):
                params['z_min'] = self.z_min_spin.value()
                params['z_max'] = self.z_max_spin.value()

        self.region_manager.update_region_from_editor(self.region, params)
        
        self._update_world_fields()
        if isinstance(self.region, CubeRegion):
            self._update_z_world_fields()
        self._update_stats()

        if not self.auto_apply_checkbox.isChecked():
            self.apply_button.setEnabled(False)

    def _delete_region(self):
        if self.region_manager is None:
            return
        self.region_manager.delete_region(self.region)

    def _update_stats(self):
        """Calculates statistics for the currently selected region."""
        array = None
        if isinstance(self.region, CubeRegion):
            # For CubeRegion, we need the full 3D data cube
            data_cube = getattr(self.viewer, 'data', None)
            if data_cube is not None:
                if data_cube.ndim == 4:
                    array = data_cube[0]
                elif data_cube.ndim == 3:
                    array = data_cube
        else:
            if hasattr(self.viewer, 'im'):
                array = getattr(self.viewer.im, 'get_array', lambda: None)()
        
        if array is None:
            self.stats_label.setText("")
            return
            
        stats = self.region.get_stats(array)
        lines = []
        for key, value in stats.items():
            if 'count' in key:
                display_value = int(value)
            else:
                display_value = self.region_manager._format_significant_digits(value, 4)
            nice_key = key.replace('_', ' ').title()
            lines.append(f"{nice_key}: {display_value}")
        self.stats_label.setText("\n".join(lines))

    def _build_window_title(self):
        region_id = getattr(self.region, 'region_id', None)
        if region_id is None:
            return "Region Editor"
        return f"Region {region_id}"

    def _update_label_field(self):
        self.nameEdit.blockSignals(True)
        self.nameEdit.setText(self.region.label_text)
        self.nameEdit.blockSignals(False)


    def _update_world_fields(self):
        converter = getattr(self.viewer, 'converter', None)
        wcs = getattr(converter, 'wcs', None)

        if converter is None or wcs is None:
            self.centerWorldXEdit.setText("")
            self.centerWorldYEdit.setText("")
            self.centerWorldXEdit.setEnabled(False)
            self.centerWorldYEdit.setEnabled(False)
            return

        self.centerWorldXEdit.setEnabled(True)
        self.centerWorldYEdit.setEnabled(True)

        pix_x = self.center_x_spin.value()
        pix_y = self.center_y_spin.value()

        slices = self._get_display_slices()
        if not slices:
            return

        pixel_vector = self._build_pixel_vector(pix_x, pix_y, slices, wcs)

        try:
            formatted_coords = converter.pix_to_world(*pixel_vector)
            
            # Determine which coordinate corresponds to the X and Y display axes.
            x_axis_idx_in_fits = slices.index('x')
            y_axis_idx_in_fits = slices.index('y')
            
            world_x_str = formatted_coords[x_axis_idx_in_fits]
            world_y_str = formatted_coords[y_axis_idx_in_fits]

        except Exception as e:
            print(f"Error during coordinate conversion in RegionEditor: {e}")
            world_x_str = "Conv. Error"
            world_y_str = "Conv. Error"

        # Update the UI text boxes without triggering feedback loops.
        self.centerWorldXEdit.blockSignals(True)
        self.centerWorldYEdit.blockSignals(True)
        
        self.centerWorldXEdit.setText(world_x_str)
        self.centerWorldYEdit.setText(world_y_str)

        self.centerWorldXEdit.blockSignals(False)
        self.centerWorldYEdit.blockSignals(False)

    def _on_label_changed(self):
        if self._updating_fields or self.region_manager is None:
            return
        text = self.nameEdit.text().strip()
        self.region_manager.update_region_from_editor(self.region, {'label': text})
        if not self.auto_apply_checkbox.isChecked():
            self.apply_button.setEnabled(True)


    def _on_world_edit_finished(self):
        converter = getattr(self.viewer, 'converter', None)
        # We must use the converter for this functionality.
        if converter is None or self.region_manager is None or self._updating_fields:
            return

        wcs = getattr(converter, 'wcs', None)
        if wcs is None:
            return

        world_x_str = self.centerWorldXEdit.text()
        world_y_str = self.centerWorldYEdit.text()

        slices = self._get_display_slices()
        if not slices:
            return
            
        ref_pixel = wcs.wcs.crpix
        try:
            ref_world = wcs.wcs_pix2world([ref_pixel], 0)[0]
        except Exception:
            self._update_world_fields()
            return
            
        target_world_values = list(ref_world)

        x_axis_idx_in_fits = slices.index('x')
        y_axis_idx_in_fits = slices.index('y')
        
        target_world_values[x_axis_idx_in_fits] = world_x_str
        target_world_values[y_axis_idx_in_fits] = world_y_str

        try:
            pix_coords = converter.world_to_pix(*target_world_values)
            
            x_pix = pix_coords[x_axis_idx_in_fits]
            y_pix = pix_coords[y_axis_idx_in_fits]
                
        except (ValueError, TypeError) as exc:
            print(f"Failed to parse world coordinates: {exc}")
            self._update_world_fields()
            return

        self._updating_fields = True
        self.center_x_spin.setValue(x_pix)
        self.center_y_spin.setValue(y_pix)
        self._updating_fields = False

        if self.auto_apply_checkbox.isChecked():
            self.apply_changes()
        else:
            self.apply_button.setEnabled(True)


    def _on_fill_the_view_clicked(self):
        """
        Handles the 'Fill the view' button click.
        Fetches the current axes limits and applies them to the region.
        """
        if self.viewer is None or not hasattr(self.viewer, 'ax'):
            return

        xlim = self.viewer.ax.get_xlim()
        ylim = self.viewer.ax.get_ylim()

        if isinstance(self.region, CubeRegion):
            try:
                xz_subwindow = self.viewer.subwindows[0]
                zlim = xz_subwindow.ax.get_ylim()
                self.region.fit_to_view(xlim, ylim, zlim)
            except (IndexError, AttributeError):
                self.region.fit_to_view(xlim, ylim)

        elif hasattr(self.region, 'fit_to_view'):
            self.region.fit_to_view(xlim, ylim)

        self.update_from_region()
        self.apply_changes()



    def _open_cutout_dialog(self):
        self.viewer.open_cutout_dialog(region=self.region)

    def _format_region_for_header(self, region):
        """
        Formats the region's parameters into a clean, header-like string,
        using world coordinates and current editor units where applicable.
        """
        header = []
        fm = self.region_manager._format_significant_digits

        if hasattr(self.viewer, 'filename'):
            source_fits = self.viewer.filename
        elif hasattr(self.viewer, 'fits_viewer') and hasattr(self.viewer.fits_viewer, 'filename'):
            source_fits = self.viewer.fits_viewer.filename
        header.append(f"Source FITS: {source_fits}")
        
        # --- Region Shape and Label ---
        header.append(f"Region Shape: {region.__class__.__name__}")
        if region.label_text:
            header.append(f"Region Label: {region.label_text}")

        state = region.get_state()
        converter = getattr(self.viewer, 'converter', None)
        wcs = getattr(converter, 'wcs', None)

        # --- Center in World Coordinates ---
        if 'center' in state and converter and wcs:
            try:
                cx, cy = state['center']
                pv = self._build_pixel_vector(cx, cy, self._get_display_slices(), wcs)
                ws = converter.pix_to_world(*pv)
                x_idx, y_idx = self._get_display_slices().index('x'), self._get_display_slices().index('y')
                axis_names = f"{wcs.axis_type_names[x_idx]}, {wcs.axis_type_names[y_idx]}"
                world_coords = f"({ws[x_idx]}, {ws[y_idx]})"
                header.append(f"Center ({axis_names}) = {world_coords}")
            except Exception:
                # Fallback to pixels if WCS fails
                header.append(f"Center [pix]: ({fm(state['center'][0])}, {fm(state['center'][1])})")
        
        # --- Size Parameters using Editor's current units ---
        size_params = {'radius': self.radius_unit_combo, 'width': self.size_unit_combo, 'height': self.size_unit_combo}
        for key, combo in size_params.items():
            if key in state:
                unit = combo.currentText()
                # Use the editor's own conversion logic to get the value in the selected unit
                pixel_value = state[key]
                display_value = self._convert_from_pix(key, pixel_value, unit)
                header.append(f"{key.title()} [{unit}]: {fm(display_value)}")
        
        if 'angle' in state:
            header.append(f"Angle [deg]: {fm(state['angle'])}")
        
        # --- Z Range in World Coordinates for CubeRegion ---
        if isinstance(region, CubeRegion) and 'z_min' in state and hasattr(self.viewer, 'format_pix'):
            try:
                ref_x, ref_y = state.get('center', (0, 0))
                z_min_pix, z_max_pix = state['z_min'], state['z_max']
                
                val_min = self.viewer.format_pix.convert_chpix_to_world('xy', ref_x, ref_y, z_min_pix)
                str_min = self.viewer.format_pix.convert_chval_to_world_str('xy', val_min)
                
                val_max = self.viewer.format_pix.convert_chpix_to_world('xy', ref_x, ref_y, z_max_pix)
                str_max = self.viewer.format_pix.convert_chval_to_world_str('xy', val_max)

                z_axis_name = "Z"
                if wcs:
                    try:
                        z_idx = [i for i, axis in enumerate(wcs.wcs.ctype) if axis.startswith(('VELO', 'VRAD', 'FREQ', 'VOPT'))][0]
                        z_axis_name = wcs.axis_type_names[z_idx]
                    except IndexError: pass
                header.append(f"{z_axis_name} Range: {str_min} to {str_max}")

            except Exception:
                 # Fallback to pixels if Z conversion fails
                header.append(f"Z Range [pix]: {fm(state['z_min'])} to {fm(state['z_max'])}")

        return "\n".join(header)



    def _show_moment_results(self):
        """
        Calculates moments and shows the results in a new, non-modal dialog.
        Output is shown as plain text with aligned columns and tight line spacing.
        """
        from PyQt6.QtGui import QFontDatabase

        LABEL_W = 12  # fixed label width for alignment

        def add_line(lines, label, value):
            """Append a single formatted line with aligned label."""
            lines.append(f"{label:<{LABEL_W}}: {value}")

        array, is_3d = (None, isinstance(self.region, CubeRegion))
        if is_3d:
            data_cube = getattr(self.viewer, 'data', None)
            if data_cube is not None and data_cube.ndim >= 3:
                array = data_cube[0] if data_cube.ndim == 4 else data_cube
        else:
            if hasattr(self.viewer, 'im'):
                array = getattr(self.viewer.im, 'get_array', lambda: None)()

        if array is None:
            err_dialog = MomentResultsDialog("Error", "Analysis data not found.", self.viewer)
            err_dialog.show()
            return


        moments = self.region.get_moments(array)
        if not moments:
            err_dialog = MomentResultsDialog("Error", "Could not calculate moments.", self.viewer)
            err_dialog.show()
            return


        header_content = self._format_region_for_header(self.region)
        content = header_content.splitlines()
        content.append("\n##### Results (World Coordinates) #####")


        converter, wcs = getattr(self.viewer, 'converter', None), getattr(self.viewer, 'wcs', None)

        try:
            if not (converter and wcs):
                add_line(content, "WCS", "information not available")
            else:
                mean_x = moments.get('mean_x_pix', 0)
                mean_y = moments.get('mean_y_pix', 0)
                mean_z = moments.get('mean_z_pix', 0)

                disp_axes = self._get_display_slices()
                x_idx, y_idx = disp_axes.index('x'), disp_axes.index('y')
                pv = self._build_pixel_vector(mean_x, mean_y, disp_axes, wcs)

                z_idx = -1
                if is_3d:
                    try:
                        z_idx = [i for i, ctype in enumerate(wcs.wcs.ctype)
                                if ctype.startswith(('VELO', 'VRAD', 'FREQ', 'VOPT'))][0]
                        pv[z_idx] = mean_z
                    except IndexError:
                        z_idx = -1

                ws = converter.pix_to_world(*pv)

                add_line(content, f"<{wcs.axis_type_names[x_idx]}>", f"{ws[x_idx]}")
                add_line(content, f"<{wcs.axis_type_names[y_idx]}>", f"{ws[y_idx]}")
                if is_3d and z_idx != -1:
                    header_unit = self.viewer.header.get('CUNIT3', '').replace(' ', '').lower()
                    if header_unit == '':
                        header_unit = 'km/s'
                    add_line(content, f"<{wcs.axis_type_names[z_idx]}>", f"{ws[z_idx]} {header_unit}")

                # scales
                self._compute_axis_scales()
                scale_x = self._axis_scales.get('x', {})
                scale_y = self._axis_scales.get('y', {})
                fm = self.region_manager._format_significant_digits

                # Sigma_x
                if 'sigma_x_pix' in moments and scale_x.get('deg_per_pix'):
                    q = moments['sigma_x_pix'] * scale_x['deg_per_pix'] * u.deg
                    add_line(content,
                            f"Sigma_{wcs.axis_type_names[x_idx]}",
                            f"{fm(q.value)} deg  ({fm(q.to(u.arcsec).value)} arcsec)")

                # Sigma_y
                if 'sigma_y_pix' in moments and scale_y.get('deg_per_pix'):
                    q = moments['sigma_y_pix'] * scale_y['deg_per_pix'] * u.deg
                    add_line(content,
                            f"Sigma_{wcs.axis_type_names[y_idx]}",
                            f"{fm(q.value)} deg  ({fm(q.to(u.arcsec).value)} arcsec)")

                # Sigma_z
                if is_3d and z_idx != -1 and 'sigma_z_pix' in moments:
                    z_scale = abs(wcs.wcs.cdelt[z_idx])
                    val = moments['sigma_z_pix'] * z_scale
                    text = f"{fm(val)} {header_unit}" if header_unit else f"{fm(val)}"
                    add_line(content,
                            f"Sigma_{wcs.axis_type_names[z_idx]}",
                            text)

        except Exception as e:
            content.append(f"WCS conversion failed: {e}")


        dialog_title = f"Moment Results (Region {self.region.region_id})"
        dialog_text = "\n".join(content) 

        dialog = MomentResultsDialog(dialog_title, "", self.viewer)

        dialog.text_edit.setAcceptRichText(False)
        dialog.text_edit.setPlainText(dialog_text)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(12)
        dialog.text_edit.setFont(mono)

        self.moment_dialogs.append(dialog)
        dialog.show()
