"""The spectral-axis unit boundary in one place.

takefits shows velocities in km/s, but astropy's WCS (wcslib) normalizes a
velocity axis to SI and cannot hold km/s.  Since TF-415 foundation step 1
(roadmap 10.7) the live WCS (``state.wcs``, ``viewer.wcs``) keeps those SI
numbers under its SI label, so astropy conversions on it are physical:

- Everything shown, typed, stored or written is in the display unit,
  ``spectral_metadata['current_axis_unit']`` (read it with
  :func:`display_unit`), mirrored in the header CUNIT.
- :func:`display_wcs` gives the copy of a live WCS whose spectral axis holds
  display-unit numbers.  Plots, read-outs, typed world values, catalogs and
  header cards use it; astropy keeps its SI label on the copy, so never take
  the unit from there.  WCSAxes on the SI WCS with a km/s format unit would
  pick wrong tick decimals, which is why plots use the copy.
- Frequency and wavelength axes are shown in their header unit, except the SI
  units themselves: Hz is shown in GHz and m in um
  (:func:`spectral_display_unit_for`; TF-415 slice A).  The in-memory header
  then holds that unit, as it holds km/s for velocity axes.

Before step 1 the loader and viewers rewrote the live WCS to km/s *numbers*
under astropy's m/s label (the "km/s-numbers convention", up to 4427609).
Stage D removed that convention and its switch (``TAKEFITS_SI_LIVE_WCS``).

Heuristics that decide what users see:

- Velocity CUNIT spellings (``'km s-1'``, ``'km.s-1'``, ``'km/sec'``,
  ``'KM/S'``, ``'m s-1'``, ...) are rewritten to ``'km/s'`` / ``'m/s'``
  before the WCS is built (:func:`normalize_velocity_cunits`); wcslib
  rejects the upper-case forms, and the rules below compare literally.
- ``'m/s'`` or a missing CUNIT is shown in km/s only when |CDELT| >
  ``MS_TO_KMS_MIN_STEP``; finer m/s cubes stay in m/s.  With a CD matrix
  the step is the CD row (:func:`axis_step`, :func:`header_axis_step`);
  ``wcs.wcs.cdelt`` reads 1 there, so step reads go through those helpers.
- A velocity axis without CUNIT and a smaller step is taken to be km/s: the
  WCS is relabelled km/s and wcslib rescales it to SI.
- A FREQ axis without CUNIT is read in Hz, a WAVE / AWAV axis in m (the
  FITS defaults), and both are then shown in GHz / um.

``apply_load_convention`` runs once in ``load_fits``.
``apply_viewer_convention`` runs for every viewer (the main window and its
XZ/ZY windows) on the loaded objects.  Neither rescales the live WCS; they
decide the display unit (metadata and header).  They differ in what users
see:

- the viewer also reads an axis the loader did not identify: it falls back
  to axis 3, so a CTYPE such as ``'VEL'`` in m/s is shown in km/s in the GUI
  but in m/s headless;
- for a km/s header the viewer only records metadata;
- the viewer has no "missing CUNIT means km/s" branch;
- 2-D images whose axis the loader did not identify are left in SI.

A rest-frequency or systemic-z change (the Unit Conversion panel, the
``set_spectral_axis`` action) goes through :func:`apply_spectral_axis` and
:func:`apply_rest_frequency` with an explicit intent: re-reference the
velocities (the observed frequencies stay) or change the metadata only (the
velocity axis stays).  It moves the live WCS, the header and the metadata
together, and :func:`refresh_display_wcs` moves the display copies that plots
already hold.

Places that still read or convert units by their own rules: the K <-> Jy
frequency axis of the unit conversion use case (header numbers, radio
formula) and the catalog ``*_kms`` columns (display-unit numbers, so m/s for
fine m/s cubes).

``tests/test_spectral_units_characterization.py`` pins this behaviour and
``tests/test_spectral_units_guard.py`` lists every WCS unit-label read, step
read and direct WCS evaluation, and keeps new unit handling from bypassing
this module.
"""
from __future__ import annotations

import math
import re
import warnings
import weakref
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u

# |CDELT| above which an 'm/s' (or unit-less) velocity axis is converted.
MS_TO_KMS_MIN_STEP = 100.0


def classify_axis_type(ctype: str):
    """Classify a CTYPE: 'frequency', 'velocity', 'wavelength' (WAVE, AWAV) or 'unknown'."""
    if not ctype:
        return 'unknown'
    ctype_upper = str(ctype).upper()
    if 'FREQ' in ctype_upper:
        return 'frequency'
    if any(tag in ctype_upper for tag in ('VRAD', 'VELO', 'VOPT')):
        return 'velocity'
    if ctype_upper[:4] in ('WAVE', 'AWAV'):
        return 'wavelength'
    return 'unknown'


def identify_spectral_axis(header):
    """Identify the first spectral axis (frequency, velocity or wavelength), 1-based."""
    if header is None:
        return None
    try:
        naxis = int(header.get('NAXIS', 0))
    except (TypeError, ValueError):
        return None

    for axis in range(1, naxis + 1):
        ctype = header.get(f'CTYPE{axis}', '')
        if classify_axis_type(ctype) != 'unknown':
            return axis
    return None


def compact_unit(value: Any) -> str:
    """Unit text without spaces, in lower case (``'KM / S'`` -> ``'km/s'``)."""
    return str(value).replace(' ', '').lower()


# Spellings without a separator, which astropy does not parse; on a velocity
# axis they can only mean these units.
_KMS_SPELLINGS = frozenset({'kms-1', 'kms^-1', 'km/sec', 'kmsec-1'})
_MS_SPELLINGS = frozenset({'ms-1', 'ms^-1', 'm/sec', 'msec-1'})


def canonical_velocity_unit(text: Any) -> Optional[str]:
    """``'km/s'`` or ``'m/s'`` for a spelling of that unit on a velocity axis, else None.

    Case is ignored ('KM/S'), and so are the FITS separators ('km s-1',
    'km.s-1').  Other velocity units (cm/s, km/h) return None.
    """
    compact = compact_unit(text)
    if compact in ('km/s', 'm/s'):
        return compact
    if compact in _KMS_SPELLINGS:
        return 'km/s'
    if compact in _MS_SPELLINGS:
        return 'm/s'
    raw = str(text).strip()
    for candidate in (raw, raw.lower()):
        try:
            unit = u.Unit(candidate)
        except Exception:
            continue
        if not unit.is_equivalent(u.m / u.s):
            continue
        metres_per_second = (1 * unit).to(u.m / u.s).value
        if math.isclose(metres_per_second, 1000.0):
            return 'km/s'
        if math.isclose(metres_per_second, 1.0):
            return 'm/s'
        return None
    return None


def normalize_velocity_cunits(header) -> List[Tuple[int, str, str]]:
    """Rewrite velocity-axis CUNIT spellings to ``'km/s'`` / ``'m/s'`` in place.

    Run before a WCS is built from ``header``.  Returns ``(axis, old, new)``
    for each rewritten card.
    """
    changes: List[Tuple[int, str, str]] = []
    if header is None:
        return changes
    try:
        naxis = int(header.get('NAXIS', 0))
    except (TypeError, ValueError):
        return changes
    for axis in range(1, naxis + 1):
        if classify_axis_type(header.get(f'CTYPE{axis}', '')) != 'velocity':
            continue
        key = f'CUNIT{axis}'
        original = header.get(key)
        if not isinstance(original, str) or not original.strip():
            continue
        canonical = canonical_velocity_unit(original)
        if canonical is not None and original.strip() != canonical:
            header[key] = canonical
            changes.append((axis, original, canonical))
    return changes


def display_unit(spectral_metadata) -> str:
    """Unit of the spectral numbers from ``spectral_metadata``, or ''.

    Accepts a bare unit (``'km/s'``) or a decorated label
    (``'Velocity [km/s]'``), matching what ``export_pv_fits`` parses.
    """
    metadata = spectral_metadata or {}
    raw = str(metadata.get('current_axis_unit', '') or '').strip()
    if not raw:
        return ''
    match = re.search(r"\[(.*?)\]", raw)
    return (match.group(1) if match else raw).strip()


def spectral_wcs_axis(wcs, spectral_metadata=None) -> Optional[int]:
    """0-based WCS axis of the spectral coordinate, or None.

    wcslib's spectral axis first; for CTYPEs it does not know ('VEL',
    'VELOCITY') the metadata axis, as the viewer falls back to it.
    """
    try:
        spec = int(getattr(wcs.wcs, 'spec', -1))
    except (TypeError, ValueError):
        spec = -1
    naxis = int(getattr(wcs.wcs, 'naxis', 0) or 0)
    if 0 <= spec < naxis:
        return spec
    axis_index = (spectral_metadata or {}).get('axis_index')
    try:
        axis_index = int(axis_index)
    except (TypeError, ValueError):
        return None
    return axis_index - 1 if 1 <= axis_index <= naxis else None


def display_numbers_unit(wcs, spectral_metadata=None) -> str:
    """Unit of the spectral numbers ``display_wcs`` holds, for files written from them.

    That is the display unit when it is a unit of the WCS axis's kind
    (km/s on a velocity axis, GHz on a frequency axis, um on a wavelength
    axis), else the WCS's own SI unit, which display_wcs then leaves alone.
    """
    shown = display_unit(spectral_metadata)
    axis = spectral_wcs_axis(wcs, spectral_metadata) if wcs is not None else None
    if axis is None:
        return shown
    try:
        wcs_unit = wcs.wcs.cunit[axis]
        text = wcs_unit.to_string().replace(' ', '')
    except Exception:
        return shown
    if shown and text and _equivalent_units(wcs_unit, shown):
        return shown
    return text or shown


def _equivalent_units(first, second) -> bool:
    try:
        a, b = u.Unit(first), u.Unit(second)
    except Exception:
        return False
    return a != u.dimensionless_unscaled and a.is_equivalent(b)


def spectral_sub_wcs(wcs, spectral_metadata=None):
    """1-D WCS of the spectral axis (as :func:`spectral_wcs_axis` finds it), or None.

    ``wcs.sub(['spectral'])`` has no axis for CTYPEs wcslib does not treat
    as spectral ('VEL', 'VELOCITY'); the viewer reads those through the
    metadata axis, so this does too.
    """
    if wcs is None:
        return None
    try:
        sub = wcs.sub(['spectral'])
        if int(sub.wcs.naxis) == 1:
            return sub
    except Exception:
        pass
    axis = spectral_wcs_axis(wcs, spectral_metadata)
    if axis is None:
        return None
    try:
        return wcs.sub([axis + 1])
    except Exception:
        return None


def _display_factor(wcs, axis: int, spectral_metadata) -> float:
    """Display units per WCS unit on the spectral ``axis`` (1.0 unless the two are units of one kind)."""
    target_text = display_unit(spectral_metadata)
    if not target_text:
        return 1.0
    try:
        wcs_unit = u.Unit(wcs.wcs.cunit[axis])
        target = u.Unit(target_text)
    except Exception:
        return 1.0
    if wcs_unit == u.dimensionless_unscaled or not wcs_unit.is_equivalent(target):
        return 1.0
    return float(wcs_unit.to(target))


def _scale_axis_step(wcs, axis: int, factor: float) -> None:
    """Multiply the step of WCS ``axis`` by ``factor``: its CD row, else CDELT (the precedence of :func:`axis_step`)."""
    if wcs.wcs.has_cd() and not wcs.wcs.has_pc():
        cd = np.array(wcs.wcs.cd, dtype=float)
        cd[axis, :] *= factor
        wcs.wcs.cd = cd
    else:
        cdelt = np.array(wcs.wcs.cdelt, dtype=float)
        cdelt[axis] *= factor
        with warnings.catch_warnings():
            # astropy warns about a CD matrix next to PCi_j, which wcslib ignores
            warnings.simplefilter("ignore", RuntimeWarning)
            wcs.wcs.cdelt = cdelt


def _decimal_scaler(factor: float):
    """``x -> x * factor``; a power of ten moves the decimal point of ``x`` instead.

    Unit changes between SI prefixes are powers of ten (Hz -> GHz, m -> um,
    m/s -> km/s).  Shifting the shortest decimal form of the number keeps
    115271201800 Hz at 115.2712018 GHz and 6.5628e-7 m at 0.65628 um, where
    a binary multiplication gives 115.27120180000001 and 0.6562800000000001.
    Works on numbers and on NumPy arrays.
    """
    exponent = math.log10(factor) if factor > 0 else float('nan')
    if not (math.isfinite(exponent) and math.isclose(exponent, round(exponent), abs_tol=1e-9)):
        return lambda value: value * factor
    shift = int(round(exponent))

    def _shift(number):
        number = float(number)
        if not math.isfinite(number):
            return number * factor
        return float(Decimal(repr(number)).scaleb(shift))

    def scale(value):
        if isinstance(value, np.ndarray):
            return np.array([_shift(item) for item in value.ravel()], dtype=float).reshape(value.shape)
        return _shift(value)

    return scale


def _scaled_axis_copy(wcs, axis: int, factor: float):
    copy = wcs.deepcopy()
    scale = _decimal_scaler(factor)
    crval = np.array(copy.wcs.crval, dtype=float)
    crval[axis] = scale(crval[axis])
    copy.wcs.crval = crval
    if copy.wcs.has_cd() and not copy.wcs.has_pc():
        cd = np.array(copy.wcs.cd, dtype=float)
        cd[axis, :] = scale(cd[axis, :])
        copy.wcs.cd = cd
    else:
        cdelt = np.array(copy.wcs.cdelt, dtype=float)
        cdelt[axis] = scale(cdelt[axis])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a CD matrix next to PCi_j
            copy.wcs.cdelt = cdelt
    # Marks a display copy (kept by deepcopy, sub() and pickle) so it is never scaled twice.
    copy._takefits_display_copy = True
    return copy


_DISPLAY_WCS_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _cdelt_tuple(wcs) -> Tuple:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "cdelt will be ignored since cd is present"
        return tuple(np.asarray(wcs.wcs.cdelt, dtype=float))


def _display_key(wcs, axis: int, factor: float) -> Tuple:
    cd = tuple(np.asarray(wcs.wcs.cd, dtype=float).ravel()) if wcs.wcs.has_cd() else ()
    return (
        axis,
        factor,
        tuple(np.asarray(wcs.wcs.crval, dtype=float)),
        _cdelt_tuple(wcs),
        tuple(np.asarray(wcs.wcs.crpix, dtype=float)),
        cd,
        tuple(str(unit) for unit in wcs.wcs.cunit),
    )


def display_wcs(wcs, spectral_metadata=None):
    """The WCS whose spectral numbers are in the display unit.

    Plotting, read-outs and typed world values use it; computations that
    need physical units use the live WCS.  The live WCS keeps SI numbers,
    and this returns a cached copy whose velocity axis is scaled to the
    display unit (the live WCS itself when nothing needs scaling).  astropy
    keeps its SI label on that copy, so never read the unit from it; WCSAxes
    on an SI WCS with a km/s format unit would pick wrong tick decimals
    (roadmap 10.7).
    """
    if wcs is None or getattr(wcs, '_takefits_display_copy', False):
        return wcs
    axis = spectral_wcs_axis(wcs, spectral_metadata)
    if axis is None:
        return wcs
    factor = _display_factor(wcs, axis, spectral_metadata)
    if factor == 1.0:
        return wcs
    key = _display_key(wcs, axis, factor)
    cached = _DISPLAY_WCS_CACHE.get(wcs)
    if cached is not None and cached[0] == key:
        return cached[1]
    copy = _scaled_axis_copy(wcs, axis, factor)
    _DISPLAY_WCS_CACHE[wcs] = (key, copy)
    return copy


def viewer_display_wcs(owner):
    """``display_wcs`` of an object with ``wcs`` and ``spectral_metadata`` (a viewer or a state)."""
    return display_wcs(getattr(owner, 'wcs', None), getattr(owner, 'spectral_metadata', None))


def _same_wcs_structure(first, second) -> bool:
    a, b = first.wcs, second.wcs
    return (
        int(a.naxis) == int(b.naxis)
        and [str(value) for value in a.ctype] == [str(value) for value in b.ctype]
        and [str(value) for value in a.cunit] == [str(value) for value in b.cunit]
        and bool(a.has_cd()) == bool(b.has_cd())
        and bool(a.has_pc()) == bool(b.has_pc())
    )


def _copy_wcs_numbers(target, source) -> None:
    """Write the reference point, linear transform and rest values of ``source`` into ``target``."""
    t, s = target.wcs, source.wcs
    t.crval = np.array(s.crval, dtype=float)
    t.crpix = np.array(s.crpix, dtype=float)
    if s.has_cd():
        t.cd = np.array(s.cd, dtype=float)
    if s.has_pc():
        with warnings.catch_warnings():
            # astropy warns about a CD matrix next to PCi_j, which wcslib ignores
            warnings.simplefilter("ignore", RuntimeWarning)
            t.cdelt = np.array(s.cdelt, dtype=float)
            t.pc = np.array(s.pc, dtype=float)
    t.restfrq = float(s.restfrq)
    t.restwav = float(s.restwav)
    t.set()


def wcs_numbers(wcs) -> Optional[Tuple]:
    """The reference point, linear transform and rest values of ``wcs``, to tell whether they moved."""
    if wcs is None:
        return None
    w = wcs.wcs
    numbers = [
        tuple(np.asarray(w.crval, dtype=float)),
        tuple(np.asarray(w.crpix, dtype=float)),
        _cdelt_tuple(wcs),
        float(w.restfrq),
        float(w.restwav),
    ]
    if w.has_cd():
        numbers.append(tuple(np.asarray(w.cd, dtype=float).ravel()))
    return tuple(numbers)


def copy_wcs_numbers(target, source) -> bool:
    """Write the numbers of ``source`` into ``target`` in place if both have the same structure.

    Returns whether it did.  For restoring a WCS that others hold (for
    example a panel's reset of its rest-frequency change).
    """
    if target is None or source is None or not _same_wcs_structure(target, source):
        return False
    _copy_wcs_numbers(target, source)
    return True


def refresh_display_wcs(wcs, spectral_metadata=None, previous=None):
    """Bring the display WCS that plots already hold up to date, in place; returns the one to use.

    WCSAxes keeps the WCS it was built with, and so do open panels.  After a
    change that only moves numbers (a rest-frequency change, or the undo that
    brings back the WCS from before it), this writes the new numbers into that
    object, so everything holding it follows without being rebuilt.
    ``previous`` is the object the plots hold (a viewer's
    ``displaymap.wcs``); without it, the cached display copy of ``wcs``.
    Nothing is written when the two differ in structure (axes, types, units,
    matrix form) or when one is a display copy and the other a live WCS.
    """
    if wcs is None:
        return None
    cached = _DISPLAY_WCS_CACHE.get(wcs)
    target = previous if previous is not None else (cached[1] if cached is not None else None)
    fresh = display_wcs(wcs, spectral_metadata)
    if target is None or target is fresh:
        return fresh
    fresh_is_copy = bool(getattr(fresh, '_takefits_display_copy', False))
    if bool(getattr(target, '_takefits_display_copy', False)) != fresh_is_copy:
        return fresh
    if not _same_wcs_structure(target, fresh):
        return fresh
    _copy_wcs_numbers(target, fresh)
    if not fresh_is_copy:
        return fresh
    # One display copy per live WCS: later display_wcs calls return the object the plots hold.
    _DISPLAY_WCS_CACHE[wcs] = (_DISPLAY_WCS_CACHE[wcs][0], target)
    return target


def _signed_row_norm(row, diagonal: int):
    """Length of a row of the linear transform, with the sign of its diagonal term."""
    magnitude = np.sqrt((row ** 2).sum())
    return np.copysign(magnitude, row[diagonal]) if row[diagonal] != 0 else magnitude


def _is_unit_row(row) -> bool:
    return bool(np.isclose(np.sqrt((np.asarray(row, dtype=np.float64) ** 2).sum()), 1.0, rtol=0.0, atol=1e-12))


def axis_step(wcs, axis: int) -> float:
    """World step per pixel of WCS ``axis``, also for a CD matrix.

    ``wcs.wcs.cdelt`` reads 1 when the header gives the step as CDi_j, so
    precision, integration widths and spectral axes built from it were wrong
    for CD-matrix cubes.  The step is the length of the axis's row of the
    linear transform, signed by its diagonal term: the CD row, or CDELT times
    the PC row.  A PC row that only rotates (length 1) leaves CDELT as the
    step; a scaled one is what astropy writes for a CD matrix (CDELT = 1).
    With both CDi_j and PCi_j, wcslib uses PCi_j and CDELTi, and so does this.

    The step is a NumPy float64, as ``cdelt[i]`` is: a Python float would
    leave float32 data in float32 when multiplied (NumPy 2 promotion).
    """
    axis = int(axis)
    if wcs.wcs.has_cd() and not wcs.wcs.has_pc():
        return _signed_row_norm(np.asarray(wcs.wcs.cd, dtype=np.float64)[axis], axis)
    with warnings.catch_warnings():
        # astropy warns about a CD matrix next to PCi_j, which wcslib ignores
        warnings.simplefilter("ignore", RuntimeWarning)
        cdelt = np.float64(wcs.wcs.cdelt[axis])
        try:
            pc_row = np.asarray(wcs.wcs.get_pc(), dtype=np.float64)[axis]  # runs wcsset
        except Exception:
            return cdelt
    if _is_unit_row(pc_row):
        return cdelt
    return _signed_row_norm(cdelt * pc_row, axis)


_PC_KEY = re.compile(r'PC\d+_\d+')


def _header_matrix_row(header, prefix: str, axis_number: int) -> Dict[str, float]:
    row = re.compile(rf'{prefix}{int(axis_number)}_(\d+)')
    return {key: float(header[key]) for key in (str(k) for k in header.keys()) if row.fullmatch(key)}


def _header_cd_row_keys(header, axis_number: int) -> List[str]:
    """The CDn_j keys of axis n when wcslib reads the header's steps from CDi_j (no PCi_j)."""
    if header is None:
        return []
    keys = [str(key) for key in header.keys()]
    if any(_PC_KEY.fullmatch(key) for key in keys):
        return []
    row = re.compile(rf'CD{int(axis_number)}_\d+')
    return [key for key in keys if row.fullmatch(key)]


def _header_scaled_pc_row(header, axis_number: int) -> Optional[List[float]]:
    """The PCn_j row of axis n when it also scales (length not 1), else None."""
    if header is None or not any(_PC_KEY.fullmatch(str(key)) for key in header.keys()):
        return None
    row = _header_matrix_row(header, 'PC', axis_number)
    row.setdefault(f'PC{int(axis_number)}_{int(axis_number)}', 1.0)  # FITS default
    values = list(row.values())
    return None if _is_unit_row(values) else values


def header_axis_step(header, axis_number: int) -> Optional[float]:
    """Step of FITS axis n in a header, read as :func:`axis_step` reads a WCS.

    CDELTn, the CD row, or CDELTn times a scaled PC row.
    """
    if header is None:
        return None
    n = int(axis_number)
    row_keys = _header_cd_row_keys(header, n)
    try:
        if row_keys:
            row = {key: float(header[key]) for key in row_keys}
            if any(row.values()):
                magnitude = math.sqrt(sum(v * v for v in row.values()))
                diagonal = row.get(f'CD{n}_{n}', 0.0)
                return math.copysign(magnitude, diagonal) if diagonal else magnitude
        value = header.get(f'CDELT{n}')
        pc_row = _header_scaled_pc_row(header, n)
        if pc_row is not None:
            cdelt = 1.0 if value is None else float(value)  # FITS default
            magnitude = abs(cdelt) * math.sqrt(sum(v * v for v in pc_row))
            diagonal = cdelt * float(header.get(f'PC{n}_{n}', 1.0))
            return math.copysign(magnitude, diagonal) if diagonal else magnitude
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def set_header_axis_step(header, axis_number: int, step: float) -> None:
    """Write the step of FITS axis n, keeping the header's form.

    A CD row is scaled as a whole, and so is CDELTn next to a scaled PC row,
    so ``step`` is on the scale of :func:`header_axis_step`; otherwise CDELTn
    is set to ``step``.
    """
    n = int(axis_number)
    current = header_axis_step(header, n)
    row_keys = _header_cd_row_keys(header, n)
    if row_keys and current:
        factor = float(step) / current
        for key in row_keys:
            header[key] = float(header[key]) * factor
        return
    if _header_scaled_pc_row(header, n) is not None and current:
        header[f'CDELT{n}'] = float(header.get(f'CDELT{n}', 1.0)) * (float(step) / current)
        return
    header[f'CDELT{n}'] = step


def wcs_axis_unit(wcs, axis: int) -> str:
    """astropy's unit label of WCS axis ``axis``, compacted.

    This is the SI label, so it reads ``'m/s'`` for a velocity axis whose
    numbers are km/s.  Use it to detect SI axes, not as the display unit.
    """
    return compact_unit(wcs.wcs.cunit[axis].to_string())


def ms_to_kms(value):
    """Convert m/s to km/s."""
    return (value * u.m / u.s).to(u.km / u.s).value


def kms_per_unit(unit_text: str) -> Optional[float]:
    """km/s in one ``unit_text`` (0.001 for ``'m/s'``), or None if not a velocity unit."""
    try:
        return (1 * u.Unit(unit_text)).to(u.km / u.s).value
    except Exception:
        return None


def convert_velocity_axis_cards(header, axis_number: int, target_unit) -> None:
    """Rescale a velocity axis's CRVAL/CDELT/CD row from its CUNIT to ``target_unit``.

    For headers whose numbers come from an SI WCS (``to_header()``) and are
    then relabelled in the display unit.  Other axes are left alone.
    """
    _convert_axis_cards(header, axis_number, target_unit, velocity_only=True)


def convert_axis_cards(header, axis_number: int, target_unit) -> None:
    """As :func:`convert_velocity_axis_cards`, for any equivalent units (Hz -> GHz, m -> um).

    Output headers whose numbers come from an SI WCS then keep the input's
    unit with matching numbers.
    """
    _convert_axis_cards(header, axis_number, target_unit, velocity_only=False)


def _convert_axis_cards(header, axis_number: int, target_unit, *, velocity_only: bool) -> None:
    current = str(header.get(f'CUNIT{axis_number}', '') or '').strip()
    target = str(target_unit or '').strip()
    if not current or not target:
        return
    try:
        current_unit, target_unit_obj = u.Unit(current), u.Unit(target)
    except Exception:
        return
    velocity = u.m / u.s
    if velocity_only and not (current_unit.is_equivalent(velocity) and target_unit_obj.is_equivalent(velocity)):
        return
    if not current_unit.is_equivalent(target_unit_obj):
        return
    factor = float(current_unit.to(target_unit_obj))
    if factor == 1.0:
        return
    scale = _decimal_scaler(factor)
    keys = [f'CRVAL{axis_number}', f'CDELT{axis_number}']
    keys += [key for key in header.keys() if key.startswith(f'CD{axis_number}_')]
    for key in keys:
        if key in header:
            header[key] = scale(float(header[key]))


def _store_kms_axis_in_header(wcs, header, axis_index: int) -> None:
    """Write the m/s axis of the SI live WCS into the header in km/s."""
    wcs_axis = axis_index - 1
    cdelt, crval = ms_to_kms(axis_step(wcs, wcs_axis)), ms_to_kms(wcs.wcs.crval[wcs_axis])
    header[f'CUNIT{axis_index}'] = 'km/s'
    set_header_axis_step(header, axis_index, cdelt)
    header[f'CRVAL{axis_index}'] = crval


# The SI units read badly on frequency and wavelength axes (115271201800 Hz,
# 0.0000006563 m), so they are shown in GHz and um; any other unit the header
# gives (MHz, Angstrom, nm, ...) is shown as it is (TF-415 slice A, D1).
_SI_DISPLAY_REPLACEMENTS = {'frequency': (u.Hz, 'GHz'), 'wavelength': (u.m, 'um')}


def spectral_display_unit_for(kind: str, header_unit) -> Optional[str]:
    """The display unit of a frequency or wavelength axis with CUNIT ``header_unit``.

    None when the header unit is not a unit of that kind (it is then left as
    it is).
    """
    si_unit, replacement = _SI_DISPLAY_REPLACEMENTS[kind]
    text = str(header_unit or '').strip()
    if not text:
        return replacement
    try:
        unit = u.Unit(text)
    except Exception:
        return None
    if not unit.is_equivalent(si_unit):
        return None
    return replacement if unit == si_unit else text


def _show_frequency_or_wavelength(header, spectral_metadata: Dict[str, Any], axis_number: int, kind: str) -> None:
    """Put a frequency or wavelength axis of the header into its display unit."""
    key = f'CUNIT{axis_number}'
    raw = str(header.get(key, '') or '').strip()
    if not raw:
        # FITS reads a FREQ axis without CUNIT in Hz, and a WAVE / AWAV axis in m.
        raw = 'Hz' if kind == 'frequency' else 'm'
        print("\033[96mCUNIT{} is not found. Interpreted {} unit as {}.\033[0m".format(axis_number, kind, raw))
        header[key] = raw
    shown = spectral_display_unit_for(kind, raw)
    if shown is None:
        spectral_metadata['current_axis_unit'] = raw
        return
    if shown != raw:
        convert_axis_cards(header, axis_number, shown)
        header[key] = shown
    spectral_metadata['current_axis_unit'] = shown


def _label_axis_unit(wcs, axis: int, unit_text: str) -> None:
    """Tell the WCS the unit takefits reads an axis in; wcslib rescales spectral axes to SI."""
    wcs.wcs.cunit[axis] = unit_text
    wcs.wcs.set()


def _mark_converted_from_ms(spectral_metadata: Dict[str, Any]) -> None:
    spectral_metadata['velocity_unit_adjusted'] = True
    spectral_metadata['velocity_unit_original'] = 'm/s'
    spectral_metadata['velocity_unit_target'] = 'km/s'
    spectral_metadata['current_axis_unit'] = 'km/s'


def apply_load_convention(wcs, header, spectral_metadata: Dict[str, Any], spec_axis_idx) -> None:
    """Decide the display unit of a freshly loaded cube (``load_fits``).

    ``spec_axis_idx`` is the 1-based spectral FITS axis, or None when the
    loader identified none.  The WCS keeps its SI numbers; only a unit-less
    axis is labelled so wcslib reads it in the assumed unit.  Mutates
    ``wcs``, ``header`` and ``spectral_metadata`` and prints the loader's
    notices.
    """
    if spec_axis_idx and spec_axis_idx <= wcs.wcs.naxis:
        wcs_axis_idx = spec_axis_idx - 1
        header_cunit_key = f'CUNIT{spec_axis_idx}'
        unit_header = compact_unit(header.get(header_cunit_key, ''))
        try:
            unit_wcs = wcs_axis_unit(wcs, wcs_axis_idx)
        except Exception:
            unit_wcs = ''

        kind = classify_axis_type(header.get(f'CTYPE{spec_axis_idx}', ''))
        if kind in ('frequency', 'wavelength'):
            _show_frequency_or_wavelength(header, spectral_metadata, spec_axis_idx, kind)
        elif unit_header == 'km/s' and unit_wcs != 'km/s':
            pass  # astropy holds the km/s header in SI; the header stays km/s
        elif unit_header in ('m/s', '') and abs(axis_step(wcs, wcs_axis_idx)) > MS_TO_KMS_MIN_STEP:
            if unit_header == '':
                print("\033[96mCUNIT{} is not found. Interpreted velocity unit as m/s.\033[0m".format(spec_axis_idx))
                _label_axis_unit(wcs, wcs_axis_idx, 'm/s')
            _store_kms_axis_in_header(wcs, header, spec_axis_idx)
            _mark_converted_from_ms(spectral_metadata)
            print("\033[96mConverted velocity unit from m/s to km/s.\033[0m")
        elif unit_header == '':
            print("\033[96mCUNIT{} is not found. Interpreted velocity unit as km/s.\033[0m".format(spec_axis_idx))
            header[header_cunit_key] = 'km/s'
            # The WCS took the unit-less numbers for m/s; relabel so wcslib scales them.
            _label_axis_unit(wcs, wcs_axis_idx, 'km/s')
            spectral_metadata['velocity_unit_adjusted'] = False  # no conversion needed
            spectral_metadata['velocity_unit_original'] = 'km/s'  # assumed
            spectral_metadata['velocity_unit_target'] = 'km/s'
            spectral_metadata['current_axis_unit'] = 'km/s'

        if spectral_metadata.get('current_axis_unit') is None:
            spectral_metadata['current_axis_unit'] = header.get(header_cunit_key, '').strip() or None
        spectral_metadata['current_axis_ctype'] = header.get(
            f'CTYPE{spec_axis_idx}', spectral_metadata.get('current_axis_ctype')
        )


def apply_viewer_convention(wcs, header, spectral_meta: Dict[str, Any]) -> bool:
    """The viewer's pass over the loaded objects; returns ``velocity_unit_converted``.

    Runs for the main window and each XZ/ZY window.  Mutates ``wcs``,
    ``header`` and ``spectral_meta``; see the module notes for how it differs
    from :func:`apply_load_convention`.
    """
    converted = bool(spectral_meta.get('velocity_unit_adjusted', False))
    axis_index = spectral_meta.get('axis_index')
    if axis_index is None and wcs.wcs.naxis >= 3:
        axis_index = 3
    if axis_index and spectral_meta.get('axis_index') is None:
        spectral_meta['axis_index'] = axis_index

    if axis_index and axis_index <= wcs.wcs.naxis:
        wcs_axis_idx = axis_index - 1
        try:
            unit_wcs = wcs_axis_unit(wcs, wcs_axis_idx)
        except Exception:
            unit_wcs = ''
        unit_header = compact_unit(header.get(f'CUNIT{axis_index}', ''))
        ctype_key = f'CTYPE{axis_index}'

        if not spectral_meta.get('velocity_unit_adjusted', False):
            if unit_header == 'km/s' and unit_wcs != 'km/s':
                spectral_meta['current_axis_unit'] = 'km/s'
                spectral_meta['current_axis_type'] = 'velocity'
                spectral_meta['current_axis_ctype'] = header.get(ctype_key, spectral_meta.get('current_axis_ctype'))
            elif unit_header in ('m/s', '') and abs(axis_step(wcs, wcs_axis_idx)) > MS_TO_KMS_MIN_STEP:
                if not unit_wcs:
                    # a unit-less axis wcslib does not know ('VEL'): read it as m/s
                    _label_axis_unit(wcs, wcs_axis_idx, 'm/s')
                _store_kms_axis_in_header(wcs, header, axis_index)
                _mark_converted_from_ms(spectral_meta)
                spectral_meta['current_axis_type'] = 'velocity'
                spectral_meta['current_axis_ctype'] = header.get(ctype_key, spectral_meta.get('current_axis_ctype'))
                converted = True
        else:
            if spectral_meta.get('current_axis_unit') is None:
                current_unit = header.get(f'CUNIT{axis_index}', '')
                spectral_meta['current_axis_unit'] = current_unit.strip() if isinstance(current_unit, str) else None
            if spectral_meta.get('current_axis_type') in (None, 'unknown'):
                spectral_meta['current_axis_type'] = 'velocity'
            spectral_meta['current_axis_ctype'] = header.get(ctype_key, spectral_meta.get('current_axis_ctype'))
    return converted


# --- Rest-frequency changes (TF-415 foundation step 2, roadmap 10.7 design note 7) ---
#
# A rest-frequency change never resamples the data, and it says what it keeps:
# - 'rereference': the observed frequencies.  The velocity axis is re-derived
#   for the new line; CRVAL and the step change together, so the channel width
#   in Hz stays as observed.
# - 'metadata': the velocity axis.  Only the rest frequency changes, for
#   example to fix RESTFRQ for K <-> Jy/beam on a velocity-native cube.

REST_FREQUENCY_INTENTS = ('rereference', 'metadata')

_SPEED_OF_LIGHT_MS = 299792458.0
_REST_FREQUENCY_KEYS = ('RESTFRQ', 'RESTFREQ')
_VELOCITY_CONVENTIONS = {'VRAD': 'radio', 'VOPT': 'optical', 'VELO': 'relativistic'}
_NON_VELOCITY_KINDS = {
    'FREQ': 'frequency', 'ENER': 'energy', 'WAVN': 'wavenumber',
    'WAVE': 'wavelength', 'AWAV': 'wavelength', 'ZOPT': 'redshift', 'BETA': 'velocity ratio',
}


def _positive_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def rest_frequency_hz(wcs=None, header=None) -> Optional[float]:
    """The cube's rest frequency in Hz, or None.

    RESTFRQ (or its older spelling RESTFREQ) from the header, else the WCS's;
    a rest wavelength alone (RESTWAV) gives c / RESTWAV.
    """
    if header is not None:
        for key in _REST_FREQUENCY_KEYS:
            value = _positive_float(header.get(key))
            if value is not None:
                return value
    if wcs is not None:
        value = _positive_float(getattr(wcs.wcs, 'restfrq', 0.0))
        if value is not None:
            return value
        wavelength = _positive_float(getattr(wcs.wcs, 'restwav', 0.0))
        if wavelength is not None:
            return _SPEED_OF_LIGHT_MS / wavelength
    if header is not None:
        wavelength = _positive_float(header.get('RESTWAV'))
        if wavelength is not None:
            return _SPEED_OF_LIGHT_MS / wavelength
    return None


def _velocity_convention(wcs, axis: int) -> Optional[str]:
    """'radio', 'optical' or 'relativistic' for a velocity axis, as wcslib reads its CTYPE.

    wcslib translates the AIPS forms ('VELO-LSR' with VELREF, 'FELO-HEL'), so
    the WCS CTYPE is read, not the header's.
    """
    return _VELOCITY_CONVENTIONS.get(str(wcs.wcs.ctype[axis]).upper()[:4])


def velocity_convention(wcs, axis: int) -> Optional[str]:
    """'radio', 'optical' or 'relativistic' for WCS axis ``axis``, as wcslib reads its CTYPE; else None."""
    return _velocity_convention(wcs, axis)


def refresh_alternate_reference(header, axis_number: int, rest_hz) -> None:
    """Keep the AIPS / CASA alternate reference the other quantity of the spectral axis.

    ALTRVAL at pixel ALTRPIX holds the radio velocity (m/s) on a frequency
    axis and the frequency (Hz) on a velocity axis.  After an axis has been
    turned from one into the other, the pair is rewritten at the new axis's
    reference pixel.  Headers without ALTRVAL are left alone, and so are axes
    this cannot evaluate (no rest frequency, relativistic velocities).
    """
    if header is None or 'ALTRVAL' not in header:
        return
    rest = _positive_float(rest_hz)
    n = int(axis_number)
    ctype = str(header.get(f'CTYPE{n}', '') or '').strip().upper()
    kind = classify_axis_type(ctype)
    if rest is None or kind not in ('frequency', 'velocity'):
        return
    default_unit = 'Hz' if kind == 'frequency' else 'm/s'
    try:
        unit = u.Unit(str(header.get(f'CUNIT{n}', '') or default_unit).strip())
        crval = float(header.get(f'CRVAL{n}', 0.0))
        if kind == 'frequency':
            frequency = crval * unit.to(u.Hz)
            value = _SPEED_OF_LIGHT_MS * (1.0 - frequency / rest)  # radio velocity
        else:
            beta = crval * unit.to(u.m / u.s) / _SPEED_OF_LIGHT_MS
            if ctype.startswith('VRAD'):
                value = rest * (1.0 - beta)
            elif ctype.startswith('VOPT'):
                value = rest / (1.0 + beta)
            else:
                return
        pixel = float(header.get(f'CRPIX{n}', 1.0))
    except (TypeError, ValueError, u.UnitsError):
        return
    if not math.isfinite(value):
        return
    header['ALTRVAL'] = value
    header['ALTRPIX'] = pixel


def write_radio_velocity_axis(header, axis_number: int, rest_hz, unit: str = 'km/s') -> bool:
    """Turn frequency axis n of a header into radio velocity in ``unit`` ('m/s' or 'km/s').

    v = c (1 - f / f0) is linear in f, so CRVAL and the step (CDELT, or the CD
    row) map exactly.  The alternate reference follows
    (:func:`refresh_alternate_reference`).  Returns False, and writes nothing,
    without a rest frequency or a frequency unit.
    """
    rest = _positive_float(rest_hz)
    target = canonical_velocity_unit(unit)
    n = int(axis_number)
    if rest is None or target is None:
        return False
    try:
        to_hz = u.Unit(str(header.get(f'CUNIT{n}', '') or 'Hz').strip()).to(u.Hz)
        crval_hz = float(header.get(f'CRVAL{n}', 0.0)) * to_hz
    except (TypeError, ValueError, u.UnitsError):
        return False
    step = header_axis_step(header, n)
    step_hz = (1.0 if step is None else float(step)) * to_hz  # FITS default CDELT 1
    per_ms = 1e-3 if target == 'km/s' else 1.0
    header[f'CRVAL{n}'] = _SPEED_OF_LIGHT_MS * (1.0 - crval_hz / rest) * per_ms
    set_header_axis_step(header, n, -_SPEED_OF_LIGHT_MS * step_hz / rest * per_ms)
    header[f'CTYPE{n}'] = 'VRAD'
    header[f'CUNIT{n}'] = target
    refresh_alternate_reference(header, n, rest)
    return True


def rest_frequency_intents(wcs, header=None, spectral_metadata=None) -> Dict[str, Optional[str]]:
    """Which rest-frequency intents apply: ``{intent: None if allowed, else why not}``.

    'rereference' needs a radio (VRAD) or optical (VOPT) velocity axis, on
    which the new velocities are a linear map of the old ones, and a current
    rest frequency.  'metadata' is refused for a velocity axis that the loader
    derived from the observed frequencies: keeping it under another rest
    frequency would change the frequencies it stands for.
    """
    meta = spectral_metadata or {}
    reasons: Dict[str, Optional[str]] = {intent: None for intent in REST_FREQUENCY_INTENTS}
    axis = spectral_wcs_axis(wcs, meta) if wcs is not None else None
    if axis is None:
        reasons['rereference'] = 'The data have no spectral axis to re-reference.'
    else:
        ctype = str(wcs.wcs.ctype[axis]).strip()
        convention = _velocity_convention(wcs, axis)
        if convention is None:
            kind = _NON_VELOCITY_KINDS.get(ctype.upper()[:4])
            if kind is not None:
                reasons['rereference'] = f"The spectral axis is {kind} ({ctype}); a rest frequency does not move it."
            else:
                reasons['rereference'] = f"The velocity convention of CTYPE '{ctype or '(none)'}' is not known."
        elif convention == 'relativistic':
            reasons['rereference'] = 'Relativistic velocities (VELO) do not re-reference linearly.'
        elif rest_frequency_hz(wcs, header) is None:
            reasons['rereference'] = 'There is no rest frequency to re-reference from.'
    if meta.get('converted_from_frequency'):
        reasons['metadata'] = (
            'The velocity axis was derived from the observed frequencies on load; '
            'keeping it under another rest frequency would change them.'
        )
    return reasons


def _header_step_keys(header, axis_number: int) -> List[str]:
    """The header keys that hold the step of FITS axis n (as :func:`set_header_axis_step` writes it)."""
    row_keys = _header_cd_row_keys(header, axis_number)
    return row_keys if row_keys and header_axis_step(header, axis_number) else [f'CDELT{int(axis_number)}']


def apply_rest_frequency(wcs, header, spectral_metadata, rest_hz, intent) -> Dict[str, Any]:
    """Change the rest frequency to ``rest_hz`` (Hz) with an explicit ``intent``.

    'rereference' keeps the observed frequencies.  Radio velocities map as
    v' = c(1 - B) + B v with B = f_old / f_new, optical ones as
    v' = c(B - 1) + B v with B = f_new / f_old.  Both are exact per pixel, so
    CRVAL and the step (CDELT, the CD row, or CDELT next to a scaled PC row)
    take the same map.  'metadata' keeps the velocity axis.

    Mutates, in step with each other: ``wcs`` (SI numbers, RESTFRQ), the
    ``header`` (display-unit numbers; RESTFRQ, and RESTFREQ / RESTWAV where
    present) and ``spectral_metadata['restfreq_hz']``.  Returns
    ``{'previous_hz', 'rest_hz', 'intent', 'scale', 'header_keys'}``.  Raises
    ValueError when the intent does not apply (:func:`rest_frequency_intents`).
    """
    if intent not in REST_FREQUENCY_INTENTS:
        raise ValueError(f"Unknown rest-frequency intent {intent!r}; use one of {REST_FREQUENCY_INTENTS}.")
    new_hz = _positive_float(rest_hz)
    if new_hz is None:
        raise ValueError('The rest frequency must be a positive number of Hz.')
    if wcs is None or header is None:
        raise ValueError('A rest-frequency change needs the header and the WCS.')
    meta = spectral_metadata if isinstance(spectral_metadata, dict) else {}
    refusal = rest_frequency_intents(wcs, header, meta)[intent]
    if refusal:
        raise ValueError(refusal)

    previous_hz = rest_frequency_hz(wcs, header)
    header_keys: List[str] = []
    scale = 1.0
    if intent == 'rereference' and new_hz != previous_hz:
        axis = spectral_wcs_axis(wcs, meta)
        radio = _velocity_convention(wcs, axis) == 'radio'
        scale = previous_hz / new_hz if radio else new_hz / previous_hz
        offset_ms = _SPEED_OF_LIGHT_MS * ((1.0 - scale) if radio else (scale - 1.0))

        axis_number = axis + 1
        unit_text = str(header.get(f'CUNIT{axis_number}', '') or '').strip() or display_unit(meta)
        try:
            offset_header = (offset_ms * u.m / u.s).to(u.Unit(unit_text)).value
        except Exception as exc:
            raise ValueError(f"CUNIT{axis_number} {unit_text!r} is not a velocity unit.") from exc

        crval = np.array(wcs.wcs.crval, dtype=float)
        crval[axis] = offset_ms + scale * crval[axis]
        wcs.wcs.crval = crval
        _scale_axis_step(wcs, axis, scale)

        header[f'CRVAL{axis_number}'] = offset_header + scale * float(header.get(f'CRVAL{axis_number}', 0.0))
        step = header_axis_step(header, axis_number)
        set_header_axis_step(header, axis_number, scale * (1.0 if step is None else step))
        header_keys += [f'CRVAL{axis_number}'] + _header_step_keys(header, axis_number)

    wcs.wcs.restfrq = new_hz
    if _positive_float(getattr(wcs.wcs, 'restwav', 0.0)) is not None:
        wcs.wcs.restwav = _SPEED_OF_LIGHT_MS / new_hz
    wcs.wcs.set()

    rest_keys = [key for key in _REST_FREQUENCY_KEYS if key in header] or ['RESTFRQ']
    for key in rest_keys:
        header[key] = new_hz
    if 'RESTWAV' in header:
        header['RESTWAV'] = _SPEED_OF_LIGHT_MS / new_hz
        rest_keys.append('RESTWAV')
    meta['restfreq_hz'] = new_hz
    if axis_number_of(wcs, meta) is not None:
        # the alternate reference is a frequency or a velocity, both follow
        refresh_alternate_reference(header, axis_number_of(wcs, meta), new_hz)
    return {
        'previous_hz': previous_hz,
        'rest_hz': new_hz,
        'intent': intent,
        'scale': scale,
        'header_keys': rest_keys + header_keys,
    }


def axis_number_of(wcs, spectral_metadata=None) -> Optional[int]:
    """The FITS axis number (1-based) of the spectral axis, or None."""
    axis = spectral_wcs_axis(wcs, spectral_metadata) if wcs is not None else None
    return None if axis is None else axis + 1


# --- Systemic redshift (TF-415 slice B) ---
#
# A systemic redshift z measures velocities from the effective rest frequency
# f0 / (1 + z) of the line whose own rest frequency is f0.  The change goes
# through apply_rest_frequency, so the observed frequencies stay and nothing
# is resampled; a line at z + dz then lies at about c dz / (1 + z).  The
# header holds RESTFRQ = f0 / (1 + z), which any reader combines with the
# axis correctly, and ZSOURCE = z.  The comment of that ZSOURCE card tells
# takefits, on reopening, that RESTFRQ is the effective rest frequency; a
# ZSOURCE written by other software is not applied (see
# systemic_redshift_from_header).

SYSTEMIC_Z_KEY = 'ZSOURCE'
_SYSTEMIC_Z_COMMENT = 'systemic z (takefits: RESTFRQ = line rest / (1+z))'
_EFFECTIVE_REST_COMMENT = 'line rest frequency / (1 + ZSOURCE) [Hz]'
_PLAIN_REST_COMMENT = 'Rest frequency [Hz]'


def _redshift(value) -> float:
    try:
        z = float(value)
    except (TypeError, ValueError):
        z = float('nan')
    if not (math.isfinite(z) and z > -1.0):
        raise ValueError('The systemic redshift z must be a number greater than -1.')
    return z


def systemic_redshift(spectral_metadata) -> float:
    """The systemic redshift z the velocities are measured from (0 when none is set)."""
    try:
        return _redshift((spectral_metadata or {}).get('systemic_z') or 0.0)
    except ValueError:
        return 0.0


def line_rest_frequency_hz(wcs=None, header=None, spectral_metadata=None) -> Optional[float]:
    """The line's own rest frequency f0 in Hz (the effective one times 1 + z), or None."""
    meta = spectral_metadata or {}
    line = _positive_float(meta.get('line_restfreq_hz'))
    if line is not None:
        return line
    rest = rest_frequency_hz(wcs, header)
    return None if rest is None else rest * (1.0 + systemic_redshift(meta))


def systemic_redshift_from_header(header) -> Tuple[float, Optional[str]]:
    """(z, notice) for a freshly loaded header.

    A header takefits wrote with a systemic redshift has ZSOURCE with
    takefits' comment, and RESTFRQ is then f0 / (1 + z).  A ZSOURCE from other
    software gives z = 0, RESTFRQ taken as the line's own rest frequency, and
    a notice saying how to use it.
    """
    if header is None or SYSTEMIC_Z_KEY not in header:
        return 0.0, None
    try:
        z = _redshift(header[SYSTEMIC_Z_KEY])
    except ValueError:
        return 0.0, None
    if 'takefits' in str(header.comments[SYSTEMIC_Z_KEY]):
        return z, None
    return 0.0, (
        f"ZSOURCE = {z:g} is in the header; velocities are measured from RESTFRQ as it is. "
        "To measure them from the systemic redshift, set z in Unit Conversion."
    )


def _write_systemic_redshift(header, z: float, foreign_zsource=None) -> None:
    rest_keys = [key for key in _REST_FREQUENCY_KEYS if key in header]
    if z:
        header[SYSTEMIC_Z_KEY] = (float(z), _SYSTEMIC_Z_COMMENT)
        for key in rest_keys:
            header.comments[key] = _EFFECTIVE_REST_COMMENT
        return
    if foreign_zsource is not None:
        header[SYSTEMIC_Z_KEY] = foreign_zsource  # the file's own value, left as it came
    elif SYSTEMIC_Z_KEY in header:
        del header[SYSTEMIC_Z_KEY]
    for key in rest_keys:
        if header.comments[key] == _EFFECTIVE_REST_COMMENT:
            header.comments[key] = _PLAIN_REST_COMMENT


def apply_spectral_axis(wcs, header, spectral_metadata, *, restfreq_hz=None, z=None, intent) -> Dict[str, Any]:
    """Set the line's rest frequency and/or the systemic redshift z (TF-415 slice B).

    The axis uses the effective rest frequency f0 / (1 + z), through
    :func:`apply_rest_frequency` with ``intent``: 'rereference' keeps the
    observed frequencies (radio velocities for a frequency cube, the axis's
    own convention for a velocity cube), 'metadata' keeps the axis.  An
    argument left out keeps its current value; z = 0 removes the systemic
    redshift.

    Mutates the WCS, the header (RESTFRQ = f0 / (1 + z), ZSOURCE) and
    ``spectral_metadata`` ('restfreq_hz', 'line_restfreq_hz', 'systemic_z').
    Returns apply_rest_frequency's summary plus 'line_rest_hz', 'z',
    'previous_line_rest_hz' and 'previous_z'.
    """
    if restfreq_hz is None and z is None:
        raise ValueError('Give the rest frequency of the line, the systemic redshift z, or both.')
    meta = spectral_metadata if isinstance(spectral_metadata, dict) else {}
    previous_z = systemic_redshift(meta)
    previous_line = line_rest_frequency_hz(wcs, header, meta)
    new_z = previous_z if z is None else _redshift(z)
    if restfreq_hz is None:
        new_line = previous_line
        if new_line is None:
            raise ValueError('The cube has no rest frequency yet: give the rest frequency of the line too.')
    else:
        new_line = _positive_float(restfreq_hz)
        if new_line is None:
            raise ValueError('The rest frequency must be a positive number of Hz.')
    summary = apply_rest_frequency(wcs, header, meta, new_line / (1.0 + new_z), intent)
    meta['line_restfreq_hz'] = new_line
    meta['systemic_z'] = new_z
    _write_systemic_redshift(header, new_z, meta.get('zsource_header'))
    if new_z or previous_z:
        summary['header_keys'].append(SYSTEMIC_Z_KEY)
    summary.update(
        {'line_rest_hz': new_line, 'previous_line_rest_hz': previous_line, 'z': new_z, 'previous_z': previous_z}
    )
    return summary


# --- The unit of spectral numbers in written files (TF-415 slice A, D2) ---
#
# Range files, workspaces, recipes and histories, marker, region and PV-path
# files store spectral world numbers in the display unit of the cube they were
# written from.  They record that unit (SPECTRAL_UNIT_KEY), and a reader turns
# the numbers into the display unit of the cube it applies them to.  A file
# without the record is read in the units takefits wrote before: velocity
# numbers in the display unit (as now), frequency and wavelength numbers in SI.

SPECTRAL_UNIT_KEY = 'spectral_unit'


def spectral_unit_tag(wcs, spectral_metadata=None) -> str:
    """The unit to record next to spectral world numbers taken from the display."""
    return display_numbers_unit(wcs, spectral_metadata) if wcs is not None else display_unit(spectral_metadata)


def _spectral_axis_kind(wcs, spectral_metadata=None) -> str:
    kind = str((spectral_metadata or {}).get('current_axis_type') or '').strip().lower()
    if kind in ('frequency', 'velocity', 'wavelength'):
        return kind
    axis = spectral_wcs_axis(wcs, spectral_metadata) if wcs is not None else None
    if axis is None:
        return 'unknown'
    return classify_axis_type(str(wcs.wcs.ctype[axis]))


def legacy_spectral_unit(wcs, spectral_metadata=None) -> str:
    """The unit of spectral numbers in a file written before files recorded it.

    The display did not scale frequency and wavelength axes before slice A,
    so their numbers are SI; velocity numbers were in the display unit.
    """
    kind = _spectral_axis_kind(wcs, spectral_metadata)
    if kind == 'frequency':
        return 'Hz'
    if kind == 'wavelength':
        return 'm'
    return spectral_unit_tag(wcs, spectral_metadata)


def stored_spectral_factor(stored_unit, wcs, spectral_metadata=None) -> Optional[float]:
    """Factor from spectral numbers stored in ``stored_unit`` to this cube's display numbers.

    ``stored_unit`` None or '' means a file without the record
    (:func:`legacy_spectral_unit`).  Returns None when the two units are of
    different kinds (km/s numbers for a GHz axis), which callers treat as
    incompatible.
    """
    current = spectral_unit_tag(wcs, spectral_metadata)
    source = str(stored_unit or '').strip() or legacy_spectral_unit(wcs, spectral_metadata)
    if not current or not source or source == current:
        return 1.0
    try:
        first, second = u.Unit(source), u.Unit(current)
    except Exception:
        return None
    if first == u.dimensionless_unscaled or second == u.dimensionless_unscaled:
        return 1.0 if first == second else None
    if not first.is_equivalent(second):
        return None
    return float(first.to(second))


def scale_spectral_value(value, factor: float):
    """``value`` times ``factor``: a number, a numeric string, or a list or tuple of them.

    Powers of ten move the decimal point exactly.  Strings that are not
    numbers (empty fields, sexagesimal text) and None come back unchanged.
    """
    if factor == 1.0 or value is None or isinstance(value, bool):
        return value
    scale = _decimal_scaler(factor)
    if isinstance(value, (list, tuple)):
        return type(value)(scale_spectral_value(item, factor) for item in value)
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return value
        if not math.isfinite(number):
            return value
        return f"{scale(number):.12g}"
    try:
        return float(scale(float(value)))
    except (TypeError, ValueError):
        return value
