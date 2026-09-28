"""Headless FITS I/O helpers (no PyQt dependencies)."""
from __future__ import annotations

import operator
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.io.fits import VerifyError
from astropy.wcs import WCS

from takefits.core.spectral_units import (
    apply_load_convention,
    classify_axis_type,
    header_axis_step,
    identify_spectral_axis,
    normalize_velocity_cunits,
    refresh_alternate_reference,
    set_header_axis_step,
    systemic_redshift_from_header,
)
from takefits.logic.freq_to_velocity import FreqToVelocity, RadioVelocityToFrequency
from takefits.logic.data_tools import (
    LAZY_SCALING_THRESHOLD_BYTES,
    MEMMAP_THRESHOLD_BYTES,
    LazyScaledArray,
    _parse_header_float,
    build_large_data_profile,
    estimate_array_nbytes,
    header_has_scaling_keywords,
    is_lazy_scaled,
)


# Preserve the historical load-time normalization boundary while keeping the
# scan itself chunked. Arrays above the memmap boundary stay lazy and are
# sanitized per slice so opening a large FITS does not page the whole file in.
_FULL_DATA_SANITIZE_MAX_BYTES = MEMMAP_THRESHOLD_BYTES
_SANITIZE_CHUNK_TARGET_BYTES = 16 * 1024 * 1024

_KNOWN_CTYPE_FIXES = {
    "GLON---TAN": "GLON-TAN",
    "GLAT--TAN": "GLAT-TAN",
}


def _normalize_known_ctype_typos(header) -> None:
    """Repair known, unambiguous celestial CTYPE separator typos in memory."""
    try:
        naxis = int(header.get("NAXIS", 0))
    except (AttributeError, TypeError, ValueError):
        return

    for axis in range(1, naxis + 1):
        key = f"CTYPE{axis}"
        original = str(header.get(key, "")).strip().upper()
        replacement = _KNOWN_CTYPE_FIXES.get(original)
        if replacement is None:
            continue
        print(f"\033[93mWarning: Corrected {key}: {original} ==> {replacement}.\033[0m")
        header[key] = replacement


def _selected_image_hdu_index(hdul, requested_hdu: int | None = None) -> int | None:
    """Return the requested or first HDU describing a non-empty image."""
    if requested_hdu is None:
        candidates = enumerate(hdul)
    else:
        index = requested_hdu if requested_hdu >= 0 else len(hdul) + requested_hdu
        if index < 0 or index >= len(hdul):
            return None
        candidates = ((index, hdul[index]),)

    for index, hdu in candidates:
        if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
            continue
        header = getattr(hdu, "header", None)
        if header is None:
            continue
        try:
            naxis = int(header.get("NAXIS", 0))
            if naxis <= 0:
                continue
            if all(int(header.get(f"NAXIS{axis}", 0)) > 0 for axis in range(1, naxis + 1)):
                return index
        except (TypeError, ValueError):
            continue
    return None


def _header_array_sizes(header) -> tuple[int, int]:
    """Return raw storage and projected float64 scaling sizes from a header."""
    try:
        naxis = int(header.get("NAXIS", 0))
        count = 1
        for axis in range(1, naxis + 1):
            count *= max(0, int(header.get(f"NAXIS{axis}", 0)))
        raw_itemsize = max(1, abs(int(header.get("BITPIX", 8))) // 8)
    except (TypeError, ValueError):
        return 0, 0
    raw_bytes = count * raw_itemsize
    projected_bytes = (
        count * np.dtype(np.float64).itemsize
        if header_has_scaling_keywords(header)
        else raw_bytes
    )
    return int(raw_bytes), int(projected_bytes)


def _wrap_lazy_scaled_data(data, header):
    """Wrap one selected raw image HDU with its own FITS scaling metadata."""
    if data is None or is_lazy_scaled(data):
        return data
    bzero = _parse_header_float(header, "BZERO", 0.0)
    bscale = _parse_header_float(header, "BSCALE", 1.0)
    bzero = 0.0 if bzero is None else bzero
    bscale = 1.0 if bscale is None else bscale
    blank = None
    if "BLANK" in header:
        try:
            blank = int(header["BLANK"])
        except (TypeError, ValueError):
            blank = None
    return LazyScaledArray(data, bzero=bzero, bscale=bscale, blank=blank)


def _iter_buffered_chunks(array, *, readwrite: bool):
    """Yield bounded 1-D buffers over an arbitrary-strided ndarray."""
    itemsize = max(1, int(np.dtype(array.dtype).itemsize))
    buffer_items = max(1, _SANITIZE_CHUNK_TARGET_BYTES // itemsize)
    op_flags = ["readwrite"] if readwrite else ["readonly"]
    with np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=op_flags,
        order="K",
        buffersize=buffer_items,
    ) as iterator:
        for chunk in iterator:
            yield chunk


def _chunk_has_invalid_values(data) -> bool:
    """Scan for Takefits sentinels/non-finite values with bounded masks."""
    integer = np.issubdtype(data.dtype, np.integer)
    for chunk in _iter_buffered_chunks(data, readwrite=False):
        with np.errstate(invalid="ignore"):
            bad = chunk < -100000
            if not integer:
                bad |= ~np.isfinite(chunk)
        if np.any(bad):
            return True
    return False


def _sanitize_loaded_data(data):
    """Sanitize a modest array in bounded chunks, copying only when required."""
    if data is None or getattr(data, "size", 0) == 0:
        return data, False
    if not _chunk_has_invalid_values(data):
        return data, False

    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    elif not data.flags.writeable:
        data = np.array(data, copy=True)

    for chunk in _iter_buffered_chunks(data, readwrite=True):
        with np.errstate(invalid="ignore"):
            bad = (chunk < -100000) | ~np.isfinite(chunk)
        if np.any(bad):
            chunk[bad] = np.nan
    return data, True


def _slice_singleton_axis(data, axis):
    """Return a view of ``data`` with FITS axis ``axis`` (1-based) removed if length 1."""

    if data is None or getattr(data, 'ndim', 0) == 0:
        return data

    data_axis = data.ndim - axis
    if data_axis < 0 or data_axis >= data.ndim:
        return data

    if data.shape[data_axis] != 1:
        return data

    indexer = [slice(None)] * data.ndim
    indexer[data_axis] = 0

    if is_lazy_scaled(data):
        return data._raw_view_op(lambda a: a[tuple(indexer)])
    return data[tuple(indexer)]


def _remove_axis_metadata(header, axis, max_axes):
    """Strip header keywords associated with FITS axis ``axis``."""

    if header is None:
        return

    simple_prefixes = (
        'NAXIS',
        'CDELT',
        'CRPIX',
        'CRVAL',
        'CTYPE',
        'CUNIT',
        'CROTA',
        'CNAME',
        'CRDER',
        'CSYER',
    )

    for prefix in simple_prefixes:
        key = f"{prefix}{axis}"
        if key in header:
            del header[key]

    matrix_prefixes = ('PC', 'CD', 'PV', 'PS', 'PT')
    for prefix in matrix_prefixes:
        for idx in range(1, max_axes + 1):
            key = f"{prefix}{idx}_{axis}"
            if key in header:
                del header[key]
            key = f"{prefix}{axis}_{idx}"
            if key in header:
                del header[key]

    if 'WCSAXES' in header and header['WCSAXES'] >= axis:
        header['WCSAXES'] = max(int(header['WCSAXES']) - 1, 0)


def _collapse_singleton_axes(data, header):
    """Remove trailing FITS axes of length 1 (e.g. redundant Stokes axis)."""

    collapsed_axes = []
    while True:
        naxis = int(header.get('NAXIS', getattr(data, 'ndim', 0) if data is not None else 0))
        if naxis <= 2 or data is None:
            break

        collapsed = False
        for axis in range(naxis, 2, -1):
            size_key = f'NAXIS{axis}'
            if header.get(size_key) != 1:
                continue

            new_data = _slice_singleton_axis(data, axis)
            if new_data is data:
                continue

            _remove_axis_metadata(header, axis, max_axes=naxis)
            header['NAXIS'] = naxis - 1
            data = new_data
            collapsed_axes.append(axis)
            collapsed = True
            break

        if not collapsed:
            break

    if collapsed_axes:
        collapsed_axes.sort()
        axes_str = ', '.join(str(ax) for ax in collapsed_axes)
        print(
            f"\033[1;33m\033[1mWarning: Dropped singleton FITS axis/axes {axes_str}. "
            "WCS adjusted to match data.\033[0m"
        )

    return data


# The loader's names for the shared helpers.
_classify_axis_type = classify_axis_type
_identify_spectral_axis = identify_spectral_axis


class _KeptFrequencyAxis:
    """Stands in for FreqToVelocity when the frequency axis is kept."""

    converted = False
    to_frequency = False


def _frequency_converter(header, frequency_axis: str, spectral_axis_index):
    """The converter for the spectral axis: to velocity (the default), kept, or to frequency.

    ``frequency_axis='frequency'`` keeps a frequency axis and turns a radio
    velocity axis into frequency; other velocity axes open in velocity, with
    a notice saying why.
    """
    kind = _classify_axis_type(header.get(f'CTYPE{spectral_axis_index}', '')) if spectral_axis_index else 'unknown'
    if frequency_axis == "frequency":
        if kind == 'frequency':
            print("\033[96mThe frequency axis stays in frequency (spectral axis: freq).\033[0m")
        elif kind == 'velocity':
            converter = RadioVelocityToFrequency(header, spectral_axis_index)
            if converter.to_frequency:
                print("\033[96mThe radio-velocity axis opens in frequency, f = f0 (1 - v/c) "
                      "(spectral axis: freq).\033[0m")
                return converter
            print(f"\033[93mThe spectral axis opens in velocity: {converter.reason}.\033[0m")
            if "rest frequency" in str(converter.reason):
                print("\033[96m  To show frequencies, set RestFreq in Unit Conversion, "
                      "save the cube and reopen it.\033[0m")
        return _KeptFrequencyAxis()
    return FreqToVelocity(header)


def _spectral_axis_mode(spectral_metadata) -> dict:
    """The mode that applied to a frequency axis, for source records; {} for other axes."""
    if spectral_metadata.get('converted_from_frequency'):
        return {"frequency_axis": "velocity"}
    if spectral_metadata.get('current_axis_type') == 'frequency':
        return {"frequency_axis": "frequency"}
    return {}


def _get_restfreq_hz(header):
    """Return RESTFRQ/RESTFREQ in Hz if available."""
    if header is None:
        return None
    restfreq = header.get('RESTFRQ', header.get('RESTFREQ'))
    if restfreq is None:
        return None
    try:
        return float(restfreq)
    except (TypeError, ValueError):
        return None


def _ensure_velocity_axis_ascending(data, header, fits_axis):
    """Flip data/header so a converted velocity axis increases with pixel index.

    Returns a tuple of (possibly flipped data, was_flipped).
    """

    if fits_axis is None or data is None or header is None:
        return data, False

    cdelt = header_axis_step(header, fits_axis)  # CDELT or CD matrix
    if cdelt is None or cdelt >= 0:
        return data, False

    naxis_key = f'NAXIS{fits_axis}'
    axis_length = header.get(naxis_key)
    data_axis = getattr(data, 'ndim', 0) - fits_axis
    if axis_length in (None, 0) and 0 <= data_axis < getattr(data, 'ndim', 0):
        axis_length = data.shape[data_axis]

    if axis_length in (None, 0) or axis_length == 1:
        set_header_axis_step(header, fits_axis, abs(cdelt))
        return data, False

    if data_axis < 0 or data_axis >= getattr(data, 'ndim', 0):
        set_header_axis_step(header, fits_axis, abs(cdelt))
        return data, False

    if is_lazy_scaled(data):
        data = data._raw_view_op(np.flip, axis=data_axis)
    else:
        data = np.flip(data, axis=data_axis)

    set_header_axis_step(header, fits_axis, abs(cdelt))

    crpix_key = f'CRPIX{fits_axis}'
    if crpix_key in header:
        try:
            crpix = float(header[crpix_key])
        except (TypeError, ValueError):
            crpix = (axis_length + 1) / 2.0
        header[crpix_key] = axis_length + 1 - crpix
    else:
        header[crpix_key] = (axis_length + 1) / 2.0

    print(
        "\033[93mVelocity axis flipped so radial velocity increases with pixel index "
        f"(FITS axis {fits_axis}).\033[0m"
    )

    return data, True


class FITSLoadError(Exception):
    """Raised when a FITS file cannot be loaded."""

    def __init__(self, message, filename=None, kind="generic", detail=None):
        self.filename = str(filename) if filename is not None else None
        self.kind = kind
        self.detail = detail
        super().__init__(message)

FREQUENCY_AXIS_MODES = ("velocity", "frequency")


def load_fits(filename, compute_wcs=True, hdu: int | None = None, frequency_axis: str = "velocity"):
    """
    Load a FITS file and process header/data.
    Optionally compute WCS and perform velocity unit conversion.

    ``frequency_axis`` says whether a radio cube shows velocity or frequency:
    'velocity' converts a frequency axis to radio velocity when the header
    gives a rest frequency (the default, as always); 'frequency' keeps a
    frequency axis and converts a radio velocity axis (VRAD) to frequency,
    with the same rest frequency.  The mode that applied is in
    ``spectral_metadata['spectral_axis_mode']`` (TF-415 slice A).
    """
    if frequency_axis not in FREQUENCY_AXIS_MODES:
        raise FITSLoadError(
            f"frequency_axis must be one of {FREQUENCY_AXIS_MODES}",
            filename=filename,
            kind="invalid_spectral_axis",
        )
    path = Path(filename)

    if not path.exists():
        raise FITSLoadError("File not found", filename=path, kind="not_found")

    if not path.is_file():
        raise FITSLoadError("Path is not a file", filename=path, kind="not_file")

    requested_hdu = None
    if hdu is not None:
        try:
            requested_hdu = operator.index(hdu)
        except TypeError as err:
            raise FITSLoadError(
                "HDU index must be an integer",
                filename=path,
                kind="invalid_hdu",
                detail=str(err),
            ) from err

    open_kwargs = dict(
        mode='readonly',
        ignore_missing_end=True,
        ignore_missing_simple=True,
        memmap=True,
        lazy_load=True,
    )
    lazy_scaling_active = False

    def _open_fits(**kw):
        return fits.open(path, **kw)

    # Probe once a file could expand past the lazy threshold in the worst case
    # (int8 -> float64 is 8x). This closes the former 1–2 GiB gap where scaled
    # files skipped lazy scaling and were eagerly expanded to several GiB.
    _probe_materialized_bytes = 0
    _probe_needs_scaling = False
    _probe_hdu_index = None
    _file_size_on_disk = 0
    try:
        _file_size_on_disk = path.stat().st_size
    except Exception:
        pass

    probe_threshold = max(1, LAZY_SCALING_THRESHOLD_BYTES // 8)
    if _file_size_on_disk >= probe_threshold:
        # Probe only the image HDU the loader will actually return. Scaling
        # keywords in an unrelated extension must not change the open strategy.
        try:
            with fits.open(path, mode='readonly', memmap=True, lazy_load=True,
                           ignore_missing_end=True, ignore_missing_simple=True) as probe:
                _probe_hdu_index = _selected_image_hdu_index(probe, requested_hdu)
                if _probe_hdu_index is not None:
                    h = probe[_probe_hdu_index].header
                    _probe_needs_scaling = header_has_scaling_keywords(h)
                    _, _probe_materialized_bytes = _header_array_sizes(h)
        except Exception:
            pass

        # Decide open strategy up-front to avoid close-and-reopen cycles.
        if (
            _probe_needs_scaling
            and _probe_materialized_bytes >= LAZY_SCALING_THRESHOLD_BYTES
        ):
            open_kwargs["do_not_scale_image_data"] = True
            lazy_scaling_active = True

    try:
        hdulist = _open_fits(**open_kwargs)
    except (OSError, FileNotFoundError, VerifyError) as err:
        raise FITSLoadError(
            "Failed to open FITS file",
            filename=path,
            kind="open_error",
            detail=str(err),
        ) from err
    except Exception as err:
        message = str(err)
        if "Cannot load a memory-mapped image" in message:
            # Fallback: astropy cannot memmap scaled data despite our probe.
            if (
                not lazy_scaling_active
                and _probe_needs_scaling
                and _probe_materialized_bytes >= LAZY_SCALING_THRESHOLD_BYTES
            ):
                open_kwargs["do_not_scale_image_data"] = True
                lazy_scaling_active = True
            else:
                open_kwargs["memmap"] = False
                open_kwargs["lazy_load"] = False
            try:
                hdulist = _open_fits(**open_kwargs)
            except Exception as retry_err:
                raise FITSLoadError(
                    "Unexpected error",
                    filename=path,
                    kind="unexpected",
                    detail=str(retry_err),
                ) from retry_err
        else:
            raise FITSLoadError(
                "Unexpected error",
                filename=path,
                kind="unexpected",
                detail=message,
            ) from err
    else:
        # Reactive fallback for files where no probe was done or the probe
        # failed. Inspect only the selected image HDU.
        if not lazy_scaling_active and open_kwargs.get("memmap", True):
            selected_index = _selected_image_hdu_index(hdulist, requested_hdu)
            selected_header = (
                hdulist[selected_index].header
                if selected_index is not None
                else None
            )
            requires_scaling = header_has_scaling_keywords(selected_header)
            if requires_scaling:
                hdulist.close()
                _, projected_bytes = _header_array_sizes(selected_header)
                if projected_bytes >= LAZY_SCALING_THRESHOLD_BYTES:
                    open_kwargs["do_not_scale_image_data"] = True
                    lazy_scaling_active = True
                else:
                    open_kwargs["memmap"] = False
                    open_kwargs["lazy_load"] = False
                try:
                    hdulist = _open_fits(**open_kwargs)
                except Exception as retry_err:
                    raise FITSLoadError(
                        "Unexpected error",
                        filename=path,
                        kind="unexpected",
                        detail=str(retry_err),
                    ) from retry_err

    if lazy_scaling_active:
        print(
            "\033[96mLazy scaling: keeping memory-mapped I/O for "
            "large scaled FITS data.\033[0m"
        )

    with hdulist as hdul:
        selected_hdu_index = _selected_image_hdu_index(hdul, requested_hdu)
        if selected_hdu_index is None:
            message = (
                f"Requested HDU {requested_hdu} does not contain non-empty image data"
                if requested_hdu is not None
                else "FITS file contains no non-empty image HDU"
            )
            raise FITSLoadError(
                message,
                filename=path,
                kind="invalid_hdu" if requested_hdu is not None else "no_image",
            )
        selected_hdu = hdul[selected_hdu_index]
        data = selected_hdu.data
        header = selected_hdu.header
        if selected_hdu_index != 0 and requested_hdu is None:
            print(
                "\033[1;33m\033[1mWarning: Primary HDU has no image data. "
                f"Using image HDU {selected_hdu_index}.\033[0m"
            )

        _normalize_known_ctype_typos(header)

        # Wrap raw memmap in LazyScaledArray when lazy scaling is active.
        if lazy_scaling_active and data is not None:
            data = _wrap_lazy_scaled_data(data, header)
            print(
                f"\033[96m  BZERO={data._bzero}, BSCALE={data._bscale}"
                + (f", BLANK={data._blank}" if data._blank is not None else "")
                + "\033[0m"
            )

        data_nbytes = estimate_array_nbytes(data)
        spectral_axis_index = _identify_spectral_axis(header)
        original_axis_ctype = header.get(f'CTYPE{spectral_axis_index}', '') if spectral_axis_index else ''
        original_axis_unit = header.get(f'CUNIT{spectral_axis_index}', '').strip() if spectral_axis_index else None
        spectral_metadata = {
            'selected_hdu_index': int(selected_hdu_index),
            'axis_index': spectral_axis_index,
            'original_axis_ctype': original_axis_ctype,
            'original_axis_type': _classify_axis_type(original_axis_ctype),
            'original_axis_unit': original_axis_unit,
            'current_axis_ctype': None,
            'current_axis_type': None,
            'current_axis_unit': None,
            'converted_from_frequency': False,
            'converted_from_velocity': False,
            'frequency_unit_original': None,
            'velocity_unit_adjusted': False,
            'velocity_unit_original': None,
            'velocity_unit_target': None,
            'restfreq_original_hz': _get_restfreq_hz(header),
            'restfreq_hz': None,
            'axis_flipped': False,
            'is_cartesian_interpretation': False,
        }
        spectral_metadata['restfreq_hz'] = spectral_metadata['restfreq_original_hz']

        if data_nbytes and data_nbytes >= MEMMAP_THRESHOLD_BYTES:
            approx_gib = data_nbytes / (1024 ** 3)
            print(
                f"\033[93mDetected large FITS data cube (~{approx_gib:.2f} GiB). "
                "Using memory-mapped lazy loading.\033[0m"
            )

        # Full-cube masks caused several hundred MiB of transient allocations.
        # Preserve the historical below-memmap sanitization behavior, but scan
        # and replace through bounded iterator buffers. Lazy/scaled arrays and
        # arrays above the memmap boundary are normalized per slice instead.
        if is_lazy_scaled(data):
            spectral_metadata['_needs_per_slice_sanitize'] = True
        elif (
            data_nbytes is None
            or data_nbytes <= _FULL_DATA_SANITIZE_MAX_BYTES
        ):
            data, replaced_invalid = _sanitize_loaded_data(data)
            if replaced_invalid:
                print(
                    "\033[1;33m\033[1mWarning: Replaced invalid or infinite "
                    "data values with NaN.\033[0m"
                )
        else:
            spectral_metadata['_needs_per_slice_sanitize'] = True

        # Frequency conversion using FreqToVelocity
        converter = _frequency_converter(header, frequency_axis, spectral_axis_index)
        if converter.converted:
            header = converter.header
            data, flipped = _ensure_velocity_axis_ascending(data, header, converter.freq_axis)
            spectral_metadata['axis_flipped'] = flipped
            spectral_metadata['converted_from_frequency'] = True
            spectral_metadata['axis_index'] = converter.freq_axis
            spectral_metadata['original_axis_ctype'] = converter.original_axis_type or spectral_metadata['original_axis_ctype']
            spectral_metadata['original_axis_type'] = _classify_axis_type(converter.original_axis_type)
            spectral_metadata['original_axis_unit'] = converter.original_axis_unit or spectral_metadata['original_axis_unit']
            spectral_metadata['frequency_unit_original'] = converter.frequency_unit_before_conversion
            spectral_metadata['current_axis_ctype'] = header.get(f'CTYPE{converter.freq_axis}', '')
            spectral_metadata['current_axis_type'] = _classify_axis_type(spectral_metadata['current_axis_ctype'])
            spectral_metadata['current_axis_unit'] = header.get(f'CUNIT{converter.freq_axis}', '').strip() or None
            spectral_metadata['restfreq_hz'] = converter.restfreq
            refresh_alternate_reference(header, converter.freq_axis, converter.restfreq)
        elif converter.to_frequency:
            # A radio-velocity axis shown in frequency; the data keep their order.
            header = converter.header
            spectral_metadata['converted_from_velocity'] = True  # the disk unit stays in original_axis_unit
            spectral_metadata['axis_index'] = converter.freq_axis
            spectral_metadata['current_axis_ctype'] = header.get(f'CTYPE{converter.freq_axis}', '')
            spectral_metadata['current_axis_type'] = 'frequency'
            spectral_metadata['current_axis_unit'] = 'Hz'
            spectral_metadata['restfreq_hz'] = converter.restfreq
            refresh_alternate_reference(header, converter.freq_axis, converter.restfreq)

        # Normalize TIMESYS value
        #if header.get("TIMESYS") == 'UTC':
        #    header["TIMESYS"] = 'utc'
        
        # Expand 2D data to 3D/4D if necessary
        if header.get('NAXIS', 0) == 2 and "CDELT3" in header:
            header['NAXIS'] = 3
            header['NAXIS3'] = 1
            if is_lazy_scaled(data):
                data = data._raw_view_op(np.expand_dims, axis=0)
            else:
                data = np.expand_dims(data, axis=0)
            print("\033[1;33m\033[1mWarning: Expanded NAXIS to 3D with 1-pixel 3rd axis.\033[0m")
            if "CDELT4" in header:
                header['NAXIS'] = 4
                header['NAXIS4'] = 1
                if is_lazy_scaled(data):
                    data = data._raw_view_op(np.expand_dims, axis=0)
                else:
                    data = np.expand_dims(data, axis=0)
                print("\033[1;33m\033[1mWarning: Expanded NAXIS to 4D with 1-pixel 4th axis.\033[0m")
        
        # Remove unnecessary PC3 keys for 2D data
        if header.get('NAXIS', 0) == 2 and "PC3_1" in header:
            for key in ["PC3_1", "PC3_2", "PC3_3"]:
                if key in header:
                    del header[key]
            print("\033[1;33m\033[1mWarning: Removed unnecessary PC3 keys from header.\033[0m")
        
        data = _collapse_singleton_axes(data, header)
        data_nbytes = estimate_array_nbytes(data)
        large_data_profile = build_large_data_profile(data, header=header)
        spectral_metadata['large_data_mode'] = bool(large_data_profile.get('enabled'))
        spectral_metadata['large_data_profile'] = large_data_profile

    for axis_number, original_unit, canonical_unit in normalize_velocity_cunits(header):
        print(f"\033[96mRead CUNIT{axis_number} '{original_unit}' as {canonical_unit}.\033[0m")

    wcs = None
    if compute_wcs:
        try:
            wcs = WCS(header)
        except Exception as e:
            print("\033[93mWarning: Unmatched celestial axes.\033[0m")
            if header.get('NAXIS', 0) == 2:
                # Identify velocity and non-velocity axes.
                velocity_indices = []
                non_velocity_indices = []
                for i in range(1, 3):
                    ctype = header.get(f'CTYPE{i}', '').upper()
                    if 'VRAD' in ctype or 'VEL' in ctype or 'VOPT' in ctype or 'FREQ' in ctype:
                        velocity_indices.append(i)  # the spectral axis (FREQ too: --spectral-axis freq)
                    else:
                        non_velocity_indices.append(i)
                
                # If exactly one axis is velocity and one is non-velocity, modify the non-velocity axis.
                if len(velocity_indices) == 1 and len(non_velocity_indices) == 1:
                    non_vel = non_velocity_indices[0]
                    orig_ctype = header.get(f'CTYPE{non_vel}', '')
                    # Remove projection info by taking only the part before any '-' character.
                    if '-' in orig_ctype:
                        new_ctype = orig_ctype.split('-')[0]
                    else:
                        new_ctype = orig_ctype
                    print(f"\033[96mDetected position-velocity diagram. Changing CTYPE{non_vel} from '{orig_ctype}' to '{new_ctype}'.\033[0m")
                    print("\033[93mInterpret as a simple Cartesian coordinate system.\033[0m")
                    spectral_metadata['is_cartesian_interpretation'] = True
                    header[f'CTYPE{non_vel}'] = new_ctype
                    # Leave CUNIT unchanged if it exists.
                    try:
                        wcs = WCS(header)
                    except Exception as e2:
                        print(f"Failed to create modified WCS: {e2}")
                        wcs = None
                else:
                    wcs = None
            else:
                wcs = None

        # Trim WCS axes if data was collapsed (e.g., dropped singleton axes).
        if wcs is not None and data is not None:
            data_ndim = getattr(data, 'ndim', 0)
            try:
                wcs_dim = wcs.pixel_n_dim
            except Exception:
                wcs_dim = wcs.wcs.naxis

            if wcs_dim > data_ndim:
                drop_count = wcs_dim - data_ndim
                try:
                    for _ in range(drop_count):
                        wcs = wcs.dropaxis(-1)
                    if wcs.pixel_n_dim == data_ndim:
                        print("\033[1;33m\033[1mWarning: Trimmed WCS axes to match data dimensions.\033[0m")
                except Exception:
                    pass
        
        # Velocity unit conversion if WCS is available
        if wcs is not None:
            spec_axis_idx = spectral_metadata['axis_index'] or _identify_spectral_axis(header)
            if spec_axis_idx is not None:
                spectral_metadata['axis_index'] = spec_axis_idx
            apply_load_convention(wcs, header, spectral_metadata, spec_axis_idx)
    final_axis_idx = spectral_metadata['axis_index'] or _identify_spectral_axis(header)
    if final_axis_idx is not None:
        spectral_metadata['axis_index'] = final_axis_idx
        spectral_metadata['current_axis_ctype'] = header.get(f'CTYPE{final_axis_idx}', spectral_metadata['current_axis_ctype'])
        spectral_metadata['current_axis_type'] = _classify_axis_type(spectral_metadata['current_axis_ctype'])
        spectral_unit = header.get(f'CUNIT{final_axis_idx}', '')
        spectral_metadata['current_axis_unit'] = spectral_unit.strip() if isinstance(spectral_unit, str) and spectral_unit.strip() else spectral_metadata['current_axis_unit']
    else:
        spectral_metadata['current_axis_type'] = spectral_metadata['current_axis_type'] or 'unknown'

    spectral_metadata['restfreq_hz'] = _get_restfreq_hz(header)
    if spectral_metadata['restfreq_original_hz'] is None:
        spectral_metadata['restfreq_original_hz'] = spectral_metadata['restfreq_hz']
    spectral_metadata['spectral_axis_mode'] = _spectral_axis_mode(spectral_metadata)
    # A systemic redshift saved by takefits (TF-415 slice B): RESTFRQ is f0 / (1 + z).
    systemic_z, zsource_notice = systemic_redshift_from_header(header)
    if zsource_notice:
        print(f"\033[93m{zsource_notice}\033[0m")
    spectral_metadata['systemic_z'] = systemic_z
    rest_hz = spectral_metadata['restfreq_hz']
    spectral_metadata['line_restfreq_hz'] = None if rest_hz is None else rest_hz * (1.0 + systemic_z)
    spectral_metadata['zsource_header'] = header.get('ZSOURCE') if zsource_notice else None

    return data, header, wcs, spectral_metadata
