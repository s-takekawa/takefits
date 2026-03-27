"""Headless FITS I/O helpers (no PyQt dependencies)."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.io.fits import VerifyError
from astropy.wcs import WCS
from astropy import units as u

from takefits.logic.freq_to_velocity import FreqToVelocity
from takefits.logic.data_tools import (
    LAZY_SCALING_THRESHOLD_BYTES,
    MEMMAP_THRESHOLD_BYTES,
    LazyScaledArray,
    build_large_data_profile,
    estimate_array_nbytes,
    header_has_scaling_keywords,
    is_lazy_scaled,
)


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


def _classify_axis_type(ctype: str):
    """Return a simple classification ('frequency', 'velocity', or 'unknown') for a CTYPE value."""
    if not ctype:
        return 'unknown'
    ctype_upper = str(ctype).upper()
    if 'FREQ' in ctype_upper:
        return 'frequency'
    if any(tag in ctype_upper for tag in ('VRAD', 'VELO', 'VOPT')):
        return 'velocity'
    return 'unknown'


def _identify_spectral_axis(header):
    """Identify the first spectral-like axis (frequency or velocity)."""
    if header is None:
        return None
    try:
        naxis = int(header.get('NAXIS', 0))
    except (TypeError, ValueError):
        return None

    for axis in range(1, naxis + 1):
        ctype = header.get(f'CTYPE{axis}', '')
        if _classify_axis_type(ctype) != 'unknown':
            return axis
    return None


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

    cdelt_key = f'CDELT{fits_axis}'
    cdelt = header.get(cdelt_key)
    if cdelt is None or cdelt >= 0:
        return data, False

    naxis_key = f'NAXIS{fits_axis}'
    axis_length = header.get(naxis_key)
    data_axis = getattr(data, 'ndim', 0) - fits_axis
    if axis_length in (None, 0) and 0 <= data_axis < getattr(data, 'ndim', 0):
        axis_length = data.shape[data_axis]

    if axis_length in (None, 0) or axis_length == 1:
        header[cdelt_key] = abs(cdelt)
        return data, False

    if data_axis < 0 or data_axis >= getattr(data, 'ndim', 0):
        header[cdelt_key] = abs(cdelt)
        return data, False

    if is_lazy_scaled(data):
        data = data._raw_view_op(np.flip, axis=data_axis)
    else:
        data = np.flip(data, axis=data_axis)

    header[cdelt_key] = abs(cdelt)

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

def load_fits(filename, compute_wcs=True):
    """
    Load a FITS file and process header/data.
    Optionally compute WCS and perform velocity unit conversion.
    """
    path = Path(filename)

    if not path.exists():
        raise FITSLoadError("File not found", filename=path, kind="not_found")

    if not path.is_file():
        raise FITSLoadError("Path is not a file", filename=path, kind="not_file")

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

    # --- Quick file-size check to decide whether a header probe is worthwhile ---
    # Only probe when the file on disk could plausibly exceed the lazy-scaling
    # threshold.  For smaller files the normal open → reactive-retry path is
    # sufficient and avoids an extra fits.open() call.
    _probe_file_bytes = 0
    _probe_needs_scaling = False
    _file_size_on_disk = 0
    try:
        _file_size_on_disk = path.stat().st_size
    except Exception:
        pass

    if _file_size_on_disk >= LAZY_SCALING_THRESHOLD_BYTES:
        # Probe header to detect scaling keywords and estimate data size.
        try:
            with fits.open(path, mode='readonly', memmap=True, lazy_load=True,
                           ignore_missing_end=True, ignore_missing_simple=True) as probe:
                for _hdu in probe:
                    _hdr = getattr(_hdu, "header", None)
                    if _hdr is None:
                        continue
                    if not _probe_needs_scaling and header_has_scaling_keywords(_hdr):
                        _probe_needs_scaling = True
                h = probe[0].header
                naxis = int(h.get('NAXIS', 0))
                if naxis == 0 and len(probe) > 1:
                    h = probe[1].header
                    naxis = int(h.get('NAXIS', 0))
                if naxis > 0:
                    bitpix = abs(int(h.get('BITPIX', 8)))
                    _probe_file_bytes = bitpix // 8
                    for ax in range(1, naxis + 1):
                        _probe_file_bytes *= int(h.get(f'NAXIS{ax}', 1))
        except Exception:
            pass

        # Decide open strategy up-front to avoid close-and-reopen cycles.
        if _probe_needs_scaling and _probe_file_bytes >= LAZY_SCALING_THRESHOLD_BYTES:
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
            if not lazy_scaling_active and _probe_file_bytes >= LAZY_SCALING_THRESHOLD_BYTES:
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
        # Reactive fallback for files where no probe was done: if the open
        # succeeded with memmap but scaling keywords are present, astropy
        # will fail or produce wrong results when data is accessed.  Re-open
        # with memmap disabled so astropy can handle BZERO/BSCALE/BLANK.
        if not lazy_scaling_active and open_kwargs.get("memmap", True):
            requires_scaling = False
            for hdu in hdulist:
                hdr = getattr(hdu, "header", None)
                if hdr is None:
                    continue
                if header_has_scaling_keywords(hdr):
                    requires_scaling = True
                    break
            if requires_scaling:
                hdulist.close()
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
        data = hdul[0].data
        header = hdul[0].header

        # Wrap raw memmap in LazyScaledArray when lazy scaling is active.
        if lazy_scaling_active and data is not None:
            from takefits.logic.data_tools import _parse_header_float
            bzero = _parse_header_float(header, "BZERO", 0.0)
            if bzero is None:
                bzero = 0.0
            bscale = _parse_header_float(header, "BSCALE", 1.0)
            if bscale is None:
                bscale = 1.0
            blank = None
            if "BLANK" in header:
                try:
                    blank = int(header["BLANK"])
                except (TypeError, ValueError):
                    blank = None
            data = LazyScaledArray(data, bzero=bzero, bscale=bscale, blank=blank)
            print(
                f"\033[96m  BZERO={bzero}, BSCALE={bscale}"
                + (f", BLANK={blank}" if blank is not None else "")
                + "\033[0m"
            )

        data_nbytes = estimate_array_nbytes(data)
        spectral_axis_index = _identify_spectral_axis(header)
        original_axis_ctype = header.get(f'CTYPE{spectral_axis_index}', '') if spectral_axis_index else ''
        original_axis_unit = header.get(f'CUNIT{spectral_axis_index}', '').strip() if spectral_axis_index else None
        spectral_metadata = {
            'axis_index': spectral_axis_index,
            'original_axis_ctype': original_axis_ctype,
            'original_axis_type': _classify_axis_type(original_axis_ctype),
            'original_axis_unit': original_axis_unit,
            'current_axis_ctype': None,
            'current_axis_type': None,
            'current_axis_unit': None,
            'converted_from_frequency': False,
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
        
        # Check if primary HDU is empty; use the next HDU if needed
        if header.get('NAXIS', 0) == 0:
            data = hdul[1].data
            header = hdul[1].header
            # Re-wrap in LazyScaledArray if lazy scaling is active for the
            # replacement HDU data (the original wrap targeted HDU[0]).
            if lazy_scaling_active and data is not None and not is_lazy_scaled(data):
                bzero = _parse_header_float(header, "BZERO", 0.0)
                if bzero is None:
                    bzero = 0.0
                bscale = _parse_header_float(header, "BSCALE", 1.0)
                if bscale is None:
                    bscale = 1.0
                blank = None
                if "BLANK" in header:
                    try:
                        blank = int(header["BLANK"])
                    except (TypeError, ValueError):
                        blank = None
                data = LazyScaledArray(data, bzero=bzero, bscale=bscale, blank=blank)
            print("\033[1;33m\033[1mWarning: NAXIS was 0. Using data from the next HDU.\033[0m")
        
        # Frequency conversion using FreqToVelocity
        converter = FreqToVelocity(header)
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

        # Replace invalid data values (< -100000) with NaN.
        # Skip for LazyScaledArray (BLANK is handled by scaling; per-slice
        # sanitisation covers remaining invalid values) and for arrays
        # larger than the memmap threshold.
        if is_lazy_scaled(data):
            spectral_metadata['_needs_per_slice_sanitize'] = True
        else:
            allow_full_scan = data_nbytes is None or data_nbytes <= MEMMAP_THRESHOLD_BYTES

            if allow_full_scan:
                with np.errstate(invalid="ignore"):
                    invalid_mask = data < -100000

                if np.any(invalid_mask):
                    if not np.issubdtype(data.dtype, np.floating):
                        data = data.astype(np.float32)
                    data[invalid_mask] = np.nan
                    print("\033[1;33m\033[1mWarning: Replaced invalid data values (< -100000) with NaN.\033[0m")

                with np.errstate(invalid="ignore"):
                    inf_mask = np.isinf(data)

                if np.any(inf_mask):
                    if not np.issubdtype(data.dtype, np.floating):
                        data = data.astype(np.float32)
                    data[inf_mask] = np.nan
                    print("\033[1;33m\033[1mWarning: Replaced infinite data values with NaN.\033[0m")
            else:
                spectral_metadata['_needs_per_slice_sanitize'] = True

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
                    if 'VRAD' in ctype or 'VEL' in ctype or  'VOPT' in ctype:
                        velocity_indices.append(i)
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
            if spec_axis_idx and spec_axis_idx <= wcs.wcs.naxis:
                wcs_axis_idx = spec_axis_idx - 1
                header_cunit_key = f'CUNIT{spec_axis_idx}'
                header_cdelt_key = f'CDELT{spec_axis_idx}'
                header_crval_key = f'CRVAL{spec_axis_idx}'
                unit_header = str(header.get(header_cunit_key, '')).replace(' ', '').lower()
                try:
                    unit_wcs = wcs.wcs.cunit[wcs_axis_idx].to_string().replace(' ', '').lower()
                except Exception:
                    unit_wcs = ''

                if unit_header == 'km/s' and unit_wcs != 'km/s':
                    try:
                        wcs_unit = u.Unit(unit_wcs) if unit_wcs else None
                    except Exception:
                        wcs_unit = None
                    if wcs_unit is not None:
                        wcs.wcs.cdelt[wcs_axis_idx] = (wcs.wcs.cdelt[wcs_axis_idx] * wcs_unit).to(u.km / u.s).value
                        wcs.wcs.crval[wcs_axis_idx] = (wcs.wcs.crval[wcs_axis_idx] * wcs_unit).to(u.km / u.s).value
                elif (unit_header in ('m/s', '') and abs(wcs.wcs.cdelt[wcs_axis_idx]) > 100.0):
                    if unit_header == '':
                        print("\033[96mCUNIT{} is not found. Interpreted velocity unit as m/s.\033[0m".format(spec_axis_idx))
                    wcs.wcs.cdelt[wcs_axis_idx] = (wcs.wcs.cdelt[wcs_axis_idx] * u.m / u.s).to(u.km / u.s).value
                    wcs.wcs.crval[wcs_axis_idx] = (wcs.wcs.crval[wcs_axis_idx] * u.m / u.s).to(u.km / u.s).value
                    header[header_cunit_key] = 'km/s'
                    header[header_cdelt_key] = wcs.wcs.cdelt[wcs_axis_idx]
                    header[header_crval_key] = wcs.wcs.crval[wcs_axis_idx]
                    spectral_metadata['velocity_unit_adjusted'] = True
                    spectral_metadata['velocity_unit_original'] = 'm/s'
                    spectral_metadata['velocity_unit_target'] = 'km/s'
                    spectral_metadata['current_axis_unit'] = 'km/s'

                    print("\033[96mConverted velocity unit from m/s to km/s.\033[0m")

                elif unit_header == '':
                    print("\033[96mCUNIT{} is not found. Interpreted velocity unit as km/s.\033[0m".format(spec_axis_idx))
                    header[header_cunit_key] = 'km/s'
                    #try:
                    #    wcs.wcs.cunit[wcs_axis_idx] = u.Unit('km/s')
                    #except Exception as e:
                    #    print(f"\033[91mError: Failed to update WCS CUNIT: {e}\033[0m")
                    spectral_metadata['velocity_unit_adjusted'] = False # No conversion needed
                    spectral_metadata['velocity_unit_original'] = 'km/s' # Assumed
                    spectral_metadata['velocity_unit_target'] = 'km/s'
                    spectral_metadata['current_axis_unit'] = 'km/s'

                if spectral_metadata.get('current_axis_unit') is None:
                    spectral_metadata['current_axis_unit'] = header.get(header_cunit_key, '').strip() or None
                spectral_metadata['current_axis_ctype'] = header.get(f'CTYPE{spec_axis_idx}', spectral_metadata['current_axis_ctype'])
            elif wcs.wcs.naxis == 2:
                for i in range(2):
                    wcs_unit = wcs.wcs.cunit[i].to_string().replace(' ', '').lower()
                    if wcs_unit == 'm/s':
                        wcs.wcs.cdelt[i] = (wcs.wcs.cdelt[i] * u.m / u.s).to(u.km / u.s).value
                        wcs.wcs.crval[i] = (wcs.wcs.crval[i] * u.m / u.s).to(u.km / u.s).value
                        spectral_metadata['velocity_unit_adjusted'] = True
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

    return data, header, wcs, spectral_metadata
