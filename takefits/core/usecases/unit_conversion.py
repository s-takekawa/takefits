"""Unit conversion usecases."""
from __future__ import annotations

from typing import Any, Literal, Tuple

import numpy as np

from takefits.core.app_state import AppState


IntensityUnit = Literal["jy/beam", "k", "jy/pix"]


def convert_intensity_unit(
    data: np.ndarray,
    header: Any,
    from_unit: str,
    to_unit: str,
    method: str = "rayleigh-jeans"
) -> Tuple[np.ndarray, str]:
    """
    Convert intensity units between Jy/beam, K (brightness temperature), and Jy/pixel.
    Supports spectral cubes with varying frequency.

    Args:
        data: Input data array
        header: FITS header
        from_unit: Source unit ("jy/beam", "k", "jy/pix")
        to_unit: Target unit ("jy/beam", "k", "jy/pix")
        method: Conversion method ("rayleigh-jeans" or "planck")

    Returns:
        (converted_data, new_bunit_string)
    """
    from astropy import constants as const

    from_unit = from_unit.lower().replace(" ", "").replace("jy/pixel", "jy/pix")
    to_unit = to_unit.lower().replace(" ", "").replace("jy/pixel", "jy/pix")

    if from_unit == to_unit:
        return data.copy(), to_unit.upper()

    # --- Helpers adapted from UnitConversionPanel ---
    def get_beam_area_sr():
        if 'BMAJ' not in header or 'BMIN' not in header:
            raise ValueError("BMAJ and BMIN required in header for conversion")
        bmaj_rad = np.deg2rad(float(header['BMAJ']))
        bmin_rad = np.deg2rad(float(header['BMIN']))
        return (np.pi * bmaj_rad * bmin_rad) / (4.0 * np.log(2.0))

    def get_pixel_area_sr():
        if 'CDELT1' not in header or 'CDELT2' not in header:
            raise ValueError("CDELT1 and CDELT2 required in header for Jy/pix conversion")
        cdelt1_rad = np.deg2rad(float(header['CDELT1']))
        cdelt2_rad = np.deg2rad(float(header['CDELT2']))
        return np.abs(cdelt1_rad * cdelt2_rad)

    def get_freq_axis_hz():
        # Extracted from UnitConversionPanel
        naxis = header.get('NAXIS', 0)
        spec_axis_num = -1
        for i in range(1, naxis + 1):
            ctype_i = header.get(f'CTYPE{i}', '')
            if 'FREQ' in ctype_i or 'VELO' in ctype_i or 'VRAD' in ctype_i:
                spec_axis_num = i
                break

        if 'RESTFRQ' in header:
            restfreq_hz = float(header['RESTFRQ'])
            if spec_axis_num == -1 or naxis < 3:
                return restfreq_hz

            n_channels = header.get(f'NAXIS{spec_axis_num}', 1)
            if n_channels <= 1:
                return restfreq_hz

            crval = header.get(f'CRVAL{spec_axis_num}', 0)
            cdelt = header.get(f'CDELT{spec_axis_num}', 1)
            crpix = header.get(f'CRPIX{spec_axis_num}', 1)
            axis_values = (np.arange(n_channels) - (crpix - 1)) * cdelt + crval

            ctype = header.get(f'CTYPE{spec_axis_num}', '')
            cunit = header.get(f'CUNIT{spec_axis_num}', '').strip().lower()

            if 'VELO' in ctype or 'VRAD' in ctype:
                vel_to_ms = {'km/s': 1000.0, 'm/s': 1.0}.get(cunit, 1.0)
                axis_values_m_s = axis_values * vel_to_ms
                return restfreq_hz * (1.0 - axis_values_m_s / const.c.to('m/s').value)
            else:  # FREQ axis
                freq_to_hz = {'ghz': 1e9, 'mhz': 1e6, 'khz': 1e3}.get(cunit, 1.0)
                return axis_values * freq_to_hz

        if spec_axis_num != -1:
            n_channels = header.get(f'NAXIS{spec_axis_num}', 1)
            crval = header.get(f'CRVAL{spec_axis_num}', 0)
            cunit = header.get(f'CUNIT{spec_axis_num}', '').strip().lower()
            factor = {'ghz': 1e9, 'mhz': 1e6, 'khz': 1e3}.get(cunit, 1.0)

            if n_channels <= 1:
                return crval * factor

            cdelt = header.get(f'CDELT{spec_axis_num}', 1)
            crpix = header.get(f'CRPIX{spec_axis_num}', 1)
            freqs = (np.arange(n_channels) - (crpix - 1)) * cdelt + crval
            return freqs * factor

        raise ValueError("Frequency axis (RESTFRQ or FREQ axis) not found.")

    def reshape_freqs(freqs_hz):
        n_dims = data.ndim
        if n_dims < 3 or not isinstance(freqs_hz, np.ndarray) or freqs_hz.ndim == 0 or len(freqs_hz) <= 1:
            return freqs_hz.mean() if isinstance(freqs_hz, np.ndarray) else freqs_hz
        else:
            freq_axis_dim = n_dims - 3
            shape = [1] * n_dims
            shape[freq_axis_dim] = -1
            return freqs_hz.reshape(tuple(shape))

    # --- Conversion Logic ---
    if from_unit == "jy/beam" and to_unit == "k":
        beam_sr = get_beam_area_sr()
        freq_hz = get_freq_axis_hz()
        freqs_use = reshape_freqs(freq_hz)

        if method == "rayleigh-jeans":
            s_nu_si = (data * 1e-26) / beam_sr
            factor = (const.c.value**2) / (2 * const.k_B.value * (freqs_use**2))
            return s_nu_si * factor, "K"
        elif method == "planck":
            if not isinstance(freqs_use, np.ndarray):
                raise ValueError("Planck requires 3D+")
            s_nu_per_sr = (data * 1e-26) / beam_sr
            h, k, c = const.h.value, const.k_B.value, const.c.value
            term1 = (h * freqs_use) / k
            term2 = (2.0 * h * freqs_use**3) / (c**2)
            log_arg = 1.0 + (term2 / s_nu_per_sr)
            # Handle invalid log inputs? UnitConversionPanel uses errstate
            with np.errstate(divide='ignore', invalid='ignore'):
                return term1 / np.log(log_arg), "K"

    elif from_unit == "k" and to_unit == "jy/beam":
        beam_sr = get_beam_area_sr()
        freq_hz = get_freq_axis_hz()
        freqs_use = reshape_freqs(freq_hz)

        if method == "rayleigh-jeans":
            factor = (2 * const.k_B.value * (freqs_use**2)) / (const.c.value**2)
            s_nu_si_per_k = factor * beam_sr / 1e-26
            return data * s_nu_si_per_k, "Jy/beam"
        elif method == "planck":
            if not isinstance(freqs_use, np.ndarray):
                raise ValueError("Planck requires 3D+")
            h, k, c = const.h.value, const.k_B.value, const.c.value
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                term1 = (h * freqs_use) / k
                term2 = (2.0 * h * freqs_use**3) / (c**2)
                exp_term = np.exp(term1 / data)
                i_nu = term2 * (1.0 / (exp_term - 1.0))
                return (i_nu * beam_sr) / 1e-26, "Jy/beam"

    elif from_unit == "jy/beam" and to_unit == "jy/pix":
        beam_sr = get_beam_area_sr()
        pix_sr = get_pixel_area_sr()
        return data * (pix_sr / beam_sr), "Jy/pixel"

    elif from_unit == "jy/pix" and to_unit == "jy/beam":
        beam_sr = get_beam_area_sr()
        pix_sr = get_pixel_area_sr()
        return data * (beam_sr / pix_sr), "Jy/beam"

    elif from_unit == "k" and to_unit == "jy/pix":
        # Chain conversions (K->Jy/beam->Jy/pix)
        interim, _ = convert_intensity_unit(data, header, "k", "jy/beam", method)
        return convert_intensity_unit(interim, header, "jy/beam", "jy/pix", method)

    elif from_unit == "jy/pix" and to_unit == "k":
        # Chain (Jy/pix->Jy/beam->K)
        interim, _ = convert_intensity_unit(data, header, "jy/pix", "jy/beam", method)
        return convert_intensity_unit(interim, header, "jy/beam", "k", method)

    else:
        raise ValueError(f"Unsupported conversion: {from_unit} -> {to_unit}")


def apply_unit_conversion(
    state: AppState,
    from_unit: str,
    to_unit: str,
    method: str = "rayleigh-jeans",
) -> AppState:
    """
    Convert state.data units and update state.header['BUNIT'].
    """
    if state.data is None:
        raise ValueError("No data loaded")
    if state.header is None:
        raise ValueError("No FITS header loaded")

    converted, new_bunit = convert_intensity_unit(
        data=state.data,
        header=state.header,
        from_unit=from_unit,
        to_unit=to_unit,
        method=method,
    )
    state.data = converted
    state.header["BUNIT"] = new_bunit
    return state
