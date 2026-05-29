"""Clump finding usecases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from takefits.core.app_state import AppState
from .utils import update_datamin_datamax_if_present


@dataclass
class ClumpResult:
    """Result from clump finding algorithm."""
    mask: np.ndarray  # Integer label mask
    n_clumps: int
    catalog: list  # List of dicts with properties per clump
    algorithm: str
    parameters: Dict[str, Any]


def run_clumpfind(
    state: AppState,
    rms: float,
    min_threshold_sigma: float = 3.0,
    step_sigma: float = 2.0,
    min_pixels: int = 10
) -> ClumpResult:
    """
    Run ClumpFind algorithm on state data.

    Args:
        state: AppState with data
        rms: RMS noise level (in data units)
        min_threshold_sigma: Minimum threshold in units of RMS
        step_sigma: Contour step size in units of RMS
        min_pixels: Minimum pixels per clump

    Returns:
        ClumpResult with mask and catalog
    """
    from takefits.logic.clumpfind import ClumpFind

    if state.data is None:
        raise ValueError("No data loaded")

    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]

    min_val = rms * min_threshold_sigma
    step = rms * step_sigma

    finder = ClumpFind(data, wcs=state.wcs)
    mask = finder.run(min_val=min_val, step=step, min_pix=min_pixels)
    catalog = finder.get_catalog()

    n_clumps = len(np.unique(mask)) - 1  # Exclude 0 (background)
    if n_clumps < 0:
        n_clumps = 0

    return ClumpResult(
        mask=mask,
        n_clumps=n_clumps,
        catalog=catalog,
        algorithm="clumpfind",
        parameters={
            "rms": rms,
            "min_threshold_sigma": min_threshold_sigma,
            "step_sigma": step_sigma,
            "min_pixels": min_pixels,
            "min_val": min_val,
            "step": step
        }
    )


def run_fellwalker(
    state: AppState,
    rms: float,
    min_threshold_sigma: float = 3.0,
    min_dip_sigma: float = 2.0,
    min_pixels: int = 10
) -> ClumpResult:
    """
    Run FellWalker (watershed-based) algorithm.

    Args:
        state: AppState with data
        rms: RMS noise level (in data units)
        min_threshold_sigma: Minimum threshold in units of RMS
        min_dip_sigma: Minimum dip (prominence) in units of RMS
        min_pixels: Minimum pixels per clump

    Returns:
        ClumpResult with mask and catalog
    """
    from takefits.logic.fellwalker import FellWalker

    if state.data is None:
        raise ValueError("No data loaded")

    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]

    min_val = rms * min_threshold_sigma
    min_dip = rms * min_dip_sigma

    walker = FellWalker(data, wcs=state.wcs)
    mask = walker.run(min_val=min_val, min_dip=min_dip, min_pix=min_pixels)
    catalog = walker.get_catalog()

    n_clumps = len(np.unique(mask)) - 1  # Exclude 0 (background)
    if n_clumps < 0:
        n_clumps = 0

    return ClumpResult(
        mask=mask,
        n_clumps=n_clumps,
        catalog=catalog,
        algorithm="fellwalker",
        parameters={
            "rms": rms,
            "min_threshold_sigma": min_threshold_sigma,
            "min_dip_sigma": min_dip_sigma,
            "min_pixels": min_pixels,
            "min_val": min_val,
            "min_dip": min_dip
        }
    )


def run_dendrogram(
    state: AppState,
    rms: float,
    min_value_sigma: float = 3.0,
    min_delta_sigma: float = 2.0,
    min_npix: int = 10,
    output_mode: str = "leaves",
    use_scimes: bool = False,
    scimes_criteria: Optional[list] = None,
    scimes_user_k: int = 0,
    scimes_save_isol: bool = True
) -> ClumpResult:
    """
    Run Dendrogram algorithm (optionally with SCIMES clustering).

    Args:
        state: AppState with data
        rms: RMS noise level (in data units)
        min_value_sigma: Minimum value in units of RMS
        min_delta_sigma: Minimum delta (height difference) in units of RMS
        min_npix: Minimum pixels per structure
        output_mode: "leaves", "roots", or "all"
        use_scimes: Whether to apply SCIMES clustering
        scimes_criteria: List of criteria for SCIMES (e.g., ["luminosity", "volume"])
        scimes_user_k: Target number of clusters for SCIMES (0=auto)
        scimes_save_isol: Whether to include isolated leaves in SCIMES result

    Returns:
        ClumpResult with mask and catalog
    """
    from takefits.logic.dendro_handler import DendroHandler

    if state.data is None:
        raise ValueError("No data loaded")

    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]

    min_value = rms * min_value_sigma
    min_delta = rms * min_delta_sigma

    # Pass header to handler!
    handler = DendroHandler(data, wcs=state.wcs, header=state.header)
    handler.run_dendrogram(min_value=min_value, min_delta=min_delta, min_npix=min_npix)

    if use_scimes and scimes_criteria:
        ok, message = handler.run_scimes(
            criteria=scimes_criteria,
            user_k=scimes_user_k,
            rms=rms,
            save_isol_leaves=scimes_save_isol
        )
        if not ok:
            raise ValueError(f"SCIMES failed: {message}")

    mask = handler.get_mask(mode=output_mode)
    catalog = handler.get_catalog()

    n_clumps = len(np.unique(mask)) - 1  # Exclude 0 (background)
    if n_clumps < 0:
        n_clumps = 0

    parameters = {
        "rms": rms,
        "min_value_sigma": min_value_sigma,
        "min_delta_sigma": min_delta_sigma,
        "min_npix": min_npix,
        "output_mode": output_mode,
        "use_scimes": use_scimes,
        "min_value": min_value,
        "min_delta": min_delta
    }

    if use_scimes:
        parameters.update({
            "scimes_criteria": scimes_criteria,
            "scimes_user_k": scimes_user_k,
            "scimes_save_isol": scimes_save_isol,
        })

    return ClumpResult(
        mask=mask,
        n_clumps=n_clumps,
        catalog=catalog,
        algorithm="dendrogram",
        parameters=parameters
    )


def check_scimes_availability() -> tuple[bool, str]:
    """
    Check if SCIMES is available.

    Returns:
        (True, "") if SCIMES is available.
        (False, error_message) if it is not.
    """
    try:
        from takefits.logic.dendro_handler import DendroHandler
        unavailable_reason = DendroHandler.scimes_unavailable_reason()
        if not unavailable_reason:
            return True, ""
        return False, f"External SCIMES package missing dependency: {unavailable_reason}"
    except ImportError as e:
        return False, f"astrodendro or other dendro dependency missing: {e}"


def generate_catalog(state: AppState, mask: np.ndarray) -> List[Dict[str, Any]]:
    """
    Generate a catalog of properties for clumps in the provided mask.

    Args:
        state: AppState containing data and WCS
        mask: Label mask (same shape as state.data)

    Returns:
        List of property dictionaries (one per clump)
    """
    from takefits.logic.cloud_catalog_utils import calculate_moments_and_props

    if state.data is None:
        raise ValueError("No data loaded")

    # Use data from state (handling 4th axis if needed)
    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]

    if mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match data shape {data.shape}")

    labels = np.unique(mask)
    labels = labels[labels > 0]

    catalog = []
    for l in labels:
        props = calculate_moments_and_props(data, mask, l, wcs=state.wcs)
        if props:
            catalog.append(props)

    return catalog


def export_clump_mask(
    state: AppState,
    result: ClumpResult,
    output_path: str,
    history_entries: Optional[list] = None
) -> str:
    """
    Export clump mask to FITS file.

    Args:
        state: AppState with original header info
        result: ClumpResult from clump finding
        output_path: Path for output FITS file
        history_entries: Optional list of HISTORY entries

    Returns:
        The output file path
    """
    from astropy.io import fits

    header = state.header.copy() if state.header is not None else fits.Header()

    # Update shape info
    header['NAXIS'] = result.mask.ndim
    for i, dim in enumerate(result.mask.shape[::-1], 1):
        header[f'NAXIS{i}'] = dim

    # Add clump finding info
    header['CLUMPALG'] = (result.algorithm, 'Clump finding algorithm')
    header['NCLUMPS'] = (result.n_clumps, 'Number of clumps found')

    # Add algorithm parameters
    for key, value in result.parameters.items():
        if isinstance(value, (int, float, str, bool)):
            key_fits = key[:8].upper()  # FITS keyword limit
            header[key_fits] = value

    # Add history
    if history_entries:
        for entry in history_entries:
            header.add_history(entry)

    update_datamin_datamax_if_present(header, result.mask)

    # Write file
    hdu = fits.PrimaryHDU(data=result.mask.astype(np.int32), header=header)
    hdu.writeto(output_path, overwrite=True)

    return output_path


def export_clump_catalog(
    result: ClumpResult,
    output_path: str,
    format: str = "csv"
) -> str:
    """
    Export clump catalog to CSV or FITS table.

    Args:
        result: ClumpResult from clump finding
        output_path: Path for output file
        format: "csv" or "fits"

    Returns:
        The output file path
    """
    if not result.catalog:
        # Write empty file
        if format == "csv":
            with open(output_path, 'w') as f:
                f.write("# No clumps found\n")
        else:
            from astropy.io import fits
            hdu = fits.BinTableHDU.from_columns([])
            hdu.writeto(output_path, overwrite=True)
        return output_path

    if format == "csv":
        import csv

        # Get all keys from catalog entries
        all_keys = set()
        for entry in result.catalog:
            all_keys.update(entry.keys())
        all_keys = sorted(all_keys)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for entry in result.catalog:
                writer.writerow(entry)

    elif format == "fits":
        from astropy.io import fits
        from astropy.table import Table

        # Convert catalog to astropy Table
        table = Table(result.catalog)
        table.write(output_path, format='fits', overwrite=True)

    else:
        raise ValueError(f"Unknown format: {format}")

    return output_path
