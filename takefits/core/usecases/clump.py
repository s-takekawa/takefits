"""Clump finding usecases."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from takefits.core.app_state import AppState
from takefits.logic.data_tools import (
    _get_available_memory_bytes,
    ensure_operation_memory_budget,
    estimate_materialized_nbytes,
    format_nbytes,
    is_lazy_scaled,
    materialize_elementwise_inputs,
)
from takefits.logic.progress import CancellationToken, ProgressReporter
from .utils import update_datamin_datamax_if_present

_CLUMP_MASK_RAM_FRACTION = 0.8
_CLUMP_MASK_FALLBACK_BYTES = 512 * 1024 ** 2
_CLUMP_EXTRA_BYTES_PER_VOXEL = {
    "clumpfind": 32,
    "fellwalker": 56,
    "dendrogram": 40,
}
_CLUMP_ESTIMATE_UNCERTAINTY_FACTOR = {
    "clumpfind": 1.20,
    "fellwalker": 1.20,
    # Astrodendro's Python structure graph depends strongly on fragmentation,
    # so retain more headroom than the dense-array algorithms.
    "dendrogram": 1.50,
}
_CLUMP_LABEL_COUNT_CHUNK_BYTES = 16 * 1024 ** 2


@dataclass
class ClumpResult:
    """Result from clump finding algorithm."""
    mask: np.ndarray  # Integer label mask
    n_clumps: int
    catalog: list  # List of dicts with properties per clump
    algorithm: str
    parameters: Dict[str, Any]


def _detect_total_ram_bytes() -> int | None:
    """Backward-compatible memory probe used by older callers/tests."""
    return _get_available_memory_bytes()


def _clump_mask_memory_limit_bytes() -> int:
    available_ram = _detect_total_ram_bytes()
    if available_ram is not None:
        return max(1, int(available_ram * _CLUMP_MASK_RAM_FRACTION))
    return _CLUMP_MASK_FALLBACK_BYTES


def _estimate_label_mask_nbytes(data) -> int | None:
    shape = getattr(data, "shape", None)
    if not shape:
        return None
    count = 1
    try:
        for dim in shape:
            count *= int(dim)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(count) * np.dtype(np.int32).itemsize


def _estimate_clump_working_nbytes(data, algorithm: str) -> int | None:
    """Estimate algorithm-specific temporary/output bytes beyond the source."""
    label_bytes = _estimate_label_mask_nbytes(data)
    if label_bytes is None:
        return None
    pixel_count = label_bytes // np.dtype(np.int32).itemsize
    key = str(algorithm or "").strip().lower()
    extra_per_voxel = _CLUMP_EXTRA_BYTES_PER_VOXEL.get(key, 40)
    uncertainty = _CLUMP_ESTIMATE_UNCERTAINTY_FACTOR.get(key, 1.25)
    needed = int(np.ceil(pixel_count * int(extra_per_voxel) * uncertainty))
    if is_lazy_scaled(data):
        needed += int(estimate_materialized_nbytes(data) or 0)
    return int(needed)


def _ensure_label_mask_memory_budget(data, algorithm: str) -> None:
    """Refuse unsafe algorithm worksets, including the result label mask."""
    label_bytes = _estimate_label_mask_nbytes(data)
    needed = _estimate_clump_working_nbytes(data, algorithm)
    if label_bytes is None or needed is None:
        return
    pixel_count = label_bytes // np.dtype(np.int32).itemsize

    limit = _clump_mask_memory_limit_bytes()
    shape = tuple(int(axis) for axis in getattr(data, "shape", ()) or ())
    if needed > limit:
        raise MemoryError(
            f"{algorithm} cannot safely run on this cube because its working "
            f"set, including the result label mask, is estimated at "
            f"{format_nbytes(needed)} ({pixel_count:,} pixels, shape={shape}); "
            f"the current safety limit is {format_nbytes(limit)}. Use Tools > "
            "Cutout, fewer channels, or smaller spatial bounds before running "
            "clump finding."
        )

    ensure_operation_memory_budget(
        needed,
        operation_name=algorithm,
        guidance=(
            "Use Tools > Cutout, fewer channels, or smaller spatial bounds "
            "before running clump finding."
        ),
    )


def _prepare_clump_data(data, algorithm: str) -> np.ndarray:
    """Preflight the algorithm then materialize a lazy cube once."""
    _ensure_label_mask_memory_budget(data, algorithm)
    prepared, = materialize_elementwise_inputs(
        data,
        operation_name=algorithm,
        output_array_count=0.0,
    )
    # Re-sample currently available RAM immediately after a lazy cube has
    # become resident. This closes most of the window in which another process
    # can invalidate the first preflight before the algorithm allocates masks.
    _ensure_label_mask_memory_budget(prepared, algorithm)
    return prepared


def _count_positive_labels(mask: np.ndarray) -> int:
    """Count distinct positive labels without sorting a full mask copy."""
    values = np.asarray(mask)
    if values.size == 0:
        return 0
    items_per_chunk = max(
        1,
        _CLUMP_LABEL_COUNT_CHUNK_BYTES // max(1, int(values.dtype.itemsize)),
    )
    labels = set()
    with np.nditer(
        values,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=items_per_chunk,
    ) as iterator:
        for block in iterator:
            for label in np.unique(block):
                label_value = int(label)
                if label_value > 0:
                    labels.add(label_value)
    return len(labels)


def _source_data_for_mask(state: AppState, mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the source data array corresponding to a label mask, if known."""
    data = state.data
    if data is None:
        return None

    mask_shape = getattr(mask, "shape", None)
    if getattr(data, "shape", None) == mask_shape:
        return data

    if getattr(data, "ndim", None) == 4:
        try:
            current_s = int(getattr(state, "current_s", 0))
        except (TypeError, ValueError):
            current_s = 0
        current_s = max(0, min(current_s, data.shape[0] - 1))
        candidate = data[current_s]
        if candidate.shape == mask_shape:
            return candidate

    return None


def _clump_mask_data_for_export(state: AppState, mask: np.ndarray) -> np.ndarray:
    """Build FITS data for a clump mask, preserving source NaN pixels."""
    mask_array = np.asarray(mask)
    source_data = _source_data_for_mask(state, mask_array)
    if source_data is None:
        return mask_array.astype(np.int32, copy=False)

    shape = tuple(int(dim) for dim in mask_array.shape)
    leading_shape = shape[:-2] if len(shape) >= 2 else shape

    def _plane_at(array, leading_index):
        plane = array[leading_index] if leading_index else array
        if np.ma.isMaskedArray(plane):
            return np.asarray(np.ma.filled(plane, np.nan))
        return np.asarray(plane)

    has_source_nan = False
    try:
        for leading_index in np.ndindex(leading_shape):
            source_plane = _plane_at(source_data, leading_index)
            if np.isnan(source_plane).any():
                has_source_nan = True
                break
    except TypeError:
        return mask_array.astype(np.int32, copy=False)

    if not has_source_nan:
        return mask_array.astype(np.int32, copy=False)

    output_bytes = mask_array.size * np.dtype(np.float32).itemsize
    plane_pixels = (
        int(shape[-2]) * int(shape[-1])
        if len(shape) >= 2
        else int(mask_array.size)
    )
    ensure_operation_memory_budget(
        output_bytes + plane_pixels * np.dtype(bool).itemsize,
        operation_name="Clump mask FITS export",
        guidance=(
            "Free memory or export clumps from a smaller cutout before saving."
        ),
    )

    export_data = mask_array.astype(np.float32, copy=True)
    for leading_index in np.ndindex(leading_shape):
        source_plane = _plane_at(source_data, leading_index)
        source_nan = np.isnan(source_plane)
        if np.any(source_nan):
            export_plane = export_data[leading_index] if leading_index else export_data
            export_plane[source_nan] = np.nan
    return export_data


def run_clumpfind(
    state: AppState,
    rms: float,
    min_threshold_sigma: float = 3.0,
    step_sigma: float = 2.0,
    min_pixels: int = 10,
    progress_callback: Optional[Callable[[Optional[int], Optional[str]], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
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

    reporter = ProgressReporter(progress_callback, cancel_token)

    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]
    data = _prepare_clump_data(data, "ClumpFind")

    min_val = rms * min_threshold_sigma
    step = rms * step_sigma

    finder = ClumpFind(data, wcs=state.wcs)
    mask = finder.run(min_val=min_val, step=step, min_pix=min_pixels, reporter=reporter)
    reporter.update(96, "Building catalog...")
    catalog = finder.get_catalog()
    reporter.update(100, "Done.")

    n_clumps = _count_positive_labels(mask)

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
    min_pixels: int = 10,
    progress_callback: Optional[Callable[[Optional[int], Optional[str]], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
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

    reporter = ProgressReporter(progress_callback, cancel_token)

    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]
    data = _prepare_clump_data(data, "FellWalker")

    min_val = rms * min_threshold_sigma
    min_dip = rms * min_dip_sigma

    walker = FellWalker(data, wcs=state.wcs)
    mask = walker.run(min_val=min_val, min_dip=min_dip, min_pix=min_pixels, reporter=reporter)
    reporter.update(96, "Building catalog...")
    catalog = walker.get_catalog()
    reporter.update(100, "Done.")

    n_clumps = _count_positive_labels(mask)

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
    scimes_save_isol: bool = True,
    progress_callback: Optional[Callable[[Optional[int], Optional[str]], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
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

    reporter = ProgressReporter(progress_callback, cancel_token)

    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]
    data = _prepare_clump_data(data, "Dendrogram")

    min_value = rms * min_value_sigma
    min_delta = rms * min_delta_sigma

    # Pass header to handler!
    handler = DendroHandler(data, wcs=state.wcs, header=state.header)
    handler.run_dendrogram(min_value=min_value, min_delta=min_delta, min_npix=min_npix, reporter=reporter)

    if use_scimes and scimes_criteria:
        ok, message = handler.run_scimes(
            criteria=scimes_criteria,
            user_k=scimes_user_k,
            rms=rms,
            save_isol_leaves=scimes_save_isol,
            reporter=reporter,
        )
        if not ok:
            raise ValueError(f"SCIMES failed: {message}")

    reporter.update(None, "Building label mask...")
    mask = handler.get_mask(mode=output_mode)
    reporter.update(None, "Building catalog...")
    # Reuse the mask we just built and match the catalog to the displayed mode
    # (the catalog previously defaulted to 'leaves' regardless of output_mode).
    catalog = handler.get_catalog(mode=output_mode, mask=mask, reporter=reporter)
    reporter.update(100, "Done.")

    n_clumps = _count_positive_labels(mask)

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
    from takefits.logic.cloud_catalog_utils import build_catalog

    if state.data is None:
        raise ValueError("No data loaded")

    # Use data from state (handling 4th axis if needed)
    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]

    if mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match data shape {data.shape}")

    return build_catalog(data, mask, wcs=state.wcs)


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

    export_data = _clump_mask_data_for_export(state, result.mask)
    update_datamin_datamax_if_present(header, export_data)

    # Write file
    from takefits.core.io.save_fits import atomic_write_fits

    hdu = fits.PrimaryHDU(data=export_data, header=header)
    atomic_write_fits(
        output_path,
        lambda temporary: hdu.writeto(temporary, overwrite=True),
    )

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
            from takefits.core.io.save_fits import atomic_write_fits

            hdu = fits.BinTableHDU.from_columns([])
            atomic_write_fits(
                output_path,
                lambda temporary: hdu.writeto(temporary, overwrite=True),
            )
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
        from astropy.table import Table
        from takefits.core.io.save_fits import atomic_write_fits

        # Convert catalog to astropy Table
        table = Table(result.catalog)
        atomic_write_fits(
            output_path,
            lambda temporary: table.write(
                temporary,
                format='fits',
                overwrite=True,
            ),
        )

    else:
        raise ValueError(f"Unknown format: {format}")

    return output_path
