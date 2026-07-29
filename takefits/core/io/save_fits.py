"""Headless FITS save helpers (no PyQt dependencies)."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from takefits.logic.data_tools import is_lazy_scaled

_DATAMIN_DATAMAX_CHUNK_BYTES = 32 * 1024 ** 2


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after an atomic replacement."""
    try:
        directory_fd = os.open(str(path), os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def atomic_write_fits(
    output_path: str | os.PathLike[str],
    writer: Callable[[Path], None],
    *,
    overwrite: bool = True,
) -> str:
    """Write a complete FITS payload before replacing the destination.

    ``writer`` receives a unique temporary path in the destination directory.
    A failed write, flush, or replacement leaves an existing destination
    untouched. Keeping the temporary file beside the destination also makes the
    final ``os.replace`` a same-filesystem operation on Windows, Linux, macOS,
    and locally synchronized folders.
    """
    requested_target = Path(output_path)
    # Match ordinary open()/Astropy semantics for an explicitly supplied
    # symlink: update its destination atomically instead of replacing the link
    # itself with a regular file.
    target = (
        requested_target.resolve(strict=False)
        if requested_target.is_symlink()
        else requested_target
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and target.exists():
        raise FileExistsError(f"File already exists: {target}")

    existing_mode = None
    try:
        existing_mode = target.stat().st_mode & 0o7777
    except OSError:
        pass

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    # Let Astropy create the file with its normal mode and metadata rather than
    # forcing it to overwrite the restrictive mode used by mkstemp.
    temporary.unlink()

    try:
        writer(temporary)
        if not temporary.is_file():
            raise OSError("FITS writer did not create its temporary output")
        if existing_mode is not None:
            try:
                os.chmod(temporary, existing_mode)
            except OSError:
                pass
        # Windows' os.fsync() delegates to the CRT _commit(), which rejects a
        # read-only descriptor with EBADF. Open read/write even though no bytes
        # are changed here so the durability barrier works on every platform.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if not overwrite and target.exists():
            raise FileExistsError(f"File already exists: {target}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise
    return str(requested_target)


def _data_nbytes(data: Any) -> int | None:
    try:
        nbytes = getattr(data, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
    except (TypeError, ValueError):
        pass

    try:
        return int(data.size) * np.dtype(data.dtype).itemsize
    except (AttributeError, TypeError, ValueError):
        return None


def _should_chunk_extrema(data: Any) -> bool:
    if isinstance(data, np.memmap):
        return True
    nbytes = _data_nbytes(data)
    return bool(nbytes is not None and nbytes > _DATAMIN_DATAMAX_CHUNK_BYTES)


def _update_extrema_from_block(
    block: Any,
    current_min: float,
    current_max: float,
    any_finite: bool,
) -> tuple[float, float, bool]:
    if np.ma.isMaskedArray(block):
        values = np.asarray(np.ma.getdata(block))
        masked = np.ma.getmaskarray(block)
    else:
        values = np.asarray(block)
        masked = None

    if values.size == 0:
        return current_min, current_max, any_finite

    with np.errstate(all="ignore"):
        finite_mask = np.isfinite(values)
    if masked is not None:
        finite_mask &= ~masked
    if not np.any(finite_mask):
        return current_min, current_max, any_finite

    if np.issubdtype(values.dtype, np.integer):
        limits = np.iinfo(values.dtype)
        min_initial = limits.max
        max_initial = limits.min
    elif np.issubdtype(values.dtype, np.bool_):
        min_initial = True
        max_initial = False
    else:
        min_initial = np.inf
        max_initial = -np.inf

    block_min = float(np.min(values, where=finite_mask, initial=min_initial))
    block_max = float(np.max(values, where=finite_mask, initial=max_initial))
    if not any_finite:
        return block_min, block_max, True
    return min(current_min, block_min), max(current_max, block_max), True


def _finite_extrema_chunked(data: Any) -> tuple[float, float, bool]:
    if is_lazy_scaled(data):
        raw = data._raw
        # Scaling always produces float64. Base the item count on that public
        # dtype so an int8/int16 raw array cannot expand a chunk past the bound.
        buffer_items = max(
            1,
            _DATAMIN_DATAMAX_CHUNK_BYTES // np.dtype(np.float64).itemsize,
        )
        dmin, dmax, any_finite = np.inf, -np.inf, False
        with np.nditer(
            raw,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            order="K",
            buffersize=buffer_items,
        ) as iterator:
            for raw_block in iterator:
                scaled_block = data._apply_scaling(raw_block)
                dmin, dmax, any_finite = _update_extrema_from_block(
                    scaled_block,
                    dmin,
                    dmax,
                    any_finite,
                )
        return dmin, dmax, any_finite

    values = np.asarray(np.ma.getdata(data) if np.ma.isMaskedArray(data) else data)
    dtype_itemsize = max(1, int(values.dtype.itemsize))
    buffer_items = max(1, _DATAMIN_DATAMAX_CHUNK_BYTES // dtype_itemsize)
    dmin, dmax, any_finite = np.inf, -np.inf, False

    mask = np.ma.getmask(data) if np.ma.isMaskedArray(data) else np.ma.nomask
    if mask is not np.ma.nomask:
        broadcast_mask = np.broadcast_to(mask, values.shape)
        with np.nditer(
            [values, broadcast_mask],
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"], ["readonly"]],
            order="K",
            buffersize=buffer_items,
        ) as iterator:
            for value_chunk, mask_chunk in iterator:
                block = np.ma.array(value_chunk, mask=mask_chunk, copy=False)
                dmin, dmax, any_finite = _update_extrema_from_block(
                    block, dmin, dmax, any_finite
                )
        return dmin, dmax, any_finite

    with np.nditer(
        values,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=buffer_items,
    ) as iterator:
        for block in iterator:
            dmin, dmax, any_finite = _update_extrema_from_block(
                block, dmin, dmax, any_finite
            )
    return dmin, dmax, any_finite


def update_datamin_datamax_if_present(
    header: Any,
    data: Any,
    *,
    ensure: bool = False,
    drop_if_all_invalid: bool = False,
) -> None:
    """Update DATAMIN/DATAMAX in header if either keyword exists."""
    if header is None:
        return
    if not ensure and 'DATAMIN' not in header and 'DATAMAX' not in header:
        return

    if is_lazy_scaled(data) or _should_chunk_extrema(data):
        dmin, dmax, any_finite = _finite_extrema_chunked(data)
    else:
        dmin, dmax, any_finite = _update_extrema_from_block(
            data,
            np.inf,
            -np.inf,
            False,
        )
    if any_finite:
        header['DATAMIN'] = float(dmin)
        header['DATAMAX'] = float(dmax)
    elif drop_if_all_invalid:
        header.pop('DATAMIN', None)
        header.pop('DATAMAX', None)
    else:
        header['DATAMIN'] = np.nan
        header['DATAMAX'] = np.nan


def write_fits(
    output_path: str,
    data: Any,
    header: Any,
    *,
    overwrite: bool = True,
    ensure_datamin_datamax: bool = False,
    drop_extrema_if_all_invalid: bool = False,
) -> str:
    """Write a FITS file with optional DATAMIN/DATAMAX refresh."""
    from astropy.io import fits

    update_datamin_datamax_if_present(
        header,
        data,
        ensure=ensure_datamin_datamax,
        drop_if_all_invalid=drop_extrema_if_all_invalid,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_lazy_scaled(data):
        # Construct from untouched raw storage. Astropy removes scaling cards
        # while reconciling a new HDU with its ndarray, so restore them only
        # after construction; otherwise the stored integers are interpreted as
        # already-physical values and round-trip incorrectly.
        hdu_header = header.copy()
        existing_scaling_cards = set()
        for key in ("BSCALE", "BZERO", "BLANK"):
            if key in hdu_header:
                existing_scaling_cards.add(key)
                hdu_header.pop(key)

        scaling_cards = {}
        if "BSCALE" in existing_scaling_cards or data._bscale != 1.0:
            scaling_cards["BSCALE"] = data._bscale
        if "BZERO" in existing_scaling_cards or data._bzero != 0.0:
            scaling_cards["BZERO"] = data._bzero
        if data._blank is not None:
            scaling_cards["BLANK"] = data._blank

        hdu = fits.PrimaryHDU(data=np.asarray(data._raw), header=hdu_header)
        for key, value in scaling_cards.items():
            hdu.header[key] = value
        atomic_write_fits(
            target,
            lambda temporary: hdu.writeto(temporary, overwrite=True),
            overwrite=overwrite,
        )
    else:
        atomic_write_fits(
            target,
            lambda temporary: fits.writeto(
                temporary,
                data,
                header,
                overwrite=True,
            ),
            overwrite=overwrite,
        )
    return str(target)
