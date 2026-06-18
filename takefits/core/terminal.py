"""Shared terminal-output state.

The single-FITS click read-out updates two lines in place with ANSI cursor
moves (``\\r`` … ``\\n`` … ``\\033[1A``) and deliberately leaves the cursor on
the first of those two lines, so the next click overwrites it. That means any
*other* print afterwards (opening a new FITS, warnings, …) would append onto the
dangling line and collide, e.g. ``Clicked at (…, 13Loading: foo.fits``.

Track whether an in-place read-out is pending so normal line-oriented output can
first move past it onto a clean line. The terminal is process-global and shared
by every window, so module-level state is the correct scope here.
"""
import sys

_inplace_pending = False


def mark_inplace_pending() -> None:
    """Record that an in-place read-out left the cursor on a dangling line."""
    global _inplace_pending
    _inplace_pending = True


def commit_inplace() -> None:
    """If an in-place read-out is dangling, move past it to a fresh line."""
    global _inplace_pending
    if not _inplace_pending:
        return
    _inplace_pending = False
    try:
        # The cursor sits on the first of the read-out's two lines; advance two
        # rows to a clean line below the block (the read-out lines are kept).
        sys.stdout.write("\n\n")
        sys.stdout.flush()
    except Exception:
        pass
