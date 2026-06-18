"""Application-level cross-window synchronisation (Phase 2).

A single :class:`WindowSyncManager` (one per process, like
:class:`~takefits.ui.window_registry.WindowRegistry`) keeps several open
``MainWindow`` viewers in step when the user turns the lock on:

* **pan/zoom** — the spatial (XY) viewport is matched in *world* coordinates,
  so different grid sizes / pixel scales / projections still line up by
  angular extent. Propagated on pan/zoom release and on Home/Back/Forward.
* **cursor** — clicks are converted through world coordinates and replayed on
  compatible targets via their normal click path, so the target crosshair,
  read-out, clicked spectrum, and orthogonal slice indicators move together.
  Out-of-data target positions still move the cursor and show "outside data".
* **spectral** — the channel (z) slider is matched by spectral world value
  (velocity/frequency) to the nearest channel of every other cube.

The lock unit is *global*: while enabled, whichever window the user touches is
the *source* and all the others are *targets*. Targets never re-broadcast
(``_applying`` guard + the fact that programmatic limit/slider changes do not
go through the nav-release path), so the source's current state is always the
single source of truth and views cannot drift.

Skip rules (silent): pan/zoom + cursor only between windows whose celestial
system matches (RA/Dec ↔ RA/Dec, Galactic ↔ Galactic — different systems are
skipped) and whose celestial axes are the first two WCS axes; spectral sync
only between windows that have a spectral axis (2-D images are excluded).
"""
from __future__ import annotations

import copy
import math
from typing import List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from takefits.ui.window_registry import WindowRegistry
from takefits.core.wcs_frames import celestial_axis_indices, native_celestial_frame


def celestial_system(wcs) -> Optional[str]:
    """Coarse celestial *system* family for frame-match gating.

    Returns ``"equatorial"`` (ICRS/FK5/FK4 — all "RA/Dec"), ``"galactic"``,
    or the normalized frame name for anything else, or ``None`` when there is
    no usable celestial frame. Equatorial realisations are deliberately
    collapsed together: astropy's high-level ``world_to_pixel`` handles the
    precession between them correctly, matching the user's "RA/Dec is RA/Dec"
    expectation, while equatorial↔galactic stays a non-match (skipped).
    """
    frame = native_celestial_frame(wcs)
    if frame in ("icrs", "fk5", "fk4"):
        return "equatorial"
    if frame:
        return frame
    return None


def _celestial_axes_are_leading(wcs) -> bool:
    """True when the celestial axes are the first two WCS axes.

    The XY plane's data coordinates are pixel indices of axes 0 and 1, so the
    celestial sub-WCS only lines up with the displayed image when the celestial
    axes occupy those slots.
    """
    idx = celestial_axis_indices(wcs)
    return idx is not None and set(idx) == {0, 1}


def _spectral_subwcs(wcs):
    """Return a 1-D spectral sub-WCS, or ``None`` if there is no spectral axis."""
    if wcs is None:
        return None
    try:
        spec = wcs.spectral
    except Exception:
        return None
    try:
        return spec if int(getattr(spec, "naxis", 0)) == 1 else None
    except Exception:
        return None


class WindowSyncManager(QObject):
    """Coordinates WCS-locked pan/zoom, cursor, and spectral sync across windows."""

    # Emitted whenever enabled/sub-toggles change, so each window's menu can
    # refresh its check marks (the lock state is global).
    state_changed = Signal()

    _instance: Optional["WindowSyncManager"] = None

    @classmethod
    def instance(cls) -> "WindowSyncManager":
        if cls._instance is None:
            cls._instance = WindowSyncManager()
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False
        self.sync_pan_zoom = True
        self.sync_cursor = True
        self.sync_spectral = True
        self.sync_color = False  # off by default (data units differ between cubes)

        # Guard against echo: set while pushing state into target windows.
        self._applying = False
        # Color sync applies via a deferred color-history flush, so its echo
        # guard must outlive the synchronous apply (released on the next tick).
        self._applying_color = False

        # Track the open set so we can spot newly opened windows and align them.
        self._known_window_ids = {id(w) for w in self._windows()}

        WindowRegistry.instance().windows_changed.connect(self._on_windows_changed)

    def can_sync(self) -> bool:
        """True when there are at least two windows (the lock is meaningful)."""
        return len(self._windows()) >= 2

    # ------------------------------------------------------------------
    # State / toggles
    # ------------------------------------------------------------------
    def set_enabled(self, value: bool) -> None:
        value = bool(value)
        if value == self.enabled:
            return
        self.enabled = value
        self.state_changed.emit()

    def set_channel(self, name: str, value: bool) -> None:
        attr = {
            "pan_zoom": "sync_pan_zoom",
            "cursor": "sync_cursor",
            "spectral": "sync_spectral",
            "color": "sync_color",
        }.get(name)
        if attr is None:
            return
        value = bool(value)
        if getattr(self, attr) == value:
            return
        setattr(self, attr, value)
        self.state_changed.emit()

    def serialize(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "pan_zoom": bool(self.sync_pan_zoom),
            "cursor": bool(self.sync_cursor),
            "spectral": bool(self.sync_spectral),
            "color": bool(self.sync_color),
        }

    def restore(self, payload: Optional[dict]) -> None:
        if not isinstance(payload, dict):
            return
        if "pan_zoom" in payload:
            self.sync_pan_zoom = bool(payload.get("pan_zoom"))
        if "cursor" in payload:
            self.sync_cursor = bool(payload.get("cursor"))
        if "spectral" in payload:
            self.sync_spectral = bool(payload.get("spectral"))
        if "color" in payload:
            self.sync_color = bool(payload.get("color"))
        self.enabled = bool(payload.get("enabled", self.enabled))
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _windows(self) -> List[object]:
        try:
            return list(WindowRegistry.instance().windows())
        except Exception:
            return []

    def _targets(self, source) -> List[object]:
        return [w for w in self._windows() if w is not source]

    def _on_windows_changed(self) -> None:
        windows = self._windows()
        previously_known = self._known_window_ids
        self._known_window_ids = {id(w) for w in windows}

        # The lock is meaningless with a single window: switch it off so the UI
        # reflects that (the menu also greys the action out).
        if len(windows) < 2:
            if self.enabled:
                self.enabled = False
            self.state_changed.emit()
            return

        # A window opened while the lock is on should join the comparison
        # immediately: align it to an existing (already-synced) window. Deferred
        # so the new window finishes constructing (subwindows, axes) first.
        if self.enabled:
            new_windows = [w for w in windows if id(w) not in previously_known]
            if new_windows:
                reference = next((w for w in windows if w not in new_windows), None)
                if reference is not None:
                    QTimer.singleShot(0, lambda ref=reference: self.sync_now(ref))
        self.state_changed.emit()

    @staticmethod
    def _spatially_compatible(source, target) -> bool:
        try:
            src_w = source.wcs
            tgt_w = target.wcs
        except Exception:
            return False
        if src_w is None or tgt_w is None:
            return False
        if not (_celestial_axes_are_leading(src_w) and _celestial_axes_are_leading(tgt_w)):
            return False
        src_sys = celestial_system(src_w)
        tgt_sys = celestial_system(tgt_w)
        return src_sys is not None and src_sys == tgt_sys

    # ------------------------------------------------------------------
    # pan/zoom
    # ------------------------------------------------------------------
    def broadcast_view(self, source) -> None:
        """Match every compatible target's view to ``source`` (world-locked).

        Syncs both the RA/Dec extent (XY plane) and the spectral extent
        (XZ/ZY z-axis) so the comparison stays aligned in all three world axes.
        """
        if not self.enabled or not self.sync_pan_zoom or self._applying:
            return
        if source is None:
            return
        ax = getattr(source, "ax", None)
        src_wcs = getattr(source, "wcs", None)
        if ax is None or src_wcs is None:
            return
        targets = self._targets(source)
        if not targets:
            return
        try:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            src_cel = src_wcs.celestial
            corners_x = [x0, x1, x1, x0]
            corners_y = [y0, y0, y1, y1]
            sky = src_cel.pixel_to_world(corners_x, corners_y)
        except Exception:
            sky = None
        z_world = self._source_z_world_range(source)

        for target in targets:
            if sky is not None and self._spatially_compatible(source, target):
                try:
                    px, py = target.wcs.celestial.world_to_pixel(sky)
                    new_xlim = (float(min(px)), float(max(px)))
                    new_ylim = (float(min(py)), float(max(py)))
                    self._apply_viewport(target, new_xlim, new_ylim)
                except Exception:
                    pass
            if z_world is not None:
                self._apply_z_range(target, z_world)

    def _source_z_world_range(self, source):
        """Spectral world values of the source's current z (XZ/ZY) extent."""
        if _spectral_subwcs(getattr(source, "wcs", None)) is None:
            return None
        subs = list(getattr(source, "subwindows", None) or [])
        if not subs or subs[0] is None:
            return None
        try:
            z0, z1 = subs[0].ax.get_ylim()  # XZ vertical axis = spectral
            spec = source.wcs.spectral
            return (spec.pixel_to_world(z0), spec.pixel_to_world(z1))
        except Exception:
            return None

    def _apply_viewport(self, target, xlim, ylim) -> None:
        toolbar = getattr(target, "toolbar", None)
        apply_fn = getattr(toolbar, "_apply_world_ranges_from_limits", None)
        if not callable(apply_fn):
            return
        prev_suppress = getattr(target, "_suppress_range_warning", False)
        self._applying = True
        try:
            target._suppress_range_warning = True
            apply_fn(xlim, ylim)
        except Exception:
            pass
        finally:
            target._suppress_range_warning = prev_suppress
            self._applying = False

    def _apply_z_range(self, target, z_world) -> None:
        apply_fn = getattr(target, "apply_synced_z_range", None)
        if not callable(apply_fn):
            return
        tgt_spec = _spectral_subwcs(getattr(target, "wcs", None))
        if tgt_spec is None:
            return
        try:
            tz0 = float(tgt_spec.world_to_pixel(z_world[0]))
            tz1 = float(tgt_spec.world_to_pixel(z_world[1]))
        except Exception:
            return
        if not (math.isfinite(tz0) and math.isfinite(tz1)):
            return
        self._applying = True
        try:
            apply_fn(tz0, tz1)
        except Exception:
            pass
        finally:
            self._applying = False

    # ------------------------------------------------------------------
    # cursor / clicked position (click-driven, full slice update)
    # ------------------------------------------------------------------
    def broadcast_click(self, source, plane: str = "xy", click_x=None, click_y=None) -> None:
        """Replay ``source``'s clicked position on every other window.

        Each compatible target behaves exactly as if it had been clicked at the
        same world position *in the same plane*: an XY click matches RA/Dec, an
        XZ click matches RA + spectral channel, a ZY click matches spectral
        channel + Dec. The target's real crosshair moves and its slices refresh
        (the click is replayed through the target's own ``_perform_click_at``,
        with the target-side terminal print suppressed).

        The work is deferred to the next event-loop tick: replaying the click
        recomputes the target's orthogonal slices via a synchronous
        ``canvas.draw()``, and doing that re-entrantly inside the *source*
        window's mouse-event handler left the target's XZ/ZY crosshair briefly
        unpainted. Running it on a clean tick avoids that flicker.
        """
        if not self.enabled or not self.sync_cursor or self._applying:
            return
        if source is None:
            return
        source_pixels = self._source_click_pixels(source, plane, click_x, click_y)
        if source_pixels is None:
            return
        QTimer.singleShot(0, lambda: self._do_broadcast_click(source, plane, source_pixels))

    def _source_click_pixels(self, source, plane: str, click_x=None, click_y=None):
        """Return source (x, y, z) pixels, preserving raw clicked axes when given."""
        try:
            xpix = float(source._get_shared_xpix())
            ypix = float(source._get_shared_ypix())
            zpix = float(source._get_shared_zpix())
        except Exception:
            return None

        try:
            raw_x = None if click_x is None else float(click_x)
            raw_y = None if click_y is None else float(click_y)
        except Exception:
            raw_x = raw_y = None

        if raw_x is None or raw_y is None:
            return xpix, ypix, zpix

        plane = str(plane or "xy").lower()
        if plane == "xy":
            xpix = raw_x
            ypix = raw_y
        elif plane == "xz":
            xpix = raw_x
            zpix = raw_y
        elif plane == "zy":
            zpix = raw_x
            ypix = raw_y
        return xpix, ypix, zpix

    def _do_broadcast_click(self, source, plane: str, source_pixels) -> None:
        if not self.enabled or not self.sync_cursor or self._applying:
            return
        if source is None or source not in self._windows():
            return
        src_wcs = getattr(source, "wcs", None)
        if src_wcs is None:
            return
        try:
            xpix, ypix, zpix = (float(value) for value in source_pixels)
        except Exception:
            return
        try:
            sky = src_wcs.celestial.pixel_to_world(xpix, ypix)
        except Exception:
            return
        spec_world = None
        src_spec = _spectral_subwcs(src_wcs)
        if src_spec is not None:
            try:
                spec_world = src_spec.pixel_to_world(zpix)
            except Exception:
                spec_world = None

        self._applying = True
        try:
            for target in self._targets(source):
                self._apply_click_to_target(source, target, plane, sky, spec_world)
        finally:
            self._applying = False

    def _apply_click_to_target(self, source, target, plane, sky, spec_world) -> None:
        if not self._spatially_compatible(source, target):
            return
        # Spatial pixels on the target for the clicked sky position.
        try:
            tx, ty = target.wcs.celestial.world_to_pixel(sky)
            tx = float(tx)
            ty = float(ty)
        except Exception:
            return
        # Channel pixel on the target for the clicked spectral position.
        tz = None
        if spec_world is not None:
            tgt_spec = _spectral_subwcs(getattr(target, "wcs", None))
            if tgt_spec is not None:
                try:
                    tz = float(tgt_spec.world_to_pixel(spec_world))
                except Exception:
                    tz = None

        plane = str(plane or "xy")
        plane_viewer, dx, dy = self._resolve_plane_target(target, plane, tx, ty, tz)
        if plane_viewer is None:
            return
        click_fn = getattr(plane_viewer, "_perform_click_at", None)
        if not callable(click_fn):
            return
        if dx is None or dy is None or not (math.isfinite(dx) and math.isfinite(dy)):
            return
        try:
            click_fn(dx, dy, announce=False)
        except Exception:
            pass

    def _resolve_plane_target(self, target, plane, tx, ty, tz):
        """Pick the target's same-plane viewer and its (data_x, data_y)."""
        subwindows = list(getattr(target, "subwindows", []) or [])
        if plane == "xy":
            return target, tx, ty
        if plane == "xz":
            viewer = subwindows[0] if len(subwindows) > 0 else None
            if viewer is None:
                return None, None, None
            return viewer, tx, tz
        if plane == "zy":
            viewer = subwindows[1] if len(subwindows) > 1 else None
            if viewer is None:
                return None, None, None
            return viewer, tz, ty
        return None, None, None

    # ------------------------------------------------------------------
    # Initial sync when the lock is switched on
    # ------------------------------------------------------------------
    def sync_now(self, source) -> None:
        """Immediately push ``source``'s view and clicked position to others."""
        if not self.enabled or source is None:
            return
        if self.sync_pan_zoom:
            self.broadcast_view(source)
        if self.sync_cursor:
            self.broadcast_click(source)
        if self.sync_spectral:
            self.broadcast_spectral(source)
        if self.sync_color:
            self.broadcast_color(source)

    # ------------------------------------------------------------------
    # color scale
    # ------------------------------------------------------------------
    def broadcast_color(self, source) -> None:
        """Copy ``source``'s color-scale settings (cmap/scale/limits) to others.

        Reuses the workspace color serialization. Off by default because the
        intensity limits are in data units, which differ between cubes — it is
        opt-in for genuinely comparable data.
        """
        if not self.enabled or not self.sync_color or self._applying or self._applying_color:
            return
        if source is None:
            return
        capture = getattr(source, "_capture_color_settings_state", None)
        if not callable(capture):
            return
        try:
            state = capture()
        except Exception:
            return
        state = self._main_color_sync_payload(state)
        self._applying_color = True
        self._applying = True
        try:
            for target in self._targets(source):
                restore = getattr(target, "_restore_color_settings_state", None)
                if callable(restore):
                    try:
                        restore(state)
                    except Exception:
                        pass
        finally:
            self._applying = False
        # Release the color guard on the next tick, after any deferred
        # color-history flushes triggered by the restore have run (they check
        # _applying_color before re-broadcasting).
        QTimer.singleShot(0, lambda: setattr(self, "_applying_color", False))

    @staticmethod
    def _main_color_sync_payload(state: dict) -> dict:
        """Keep color sync scoped to the main FITS viewers only."""
        if not isinstance(state, dict):
            return {}
        payload = {
            "schema": state.get("schema", 2),
            "global": {},
            "main_viewers": {},
        }

        global_state = state.get("global")
        main_global = None
        if isinstance(global_state, dict) and isinstance(global_state.get("main"), dict):
            main_global = global_state.get("main")
        elif isinstance(state.get("main"), dict):
            main_global = state.get("main")
        if isinstance(main_global, dict):
            payload["global"]["main"] = copy.deepcopy(main_global)
            payload["main"] = copy.deepcopy(main_global)

        main_panel_settings = state.get("main_panel_settings")
        if isinstance(main_panel_settings, dict):
            payload["main_panel_settings"] = copy.deepcopy(main_panel_settings)

        main_viewers = state.get("main_viewers")
        if isinstance(main_viewers, dict):
            payload["main_viewers"] = copy.deepcopy(main_viewers)

        return payload

    # ------------------------------------------------------------------
    # spectral
    # ------------------------------------------------------------------
    def broadcast_spectral(self, source) -> None:
        if not self.enabled or not self.sync_spectral or self._applying:
            return
        if source is None:
            return
        src_spec = _spectral_subwcs(getattr(source, "wcs", None))
        slider = getattr(source, "slider", None)
        if src_spec is None or slider is None:
            return
        try:
            world = src_spec.pixel_to_world(int(slider.value()))
        except Exception:
            return
        for target in self._targets(source):
            tgt_spec = _spectral_subwcs(getattr(target, "wcs", None))
            tgt_slider = getattr(target, "slider", None)
            if tgt_spec is None or tgt_slider is None:
                continue
            try:
                channel = int(round(float(tgt_spec.world_to_pixel(world))))
            except Exception:
                # Quantity mismatch with no usable conversion -> skip this window.
                continue
            lo = tgt_slider.minimum()
            hi = tgt_slider.maximum()
            channel = max(lo, min(hi, channel))
            if channel == tgt_slider.value():
                continue
            self._applying = True
            try:
                tgt_slider.setValue(channel)
            except Exception:
                pass
            finally:
                self._applying = False

    # ------------------------------------------------------------------
    # spatial slice sliders (XZ -> y, ZY -> x)
    # ------------------------------------------------------------------
    def broadcast_slice(self, source, plane: str) -> None:
        """Sync the spatial slice slider of the XZ (y) / ZY (x) planes.

        Scrolling the XZ slider selects which y row is shown; the ZY slider
        selects which x column. These are spatial selections, so they are
        matched in world coordinates (Dec for XZ, RA for ZY) and applied to the
        corresponding subwindow slider on every compatible target.
        """
        if not self.enabled or not self.sync_cursor or self._applying:
            return
        if source is None or plane not in ("xz", "zy"):
            return
        src_wcs = getattr(source, "wcs", None)
        if src_wcs is None:
            return
        try:
            xpix = float(source._get_shared_xpix())
            ypix = float(source._get_shared_ypix())
            sky = src_wcs.celestial.pixel_to_world(xpix, ypix)
        except Exception:
            return
        # XZ slider moves y (subwindow 0); ZY slider moves x (subwindow 1).
        sub_index = 0 if plane == "xz" else 1
        self._applying = True
        try:
            for target in self._targets(source):
                if not self._spatially_compatible(source, target):
                    continue
                try:
                    tx, ty = target.wcs.celestial.world_to_pixel(sky)
                    pix = float(ty) if plane == "xz" else float(tx)
                except Exception:
                    continue
                self._set_target_slice_slider(target, sub_index, pix)
        finally:
            self._applying = False

    @staticmethod
    def _set_target_slice_slider(target, sub_index, pix) -> None:
        subs = list(getattr(target, "subwindows", None) or [])
        if len(subs) <= sub_index or subs[sub_index] is None:
            return
        slider = getattr(subs[sub_index], "slider", None)
        if slider is None or not math.isfinite(pix):
            return
        value = int(round(pix))
        value = max(slider.minimum(), min(slider.maximum(), value))
        if value == slider.value():
            return
        try:
            slider.setValue(value)
        except Exception:
            pass
