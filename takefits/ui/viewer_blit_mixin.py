"""
Mixin class providing blit/background rendering methods for FITSViewer.

This mixin contains rendering helper methods that operate on ViewerState
and matplotlib objects without creating PyQt widgets.
"""
from takefits.core.contour_manager import ContourManager


class ViewerBlitMixin:
    """
    Mixin providing blit and background management methods.

    Assumes the inheriting class has:
    - self.state (ViewerState instance)
    - self._contour_layer_id (optional, for contour rendering)
    - get_viewer_state(plane) method (from ViewerCoordinatorMixin)
    - _get_plane_chlabel, _get_plane_chval_box methods (from ViewerCoordinatorMixin)
    - _get_shared_xpix, _get_shared_ypix, _get_shared_zpix methods (from ViewerCoordinatorMixin)
    """

    def _invalidate_image_background(self):
        if hasattr(self, 'state') and self.state is not None:
            self.state.image_background = None

    def _ensure_image_background(self, state):
        """Ensure image background is cached for blitting."""
        if state is None or state.canvas is None or state.ax is None or state.im is None:
            return None
        if getattr(state, "image_background", None) is None:
            vis = True
            try:
                vis = state.im.get_visible()
                state.im.set_visible(False)
            except Exception:
                pass
            try:
                state.canvas.draw()
            except Exception:
                state.image_background = None
            else:
                try:
                    state.image_background = state.canvas.copy_from_bbox(state.ax.bbox)
                except Exception:
                    state.image_background = None
            try:
                state.im.set_visible(vis)
            except Exception:
                pass
        return state.image_background

    @staticmethod
    def _suppress_axes_grid(ax):
        """Temporarily hide the WCS coordinate grid; returns coords to restore.

        WCSAxes draws the grid inside ``draw_wcsaxes``. Calling that from a manual
        blit path (to repaint ticks above the image) would re-draw the grid too:
        for the contour-style grid (XZ/ZY spectral planes) every call leaks a new
        ``QuadContourSet`` into ``ax.collections`` -> progressive darkening; for
        the line grid it double-composites the semi-transparent lines. The grid is
        already baked into the cached background, so it must be suppressed here.
        """
        suppressed = []
        if ax is None:
            return suppressed
        coords = []
        seen = set()
        coord_maps = getattr(ax, "_all_coords", None)
        if not coord_maps:
            coord_maps = (getattr(ax, "coords", None),)
        for coord_map in coord_maps:
            if coord_map is None:
                continue
            try:
                candidates = list(coord_map)
            except Exception:
                continue
            for coord in candidates:
                identity = id(coord)
                if identity in seen:
                    continue
                seen.add(identity)
                coords.append(coord)
        for coord in coords:
            kwargs = getattr(coord, "_grid_lines_kwargs", None)
            if isinstance(kwargs, dict) and kwargs.get("visible"):
                kwargs["visible"] = False
                suppressed.append(coord)
        return suppressed

    @staticmethod
    def _restore_axes_grid(suppressed):
        for coord in suppressed or ():
            kwargs = getattr(coord, "_grid_lines_kwargs", None)
            if isinstance(kwargs, dict):
                kwargs["visible"] = True

    def _draw_axis_foreground(self, state, *, include_ticks: bool = True):
        """
        Redraw axis frame/ticks above image artists during manual blit paths.

        WCSAxes does not use the regular Matplotlib xaxis/yaxis artists, so
        prefer draw_wcsaxes(renderer) when available.
        """
        if state is None:
            return
        ax = getattr(state, "ax", None)
        canvas = getattr(state, "canvas", None)
        if ax is None or canvas is None:
            return

        renderer = None
        try:
            renderer = canvas.get_renderer()
        except Exception:
            renderer = None

        drew_wcs_ticks = False
        draw_wcsaxes = getattr(ax, "draw_wcsaxes", None)
        if include_ticks and callable(draw_wcsaxes) and renderer is not None:
            # Keep the grid out of this repaint (it lives in the cached
            # background); see _suppress_axes_grid for why.
            suppressed = self._suppress_axes_grid(ax)
            try:
                draw_wcsaxes(renderer)
                drew_wcs_ticks = True
            except Exception:
                pass
            finally:
                self._restore_axes_grid(suppressed)

        # Ensure the visible frame border is repainted last.
        coords = getattr(ax, "coords", None)
        frame = getattr(coords, "frame", None) if coords is not None else None
        if frame is not None and renderer is not None:
            try:
                frame.draw(renderer)
            except Exception:
                pass

        try:
            for spine in getattr(ax, "spines", {}).values():
                if spine is not None:
                    ax.draw_artist(spine)
        except Exception:
            pass

        if include_ticks and not drew_wcs_ticks:
            for axis_name in ("xaxis", "yaxis"):
                axis = getattr(ax, axis_name, None)
                if axis is None:
                    continue
                try:
                    ax.draw_artist(axis)
                except Exception:
                    continue

    def _pv_overlay_artists_for_background(self, state):
        if state is None or getattr(state, "plane", None) != "xy":
            return []
        viewer = getattr(state, "viewer", None) or self
        owners = []
        for candidate in (
            viewer,
            getattr(viewer, "parent", None),
            self,
            getattr(self, "parent", None),
        ):
            if candidate is not None and all(candidate is not owner for owner in owners):
                owners.append(candidate)

        for owner in owners:
            panel = getattr(owner, "control_panel", None)
            pvd = getattr(panel, "pvd_panel", None) if panel is not None else None
            if pvd is None:
                continue
            if hasattr(pvd, "main_overlay_artists"):
                try:
                    return list(pvd.main_overlay_artists() or [])
                except Exception:
                    return []
            return [
                artist for artist in (
                    getattr(pvd, "arrow_artist", None),
                    *list(getattr(pvd, "width_indicators", []) or []),
                    getattr(pvd, "pos_indicator_on_arrow", None),
                )
                if artist is not None
            ]
        return []

    def _pv_inactive_artists_for_overlay(self, state):
        """Inactive PV-slit artists to repaint above the image during a fast blit.

        Unlike the active slit (kept in the dynamic overlay list), inactive slits
        are normally baked into the cached background. The fast channel-change blit
        repaints the image over that background, so without repainting them here
        the inactive slits would vanish until the next full draw."""
        if state is None or getattr(state, "plane", None) != "xy":
            return []
        viewer = getattr(state, "viewer", None) or self
        owners = []
        for candidate in (
            viewer,
            getattr(viewer, "parent", None),
            self,
            getattr(self, "parent", None),
        ):
            if candidate is not None and all(candidate is not owner for owner in owners):
                owners.append(candidate)

        for owner in owners:
            panel = getattr(owner, "control_panel", None)
            pvd = getattr(panel, "pvd_panel", None) if panel is not None else None
            if pvd is None:
                continue
            getter = getattr(pvd, "inactive_overlay_artists", None)
            if callable(getter):
                try:
                    return list(getter() or [])
                except Exception:
                    return []
            return []
        return []

    def _capture_overlay_background_quick(self, state):
        """
        Lightweight overlay background capture used during drag updates.

        Avoids expensive cross-plane marker redraw side effects while still
        keeping cursor/region/marker artists out of the cached background.
        """
        if state is None or state.canvas is None or state.overlay_ax is None:
            return None

        viewer = state.viewer or self
        region_manager = getattr(viewer, 'region_manager', None)
        marker_manager = getattr(viewer, 'marker_manager', None)

        hidden_regions = []
        hidden_markers = []
        hidden_artists = []

        if region_manager is not None:
            try:
                hidden_regions = region_manager.prepare_for_background_capture()
            except Exception:
                hidden_regions = []
        if marker_manager is not None:
            try:
                hidden_markers = marker_manager.prepare_for_background_capture(state.plane)
            except Exception:
                hidden_markers = []

        for artist in (state.hline, state.vline, getattr(state, "cpoint", None), state.chlabel):
            if artist is None:
                continue
            try:
                if artist.get_visible():
                    artist.set_visible(False)
                    hidden_artists.append(artist)
            except Exception:
                continue
        hpbw = getattr(state, 'hpbw', None)
        hpbw_artist = getattr(hpbw, 'ellipse', None) if hpbw is not None else None
        if hpbw_artist is not None:
            try:
                if hpbw_artist.get_visible():
                    hpbw_artist.set_visible(False)
                    hidden_artists.append(hpbw_artist)
            except Exception:
                pass
        for artist in self._pv_overlay_artists_for_background(state):
            try:
                if artist is not None and artist.get_visible():
                    artist.set_visible(False)
                    hidden_artists.append(artist)
            except Exception:
                continue

        try:
            background = state.canvas.copy_from_bbox(state.overlay_ax.bbox)
        except Exception:
            background = None

        for artist in hidden_artists:
            try:
                artist.set_visible(True)
            except Exception:
                pass

        if region_manager is not None and hidden_regions:
            try:
                region_manager.restore_after_background_capture(hidden_regions)
            except Exception:
                pass
        if marker_manager is not None and hidden_markers:
            try:
                marker_manager.restore_after_background_capture(hidden_markers)
            except Exception:
                pass

        return background

    def _plane_grid_visible(self, state) -> bool:
        """True when the WCS coordinate grid is shown on this plane (TF-404)."""
        viewer = getattr(state, "viewer", None) or self
        dm = getattr(viewer, "displaymap", None)
        return bool(getattr(dm, "grid_visible", False))

    def _fast_blit_image_and_overlay(
        self,
        *,
        include_ticks: bool = True,
        include_colorbar: bool = True,
        quick_overlay_background: bool = False,
    ):
        """Fast blit the image and overlay using cached background."""
        perf_token = None
        if hasattr(self, "_perf_start"):
            perf_token = self._perf_start(f"{getattr(self, 'plane', '?')} _fast_blit_image_and_overlay")
        state = getattr(self, 'state', None)
        if state is None or state.canvas is None or state.ax is None or state.im is None:
            if perf_token and hasattr(self, "_perf_end"):
                self._perf_end(perf_token)
            return
        if self._plane_grid_visible(state):
            # The coordinate grid sits above the image. A fast image blit draws
            # the new image over the cached background and would cover the grid
            # (it vanishes), while repainting it here would leak/duplicate the
            # contour grid. Fall back to a clean full draw so WCSAxes composites
            # the grid over the new image correctly. The grid is opt-in, so the
            # extra cost only applies while it is enabled.
            self._draw_canvas_with_image(state)
            self._invalidate_image_background()
            try:
                state._background = None
                state.image_background = None
            except Exception:
                pass
            if perf_token and hasattr(self, "_perf_end"):
                self._perf_end(perf_token)
            return
        prev_anim = None
        try:
            prev_anim = state.im.get_animated()
            if not prev_anim:
                state.im.set_animated(True)
        except Exception:
            prev_anim = None
        bg = self._ensure_image_background(state)
        if bg is None:
            if prev_anim is not None:
                try:
                    state.im.set_animated(prev_anim)
                except Exception:
                    pass
            if perf_token and hasattr(self, "_perf_end"):
                self._perf_end(perf_token)
            return
        try:
            state.canvas.restore_region(bg)
        except Exception:
            self._invalidate_image_background()
            if prev_anim is not None:
                try:
                    state.im.set_animated(prev_anim)
                except Exception:
                    pass
            if perf_token and hasattr(self, "_perf_end"):
                self._perf_end(perf_token)
            return
        try:
            state.ax.draw_artist(state.im)
        except Exception:
            if prev_anim is not None:
                try:
                    state.im.set_animated(prev_anim)
                except Exception:
                    pass
            if perf_token and hasattr(self, "_perf_end"):
                self._perf_end(perf_token)
            return
        # Draw contour artists after image, before blit
        layer_id = getattr(self, '_contour_layer_id', None)
        if layer_id:
            try:
                manager = ContourManager.instance()
                layer = manager._layers.get(layer_id)
                if layer:
                    for artist in layer.get_generated_artists():
                        if artist and getattr(artist, 'axes', None) is not None:
                            artist.axes.draw_artist(artist)
                    for artist in layer.get_overlay_artists():
                        if artist and getattr(artist, 'axes', None) is not None:
                            artist.axes.draw_artist(artist)
            except Exception:
                pass
        # Repaint inactive PV slits on top of the freshly drawn image; they live
        # in the cached background, which the image above has just covered.
        for artist in self._pv_inactive_artists_for_overlay(state):
            try:
                ax = getattr(artist, "axes", None)
                if ax is not None and artist.get_visible():
                    ax.draw_artist(artist)
            except Exception:
                continue
        self._draw_axis_foreground(state, include_ticks=include_ticks)
        try:
            state.canvas.blit(state.ax.bbox)
        except Exception:
            if prev_anim is not None:
                try:
                    state.im.set_animated(prev_anim)
                except Exception:
                    pass
            if perf_token and hasattr(self, "_perf_end"):
                self._perf_end(perf_token)
            return
        if include_colorbar:
            try:
                blit_colorbar = getattr(self, "_blit_colorbar_foreground_for_state", None)
                if callable(blit_colorbar):
                    blit_colorbar(state, force=False)
            except Exception:
                pass
        try:
            if state.overlay_ax is not None:
                if quick_overlay_background:
                    bg = self._capture_overlay_background_quick(state)
                else:
                    bg = state.copy_overlay_background()
                if bg is not None:
                    state.update_background(bg)
        except Exception:
            pass
        if prev_anim is not None:
            try:
                state.im.set_animated(prev_anim)
            except Exception:
                pass
        if perf_token and hasattr(self, "_perf_end"):
            self._perf_end(perf_token)

    def _draw_canvas_with_image(self, state):
        """Draw canvas ensuring image is not animated during draw."""
        if state is None or state.canvas is None:
            return
        im = getattr(state, "im", None)
        prev_anim = None
        if im is not None:
            try:
                prev_anim = im.get_animated()
                if prev_anim:
                    im.set_animated(False)
            except Exception:
                prev_anim = None
        try:
            state.canvas.draw()
        finally:
            if im is not None and prev_anim is not None:
                try:
                    im.set_animated(prev_anim)
                except Exception:
                    pass

    def _sync_channel_controls(self, viewer, k: int):
        """Synchronize channel controls (slider, label) for a viewer."""
        if viewer is None:
            return
        slider = getattr(viewer, 'slider', None)
        if slider is not None:
            try:
                prev_block = slider.blockSignals(True)
                slider.setValue(int(k))
                slider.blockSignals(prev_block)
            except Exception:
                try:
                    slider.blockSignals(False)
                except Exception:
                    pass
        current_value_label = getattr(viewer, 'current_value_label', None)
        if current_value_label is not None:
            try:
                current_value_label.setText(str(int(k) + 1))
            except Exception:
                pass
        chlabel = viewer._get_plane_chlabel(viewer.plane)
        chval_box = viewer._get_plane_chval_box(viewer.plane)
        if chlabel is None and chval_box is None:
            return
        try:
            if viewer.plane == 'xy':
                xpix = viewer._get_shared_xpix()
                ypix = viewer._get_shared_ypix()
                z = viewer.format_pix.convert_chpix_to_world(viewer.plane, xpix, ypix, k)
            elif viewer.plane == 'xz':
                xpix = viewer._get_shared_xpix()
                zpix = viewer._get_shared_zpix()
                z = viewer.format_pix.convert_chpix_to_world(viewer.plane, xpix, k, zpix)
            elif viewer.plane == 'zy':
                ypix = viewer._get_shared_ypix()
                xpix = viewer._get_shared_xpix()
                z = viewer.format_pix.convert_chpix_to_world(viewer.plane, k, ypix, xpix)
            else:
                return
            z_str = viewer.format_pix.convert_chval_to_world_str(viewer.plane, z)
        except Exception:
            return

        if chlabel is not None:
            try:
                chlabel.set_text("%s" % z_str)
                if not chlabel.get_visible():
                    chlabel.set_visible(True)
            except Exception:
                pass
        if chval_box is not None:
            try:
                chval_box.setText("%s" % z_str)
                chval_box.setCursorPosition(0)
            except Exception:
                pass

    def _refresh_overlay_background(self, plane: str):
        """Refresh overlay background for a plane (alias for _rebuild_overlay_background)."""
        return self._rebuild_overlay_background(plane)

    def _rebuild_overlay_background(self, plane: str):
        """Rebuild and cache the overlay background for a plane."""
        perf_token = None
        if hasattr(self, "_perf_start"):
            perf_token = self._perf_start(f"{getattr(self, 'plane', '?')} _rebuild_overlay_background {plane}")
        state = self.get_viewer_state(plane)
        if state is None or state.canvas is None or state.overlay_ax is None:
            if perf_token and hasattr(self, "_perf_end"):
                self._perf_end(perf_token)
            return None
        viewer = state.viewer or self

        hidden_regions = []
        hidden_markers = []
        region_manager = getattr(viewer, 'region_manager', None)
        marker_manager = getattr(viewer, 'marker_manager', None)
        if region_manager is not None:
            hidden_regions = region_manager.prepare_for_background_capture()
        if marker_manager is not None:
            hidden_markers = marker_manager.prepare_for_background_capture(plane)

        hidden_artists = []
        for artist in (state.hline, state.vline, getattr(state, "cpoint", None), state.chlabel):
            if artist is None:
                continue
            try:
                if artist.get_visible():
                    artist.set_visible(False)
                    hidden_artists.append(artist)
            except Exception:
                continue
        hpbw = getattr(state, 'hpbw', None)
        hpbw_artist = getattr(hpbw, 'ellipse', None) if hpbw is not None else None
        if hpbw_artist is not None:
            try:
                if hpbw_artist.get_visible():
                    hpbw_artist.set_visible(False)
                    hidden_artists.append(hpbw_artist)
            except Exception:
                pass
        for artist in self._pv_overlay_artists_for_background(state):
            try:
                if artist is not None and artist.get_visible():
                    artist.set_visible(False)
                    hidden_artists.append(artist)
            except Exception:
                continue

        im = getattr(state, "im", None)
        prev_anim = None
        if im is not None:
            try:
                prev_anim = im.get_animated()
                if prev_anim:
                    im.set_animated(False)
            except Exception:
                prev_anim = None

        try:
            state.canvas.draw()
            bg = state.canvas.copy_from_bbox(state.overlay_ax.bbox)
        except Exception:
            bg = None

        if im is not None and prev_anim is not None:
            try:
                im.set_animated(prev_anim)
            except Exception:
                pass

        for artist in hidden_artists:
            try:
                artist.set_visible(True)
            except Exception:
                pass

        if region_manager is not None:
            region_manager.restore_after_background_capture(hidden_regions)
        if marker_manager is not None:
            if hidden_markers:
                marker_manager.restore_after_background_capture(hidden_markers)
            marker_manager.draw_markers_for_blit(plane)

        if bg is not None:
            state.update_background(bg)
            try:
                viewer._background = bg
                viewer._background_initialized = True
            except Exception:
                pass
        if perf_token and hasattr(self, "_perf_end"):
            self._perf_end(perf_token)
        return bg

    def _reset_blit_caches(self, viewer):
        """Reset all blit caches for a viewer."""
        if viewer is None:
            return
        state = getattr(viewer, 'state', None)
        if state is not None:
            state.image_background = None
            state._background = None
        try:
            viewer._background = None
        except Exception:
            pass
