"""Mouse-interaction delegate for the Polyline PV path.

Per ``docs/dev/PV_PATH_UX_ROADMAP.md`` (Future Mode Notes → Polyline), polyline
interaction is implemented as a delegate rather than being bolted into the
straight-line mouse handlers with ``if path_type == "polyline"`` branches. The
delegate owns the *interaction*; the polyline vertices themselves live on the
``PVdiagram`` (``pv.polyline_vertices``) so they flow through the existing
workspace/recipe/undo serialization.

Locked interaction spec:

- Add phase: left-click empty → append node; a rubber-band preview follows the
  cursor. PV is not recomputed per node. Backspace removes the last node.
- Finalize (確定): double-click or Enter (needs >= 2 nodes); Escape cancels.
  Finalize computes PV once and commits the path (one undo step).
- Edit phase (after finalize): drag a node → move; click a segment → insert a
  node; click a node to select it, then Delete/Backspace removes it; click an
  inactive polyline → activate it for editing; click empty → new path.
  Hit-test order = active node → active segment → inactive path → empty.
  Double-clicking the start/end node re-opens the path to add more nodes from
  that endpoint (start prepends to preserve orientation, end appends); Escape
  cancels the extension and restores the path, double-click/Enter re-finalizes.
"""
from __future__ import annotations

import math


class PolylinePathInteraction:
    def __init__(self, pv):
        self.pv = pv
        self._drag_index = None  # index of the node being dragged (edit phase)
        self._moved = False      # whether the current drag actually moved geometry
        self._extend = None      # None | "start" | "end" while re-extending a path
        self._extend_base = None  # vertices snapshot to restore if extension cancelled

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def _verts(self):
        return self.pv.polyline_vertices

    def _tol(self):
        try:
            return float(self.pv.get_tolerance())
        except Exception:
            return 5.0

    def _node_hit(self, x, y):
        tol = self._tol()
        best_i, best_d = None, tol
        for i, (vx, vy) in enumerate(self._verts):
            d = math.hypot(x - vx, y - vy)
            if d <= best_d:
                best_i, best_d = i, d
        return best_i

    def _endpoint_double_clicked(self, x, y):
        """Return "start"/"end" if (x, y) hits the first/last node, else None."""
        node = self._node_hit(x, y)
        if node is None or len(self._verts) < 2:
            return None
        if node == 0:
            return "start"
        if node == len(self._verts) - 1:
            return "end"
        return None

    def _begin_extend(self, end):
        """Re-open a finished path so the user can add nodes from one endpoint."""
        pv = self.pv
        pv.polyline_finished = False
        pv.polyline_selected_index = None
        pv.polyline_extend_from_start = (end == "start")
        self._extend = end
        self._extend_base = list(self._verts)  # restored if the extension is cancelled
        self._drag_index = None
        self._moved = False
        self._redraw()

    @staticmethod
    def _project_to_segment(x, y, x0, y0, x1, y1):
        """Project (x, y) onto a segment; return (t, px, py, dist) or None."""
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-12:
            return None
        t = ((x - x0) * dx + (y - y0) * dy) / seg_len_sq
        px, py = x0 + t * dx, y0 + t * dy
        return t, px, py, math.hypot(x - px, y - py)

    def _smooth(self):
        geom = self.pv._current_polyline_path_geometry()
        return bool(geom is not None and geom.is_smooth)

    def _curve_points(self):
        """Densified spline curve when smooth, else the raw control vertices."""
        geom = self.pv._current_polyline_path_geometry()
        if geom is not None and geom.is_smooth:
            return [(float(px), float(py)) for (px, py) in geom.effective_vertices]
        return [(float(vx), float(vy)) for (vx, vy) in self._verts]

    def _segment_hit(self, x, y):
        """Return (segment_index, projected_point) if (x, y) is on a segment.

        ``segment_index`` is always a *control* interval, so an inserted node
        becomes a new control point. In smooth mode the distance test runs
        against the densified curve, and the projected point (which lies on the
        old curve) is mapped back to the nearest control interval, so the refit
        spline barely moves until the user drags the new node.
        """
        verts = self._verts
        if len(verts) < 2:
            return None
        tol = self._tol()

        if self._smooth() and len(verts) >= 3:
            # Nearest point on the densified curve. Unlike the straight case we
            # clamp t into [0, 1] instead of skipping segment endpoints, because
            # densified sub-segments share vertices -- skipping endpoints would
            # leave an un-hittable gap at every densified vertex.
            curve = self._curve_points()
            best_pt = None
            best_d = tol
            for i in range(len(curve) - 1):
                x0, y0 = curve[i]
                x1, y1 = curve[i + 1]
                proj = self._project_to_segment(x, y, x0, y0, x1, y1)
                if proj is None:
                    continue
                t, px, py, _d = proj
                t = min(1.0, max(0.0, t))
                px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                d = math.hypot(x - px, y - py)
                if d <= best_d:
                    best_pt = (px, py)
                    best_d = d
            if best_pt is None:
                return None
            seg_index = 0
            seg_best = None
            for i in range(len(verts) - 1):
                proj = self._project_to_segment(
                    best_pt[0], best_pt[1],
                    verts[i][0], verts[i][1], verts[i + 1][0], verts[i + 1][1],
                )
                if proj is None:
                    continue
                d = proj[3]
                if seg_best is None or d < seg_best:
                    seg_best = d
                    seg_index = i
            return (seg_index, best_pt)

        best = None
        best_d = tol
        for i in range(len(verts) - 1):
            proj = self._project_to_segment(
                x, y, verts[i][0], verts[i][1], verts[i + 1][0], verts[i + 1][1]
            )
            if proj is None:
                continue
            t, px, py, d = proj
            if t <= 0.0 or t >= 1.0:
                continue  # endpoints handled by node hit-test
            if d <= best_d:
                best = (i, (px, py))
                best_d = d
        return best

    def _redraw(self, rubber_xy=None):
        self.pv._draw_polyline_overlay(rubber_xy=rubber_xy)

    # ------------------------------------------------------------------
    # mouse events (forwarded from PVdiagram with image-pixel coords)
    # ------------------------------------------------------------------
    def on_press(self, x, y, event):
        pv = self.pv
        if not pv.polyline_finished:
            # --- Add phase ---
            if getattr(event, "dblclick", False):
                # The preceding single click already placed the last node; the
                # double-click just confirms. Finalize if we have a real path.
                if len(self._verts) >= 2:
                    pv.apply_controls()  # dispatches to _apply_polyline (finalize)
                self._extend = None
                return
            # With no nodes placed yet (e.g. just after Clear), a click on an
            # already-placed (inactive) polyline re-selects it for editing rather
            # than starting a brand-new path.
            if not self._verts and pv._activate_path_at_point(x, y, self._tol()):
                grabbed = self._node_hit(x, y)
                if grabbed is not None:
                    pv.polyline_selected_index = grabbed
                    self._drag_index = grabbed
                    self._moved = False
                self._redraw()
                return
            # Ignore a click landing on the node we are growing from
            # (zero-length segment).
            if self._verts:
                ax, ay = self._verts[0] if self._extend == "start" else self._verts[-1]
                if math.hypot(x - ax, y - ay) <= self._tol():
                    return
            # When re-extending from the start node, prepend so the existing
            # path keeps its orientation; otherwise append (default add).
            if self._extend == "start":
                self._verts.insert(0, (float(x), float(y)))
            else:
                self._verts.append((float(x), float(y)))
            self._redraw(rubber_xy=(x, y))
            return

        # --- Edit phase ---
        # Double-clicking an endpoint re-opens the path for adding nodes from
        # that end (start node prepends, end node appends).
        if getattr(event, "dblclick", False):
            end = self._endpoint_double_clicked(x, y)
            if end is not None:
                self._begin_extend(end)
                return
        node = self._node_hit(x, y)
        if node is not None:
            pv.polyline_selected_index = node
            self._drag_index = node
            self._moved = False
            self._redraw()
            return
        seg = self._segment_hit(x, y)
        if seg is not None:
            seg_index, point = seg
            self._verts.insert(seg_index + 1, (float(point[0]), float(point[1])))
            pv.polyline_selected_index = seg_index + 1
            self._drag_index = seg_index + 1
            self._moved = True  # insertion changes geometry → commit on release
            self._redraw()
            return
        # Re-select an inactive polyline under the cursor, then grab a node if hit.
        if pv._activate_path_at_point(x, y, self._tol()):
            grabbed = self._node_hit(x, y)
            if grabbed is not None:
                pv.polyline_selected_index = grabbed
                self._drag_index = grabbed
                self._moved = False
            self._redraw()
            return
        # Empty space → start a brand-new polyline (commit current via begin_new_path).
        pv.begin_new_path()
        pv.polyline_finished = False
        pv.polyline_selected_index = None
        pv.polyline_extend_from_start = False
        self._extend = None
        self._drag_index = None
        self._moved = False
        self._verts.append((float(x), float(y)))
        self._redraw(rubber_xy=(x, y))

    def on_motion(self, x, y, event):
        pv = self.pv
        if not pv.polyline_finished:
            if self._verts:
                self._redraw(rubber_xy=(x, y))
            return
        if self._drag_index is not None and 0 <= self._drag_index < len(self._verts):
            self._verts[self._drag_index] = (float(x), float(y))
            self._moved = True
            self._redraw()
            # Keep the position indicator (and width ticks) tracking the geometry
            # as the node moves, mirroring straight/ellipse live editing.
            pv._update_main_window_marker(pv.last_position_coord)
            pv._request_main_overlay_redraw()

    def on_release(self, event_xy, event):
        pv = self.pv
        if self._drag_index is not None:
            self._drag_index = None
            # Only recompute/record when geometry actually changed; a plain click
            # just selects a node (keeping it selected so Delete can remove it).
            if self._moved and len(self._verts) >= 2:
                pv.apply_controls()
            self._moved = False

    def on_key(self, key):
        """Return True if the key was handled in polyline mode."""
        pv = self.pv
        if key in ("enter", "return"):
            if not pv.polyline_finished and len(self._verts) >= 2:
                pv.apply_controls()
            self._extend = None
            self._extend_base = None
            return True
        if key == "escape":
            if not pv.polyline_finished:
                if self._extend is not None:
                    # Cancel an extension: restore the path as it was, re-commit.
                    if self._extend_base is not None:
                        self._verts[:] = self._extend_base
                    pv.polyline_extend_from_start = False
                    self._extend = None
                    self._extend_base = None
                    self._drag_index = None
                    self._moved = False
                    if len(self._verts) >= 2:
                        pv.apply_controls()
                    else:
                        pv._clear_polyline_path()
                        pv._request_main_overlay_redraw()
                    return True
                # Cancel an in-progress (un-finalized) path.
                pv._clear_polyline_path()
                pv._request_main_overlay_redraw()
                self._drag_index = None
                self._moved = False
                return True
            # Finished path: deselect a selected node first, otherwise delete the
            # whole polyline.
            if pv.polyline_selected_index is not None:
                pv.polyline_selected_index = None
                pv._draw_polyline_overlay()
                pv._request_main_overlay_redraw()
                return True
            pv.clear_arrow()
            return True
        if key in ("delete", "backspace"):
            if not pv.polyline_finished:
                # While drawing: remove the most recently placed node.
                if self._verts:
                    self._verts.pop()
                    pv.polyline_selected_index = None
                    self._redraw()
                return True
            idx = pv.polyline_selected_index
            if idx is not None and 0 <= idx < len(self._verts):
                if len(self._verts) > 2:
                    del self._verts[idx]
                    pv.polyline_selected_index = None
                    pv.apply_controls()  # recompute + one undo step
                else:
                    pv.clear_arrow()  # <2 nodes left → remove the whole path
                return True
            # Finished path with no node selected → delete the whole polyline.
            pv.clear_arrow()
            return True
        return False
