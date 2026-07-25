"""
ViewerState: Per-plane state container for FITSViewer instances.

Holds all matplotlib objects, UI elements, crosshairs, and state flags
for a single viewer plane (xy, xz, or zy).
"""
from typing import Optional, Any
from weakref import ref as weakref


class ViewerState:
    """
    Encapsulates all per-plane state for a viewer.
    Each FITSViewer/SubWindow owns one ViewerState instance.
    """

    def __init__(self, plane: str):
        """
        Initialize state for a specific plane.

        Args:
            plane: The plane identifier ('xy', 'xz', or 'zy')
        """
        self.plane = plane

        # Matplotlib objects
        self.canvas: Any = None
        self.fig: Any = None
        self.ax: Any = None
        self.overlay_ax: Any = None
        self.im: Any = None
        self._background: Any = None

        # Crosshair lines
        self.hline: Any = None
        self.vline: Any = None
        self.cpoint: Any = None

        # Colorbar
        self.colorbar: Any = None
        self.cax: Any = None

        # Labels
        self.plabel: Any = None  # Position label (QLabel)
        self.chlabel: Any = None  # Channel label (matplotlib text)
        self.hpbw: Any = None  # Half-power beam width ellipse
        self.cursor_x: Optional[float] = None
        self.cursor_y: Optional[float] = None
        self.image_background: Any = None

        # UI elements
        self.slider: Any = None
        self.chval_box: Any = None
        self.xmin_input: Any = None
        self.xmax_input: Any = None
        self.ymin_input: Any = None
        self.ymax_input: Any = None
        self.zmin_input: Any = None
        self.zmax_input: Any = None

        # Axis metadata
        self.ax_xunit: Optional[str] = None
        self.ax_yunit: Optional[str] = None
        self.ax_xtype: Optional[str] = None
        self.ax_ytype: Optional[str] = None
        self.ax_coord: Any = None  # List of coordinate axes

        # State flags
        self.clicked: bool = False

        # Weak reference to the owning viewer (set by coordinator)
        self._viewer_ref: Optional[weakref] = None

    @property
    def viewer(self):
        """Return the owning viewer if still alive, else None."""
        if self._viewer_ref is None:
            return None
        return self._viewer_ref()

    def set_viewer(self, viewer):
        """Set a weak reference to the owning viewer."""
        self._viewer_ref = weakref(viewer) if viewer is not None else None

    # ----- Convenience property accessors that match Common naming -----

    def update_canvas(self, canvas):
        """Update the canvas reference."""
        self.canvas = canvas

    def update_fig(self, fig):
        """Update the figure reference."""
        self.fig = fig

    def update_ax(self, ax):
        """Update the axes reference."""
        self.ax = ax

    def update_overlay_ax(self, overlay_ax):
        """Update the overlay axes reference."""
        self.overlay_ax = overlay_ax

    def update_im(self, im):
        """Update the image reference."""
        self.im = im

    def update_background(self, background):
        """Update the cached background for blitting."""
        self._background = background

    def update_lines(self, hline, vline, cpoint=None):
        """Update crosshair line references."""
        self.hline = hline
        self.vline = vline
        self.cpoint = cpoint

    def update_colorbar(self, colorbar):
        """Update the colorbar reference."""
        self.colorbar = colorbar

    def update_cax(self, cax):
        """Update the colorbar axes reference."""
        self.cax = cax

    def update_poslabel(self, label):
        """Update the position label reference."""
        self.plabel = label

    def update_chlabel(self, label):
        """Update the channel label reference."""
        self.chlabel = label

    def update_hpbw(self, hpbw):
        """Update the HPBW ellipse reference."""
        self.hpbw = hpbw

    def update_slider(self, slider):
        """Update the slider reference."""
        self.slider = slider

    def update_chval_box(self, textbox):
        """Update the channel value textbox reference."""
        self.chval_box = textbox

    def update_xrange_input(self, xmin_input, xmax_input):
        """Update X-range input field references."""
        self.xmin_input = xmin_input
        self.xmax_input = xmax_input

    def update_yrange_input(self, ymin_input, ymax_input):
        """Update Y-range input field references."""
        self.ymin_input = ymin_input
        self.ymax_input = ymax_input

    def update_zrange_input(self, zmin_input, zmax_input):
        """Update Z-range input field references."""
        self.zmin_input = zmin_input
        self.zmax_input = zmax_input

    def update_ax_units(self, xunit, yunit):
        """Update axis unit metadata."""
        self.ax_xunit = xunit
        self.ax_yunit = yunit

    def update_ax_types(self, xtype, ytype):
        """Update axis type metadata."""
        self.ax_xtype = xtype
        self.ax_ytype = ytype

    def update_ax_coord(self, ax_coord):
        """Update the axis coordinate reference list."""
        self.ax_coord = ax_coord

    def copy_overlay_background(self):
        """
        Capture the overlay background for blitting.
        Returns the captured background or None if not possible.
        """
        if self.canvas is None or self.overlay_ax is None:
            return None

        viewer = self.viewer
        hidden_patches = []
        hidden_markers = []
        hidden_artists = []
        region_manager = None
        marker_manager = None

        if viewer is not None:
            region_manager = getattr(viewer, 'region_manager', None)
            marker_manager = getattr(viewer, 'marker_manager', None)
            if region_manager is not None:
                hidden_patches = region_manager.prepare_for_background_capture()
            if marker_manager is not None:
                hidden_markers = marker_manager.prepare_for_background_capture(self.plane)

        # Hide cursor/label artists so they are not baked into the background.
        for artist in (self.hline, self.vline, self.cpoint, self.chlabel):
            if artist is None:
                continue
            try:
                if artist.get_visible():
                    artist.set_visible(False)
                    hidden_artists.append(artist)
            except Exception:
                continue
        hpbw = getattr(self, 'hpbw', None)
        hpbw_artist = getattr(hpbw, 'ellipse', None) if hpbw is not None else None
        if hpbw_artist is not None:
            try:
                if hpbw_artist.get_visible():
                    hpbw_artist.set_visible(False)
                    hidden_artists.append(hpbw_artist)
            except Exception:
                pass

        background = self.canvas.copy_from_bbox(self.overlay_ax.bbox)

        if region_manager is not None:
            region_manager.restore_after_background_capture(hidden_patches)
            region_manager.draw_regions_for_blit()
        if marker_manager is not None:
            if hidden_markers:
                marker_manager.restore_after_background_capture(hidden_markers)
            marker_manager.draw_markers_for_blit(self.plane)
            if self.plane == 'xy':
                marker_manager.redraw_planes(['xz', 'zy'])

        for artist in hidden_artists:
            try:
                artist.set_visible(True)
            except Exception:
                pass

        return background
