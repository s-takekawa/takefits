import matplotlib as mpl
import numpy as np

from takefits.core.plotting.beam import beam_pixel_geometry

class AddHPBW:
    initial_config = None
    def __init__(self, ax, header, config):
        self.ax = ax
        self.header = header
        if AddHPBW.initial_config is None:
            AddHPBW.initial_config = config
        self.config = AddHPBW.initial_config
        self.ellipse = None
        
        self.create_beam()

        

    def create_beam(self):
        hpbw_major_pix, hpbw_minor_pix, bpa = self.calculate_hpbw_in_pixels()
        if self.ellipse:
            self.ellipse.remove()
        self.ellipse = mpl.patches.Ellipse(
            (0, 0),
            width=hpbw_major_pix, height=hpbw_minor_pix,
            angle=bpa+90., edgecolor=self.config.get('beam_edgecolor', 'None'), facecolor=self.config.get('beam_facecolor', 'white'), lw=self.config.get('beam_linewidth', 0)
        )
        self.ax.add_patch(self.ellipse)
        self.update_position()

        self.ax.figure.canvas.mpl_connect("draw_event", self.update_position)
        self.ax.figure.canvas.mpl_connect("motion_notify_event", self.update_position)
        # Wrap savefig to temporarily show HPBW ellipse when saving
        fig = self.ax.figure
        orig_savefig = fig.savefig
        def savefig_with_beam(*args, **kwargs):
            self.ellipse.set_visible(True)
            try:
                return orig_savefig(*args, **kwargs)
            finally:
                self.ellipse.set_visible(False)
        fig.savefig = savefig_with_beam

    def refresh_geometry_from_header(self):
        """Refresh beam size/angle from current FITS header without recreating hooks."""
        if self.ellipse is None:
            self.create_beam()
            return
        hpbw_major_pix, hpbw_minor_pix, bpa = self.calculate_hpbw_in_pixels()
        self.ellipse.width = hpbw_major_pix
        self.ellipse.height = hpbw_minor_pix
        self.ellipse.angle = bpa + 90.0
        self.update_position()


    def calculate_hpbw_in_pixels(self):
        """Beam size in pixels, shared with the headless renderer.

        The geometry lives in `core/plotting/beam.py` so the GUI and headless
        exports cannot drift apart. A header with no usable beam yields a
        zero-size ellipse here, which is what this class has always drawn.
        """
        geometry = beam_pixel_geometry(self.header)
        if geometry is None:
            return 0, 0, 0
        return geometry

    def update_position(self, event=None):
        self.relative_x = self.config.get('beam_pos_x', 0.1)
        self.relative_y = self.config.get('beam_pos_y', 0.1)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        center_x = xlim[0] + self.relative_x * (xlim[1] - xlim[0])
        center_y = ylim[0] + self.relative_y * (ylim[1] - ylim[0])
        self.ellipse.set_center((center_x, center_y))
        if not self.ellipse._visible: self.ellipse.set_visible(True)
        self.ellipse.set_facecolor(self.config.get('beam_facecolor'))
        self.ellipse.set_edgecolor(self.config.get('beam_edgecolor'))
        self.ellipse.set_linewidth(self.config.get('beam_linewidth'))
        try:
            self.ax.draw_artist(self.ellipse)
        except AttributeError:
            # skip backends without renderer (e.g. PDF)
            pass
        self.ellipse.set_visible(False)
        
    def update_ax(self, new_ax):
        self.ax = new_ax
        self.create_beam()
