import matplotlib as mpl
import numpy as np

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


    def calculate_hpbw_in_pixels(self):
        cunit1 = self.header.get('CUNIT1', '').strip().lower()
        cunit2 = self.header.get('CUNIT2', '').strip().lower()
        if cunit1 == '' and cunit2 == '': pass
        elif cunit1 != 'deg' or cunit2 != 'deg':
            #raise ValueError("CUNIT1 and CUNIT2 must be in degrees (deg).")
            return 0, 0, 0
    
        bmaj = self.header.get('BMAJ', 0)
        bmin = self.header.get('BMIN', 0)
        bpa = self.header.get('BPA', 0)
        
        try:
            cdelt1 = abs(self.header['CDELT1'])
            cdelt2 = abs(self.header['CDELT2'])
        except: #to be developed!
            print("\033[1;31m\033[1mWarning: Coordinates are defined by CD matrix.\033[0m")
            print("\033[1;31m\033[1mCoordinate axis is tilted to the frame.\033[0m")
            cdelt1 = np.sqrt((self.header['CD1_1'])**2 + (self.header['CD2_1'])**2)
            cdelt2 = np.sqrt((self.header['CD1_2'])**2 + (self.header['CD2_2'])**2)

        #if bmaj is None or bmin is None:
        #    raise ValueError("BMAJ or BMIN is not available in the FITS header.")

        hpbw_major_pix = bmaj / cdelt1
        hpbw_minor_pix = bmin / cdelt2

        return hpbw_major_pix, hpbw_minor_pix, bpa

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

