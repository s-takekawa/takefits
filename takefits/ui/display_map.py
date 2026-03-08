from takefits.core.common import Common
from matplotlib.axes import Axes
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from takefits.logic.data_tools import fast_nanminmax, estimate_array_nbytes, MEMMAP_THRESHOLD_BYTES

class TransparentOverlayAxes(Axes):
    def contains(self, mouseevent):
        return False, {}

class DisplayMap:
    def __init__(self, data, header, wcs, config):
        if 'CUNIT3' in header:
            if 'm/s' in header['CUNIT3'] and 'km/s' not in header['CUNIT3']:
                if abs(header['CDELT3']) >= 100:
                    self.third_axis_label = 'Velocity  [km/s]'
                else:
                    self.third_axis_label = 'Velocity  [m/s]'
            elif 'km/s' in header['CUNIT3']:
                self.third_axis_label = 'Velocity  [km/s]'
        else: self.third_axis_label = 'Velocity  [km/s]'
        if 'CTYPE3' in header and 'FREQ' in header['CTYPE3']:
            self.third_axis_label = 'Frequency'
            if 'CUNIT3' in header:
                cunit3 = header['CUNIT3']
                self.third_axis_label = self.third_axis_label + f'  [{cunit3}]'

        elif ('CTYPE3' in header and 'LSR' in header['CTYPE3']) or ('SPECSYS' in header and 'LSR' in header['SPECSYS']):
            #self.third_axis_label = 'LSR ' + self.third_axis_label
            self.third_axis_label = 'LSR ' + 'Velocity  [km/s]'

        self.coords_dict = {'glon': 'Galactic Longitude',
               'glat': 'Galactic Latitude',
               'ra': 'Right Ascension',
               'dec': 'Declination',
               'vopt': f'{self.third_axis_label}',
               'vrad': f'{self.third_axis_label}',
               'freq': f'{self.third_axis_label}'
               }
               
        self.config = config
        self.colorscale = config.get('colorscale', 'Rainbow')  # default color pattern
        self.coord_wrap = config.get('coord_wrap', 180)
        self.axislabel_fontsize = config.get('axislabel_fontsize', 14)
        self.axislabel_fontfamily = config.get('axislabel_fontfamily', 'Arial')
        self.axislabel_color = config.get('axislabel_color', 'black')
        self.default_ticks_position = config.get('default_ticks_position', 'btlr')
        self.xticklabel_position = config.get('xticklabel_position', 'b')
        self.yticklabel_position = config.get('yticklabel_position', 'l')
        self.tick_direction = config.get('tick_direction', 'out')
        self.tick_length = config.get('tick_length', 4)  # default: 4
        self.mtick_length = config.get('mtick_length', 2)  # default: 2
        self.tick_width = config.get('tick_width', 1)
        self.tick_labelsize = config.get('tick_labelsize', 10)
        self.tick_color = config.get('tick_color', 'black')
        self.tick_labelcolor = config.get('tick_labelcolor', 'black')
        self.tick_pad_x = config.get('tick_pad_x', 5)
        self.tick_pad_y = config.get('tick_pad_y', 5)

        self.x_mtick_freq = config.get('x_mtick_freq', 5)
        self.y_mtick_freq = config.get('y_mtick_freq', 5)
        self.z_mtick_freq = config.get('z_mtick_freq', 5)
        
        self.tick_xlabelrotation = config.get('tick_xlabelrotation', 0)
        self.tick_ylabelrotation = config.get('tick_ylabelrotation', 0)
        self.fig_background_color = config.get('fig_background_color', '#ececec')
        self.ax_background_color = config.get('ax_background_color', 'white')
        self.bad_color = config.get('bad_color', 'black')
        self.ax_pos_l = config.get('ax_pos_l', 0.15)
        self.ax_pos_r = config.get('ax_pos_r', 0.85)
        self.ax_pos_t = config.get('ax_pos_t', 0.9)
        self.ax_pos_b = config.get('ax_pos_b', 0.12)
        
        self.cbar_pos_x = config.get('cbar_pos_x', 0.9)
        self.cbar_pos_y = config.get('cbar_pos_y', 0.11)
        self.cbar_width = config.get('cbar_width', 0.02)
        self.cbar_height = config.get('cbar_height', 0.77)
        
        self.decimal = config.get('decimal', True)
        
        self.tick_font = config.get('tick_font', 'Arial')
        self.tick_font_weight = config.get('tick_font_weight', 'normal')
        
        self.colorbar_orientation = config.get('colorbar_orientation', 'vertical')
        self.colorbar_tick_color = config.get('colorbar_tick_color', 'black')
        self.colorbar_tick_length = config.get('colorbar_tick_length', 2)
        self.colorbar_mtick_length = config.get('colorbar_mtick_length', 1)
        self.colorbar_tick_width = config.get('colorbar_tick_width', 1)
        self.colorbar_tick_direction = config.get('colorbar_tick_direction', 'out')
        self.colorbar_mtick_freq = config.get('colorbar_mtick_freq', 2)
        self.colorbar_tick_labelleft = config.get('colorbar_tick_labelleft', False)
        self.colorbar_tick_labeltop = config.get('colorbar_tick_labeltop', False)
        self.colorbar_tick_left = config.get('colorbar_tick_left', True)
        self.colorbar_tick_right = config.get('colorbar_tick_right', True)
        self.colorbar_tick_top = config.get('colorbar_tick_top', False)
        self.colorbar_tick_bottom = config.get('colorbar_tick_bottom', True)
        self.colorbar_label = config.get('colorbar_label', None)
        self.colorbar_label_fontsize = config.get('colorbar_label_fontsize', 12)
        self.colorbar_label_color = config.get('colorbar_label_color', 'black')
        self.colorbar_tick_labelcolor = config.get('colorbar_tick_labelcolor', 'black')
        
        self.data = data
        self.wcs = wcs
        self.header = header 

        self.default_cmin, self.default_cmax = self._initial_limits()

        if np.isnan(self.default_cmin) or np.isnan(self.default_cmax):
            self.default_cmin, self.default_cmax = 0.0, 0.0

        self.colorbar = None
        self.cax = None

    def _initial_limits(self):
        """Derive display limits without forcing a full scan on large cubes."""
        datamin = self.header.get('DATAMIN')
        datamax = self.header.get('DATAMAX')
        if datamin is not None and datamax is not None:
            try:
                cmin = float(datamin)
                cmax = float(datamax)
            except (TypeError, ValueError):
                cmin = cmax = np.nan
            else:
                if not np.isfinite(cmin) or not np.isfinite(cmax):
                    cmin = cmax = np.nan
                else:
                    if cmin > cmax:
                        cmin, cmax = cmax, cmin
                    if cmin != cmax:
                        return cmin, cmax

        approx_bytes = estimate_array_nbytes(self.data)
        if approx_bytes and approx_bytes >= MEMMAP_THRESHOLD_BYTES:
            sample = self._subsample_large_array(self.data)
            cmin, cmax = fast_nanminmax(sample)
        else:
            cmin, cmax = fast_nanminmax(self.data)

        if np.isnan(cmin) or np.isnan(cmax):
            # Fallback: try a representative major-plane slice, then give up to (0, 0)
            try:
                slicer = []
                for axis in range(max(self.data.ndim, 2)):
                    if axis < self.data.ndim - 2:
                        slicer.append(0)
                    else:
                        slicer.append(slice(None))
                first_slice = self.data[tuple(slicer[: self.data.ndim])]
            except Exception:
                first_slice = self.data
            cmin, cmax = fast_nanminmax(first_slice)
        if np.isnan(cmin) or np.isnan(cmax):
            cmin = cmax = 0.0
        return cmin, cmax

    def _subsample_large_array(self, array, points_per_axis=24):
        """
        Edge-aware subsampling that keeps the sample size modest while covering the volume.
        """
        if array.ndim == 0:
            return array

        axis_indices = [self._axis_indices(size, points_per_axis) for size in array.shape]
        try:
            mesh = np.ix_(*axis_indices)
            return array[mesh]
        except Exception:
            # As a fallback, use flattened step-sampling.
            flat = array.reshape(-1)
            step = max(1, flat.size // (points_per_axis ** max(array.ndim, 1)))
            return flat[::step]

    @staticmethod
    def _axis_indices(size, max_points):
        if size <= max_points:
            return np.arange(size, dtype=int)
        # Ensure we always include endpoints and central values.
        return np.unique(
            np.round(
                np.linspace(0, size - 1, max_points, dtype=float)
            ).astype(int)
        )
    
    def display(self, fig, plane):
        if self.wcs is not None:
            ndim = self.wcs.naxis
        else:
            ndim = self.data.ndim

        if ndim == 2:
            self.slices = ('x', 'y')
            self.imdata = self.data
            # Check if the header indicates a position-velocity diagram.
            pv = False
            if hasattr(self, 'header') and self.header is not None:
                for i in [1, 2]:
                    ctype = self.header.get(f'CTYPE{i}', '').upper()
                    if 'VRAD' in ctype or 'VEL' in ctype or 'VOPT' in ctype:
                        pv = True
                        break
            if pv:
                aspect = 'auto'
            else:
                aspect = 'equal'

        elif ndim == 3:
            if plane=='xy':
                self.slices = ('x', 'y', 0)
                if self.data.ndim == 3: self.imdata = self.data[0, :, :]
                elif self.data.ndim == 2: self.imdata = self.data
                aspect = 'equal'
            elif plane=='xz':
                self.slices = ('x', 0, 'y')
                self.imdata = self.data[:, 0, :]
                aspect = 'auto'
            elif plane=='zy':
                self.slices = (0, 'y', 'x')
                self.imdata = self.data[:, :, 0].T
                aspect = 'auto'
        elif ndim ==4:
            if plane=='xy':
                self.slices = ('x', 'y', 0, 0)
                self.imdata = self.data[0, 0, :, :]
                aspect = 'equal'
            elif plane=='xz':
                self.slices = ('x', 0, 'y', 0)
                self.imdata = self.data[0, :, 0, :]
                aspect = 'auto'
            elif plane=='zy':
                self.slices = (0, 'y', 'x', 0)
                self.imdata = self.data[0, :, :, 0].T
                aspect = 'auto'

        self.fig = fig
        self.plane = plane
        
        #tick label param
        plt.rcParams['font.family'] = self.tick_font
        plt.rcParams['font.weight'] = self.tick_font_weight
        
        from astropy.visualization.wcsaxes import WCSAxes
        # Calculate axes position as [left, bottom, width, height] in figure coordinates.
        pos = [self.ax_pos_l, self.ax_pos_b, self.ax_pos_r - self.ax_pos_l, self.ax_pos_t - self.ax_pos_b]
        # Create a WCSAxes with slices set.
        self.ax = WCSAxes(self.fig, pos, wcs=self.wcs, slices=self.slices)
        # Add the newly created axes to the figure.
        self.fig.add_axes(self.ax)

        fig.subplots_adjust(left=self.ax_pos_l, right = self.ax_pos_r, bottom = self.ax_pos_b, top= self.ax_pos_t) 
        #self.cax = self.fig.add_axes([self.cbar_pos_x, self.cbar_pos_y, self.cbar_width, self.cbar_height])  # [left, bottom, width, height] in figure coordinates
        
        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider = make_axes_locatable(self.ax)
        # Append axes to the right of ax
        #self.cax = divider.append_axes("right", size=0.1, pad=0.1)
        

        
        cmap = plt.get_cmap(self.colorscale)
        cmap.set_bad(self.bad_color, 1.)
        self.ax.patch.set_zorder(0)
        self.ax.set_axisbelow(False)

        self.im = self.ax.imshow(self.imdata, cmap=cmap, aspect = aspect, origin='lower', zorder=-1)
        self.im.set_clim(self.default_cmin, self.default_cmax)
        

        # Create an overlay axes
        self.overlay_ax = self.fig.add_axes(self.ax.get_position(), sharex=self.ax, sharey=self.ax, zorder=100, frameon=False)
        self.overlay_ax.__class__ = TransparentOverlayAxes
        self.overlay_ax.patch.set_alpha(0)
        self.overlay_ax.set_xticks([])
        self.overlay_ax.set_yticks([])
        self.overlay_ax.set_navigate(False)
        self.overlay_ax.set_picker(False)


        #self.colorbar = self.fig.colorbar(self.im, cax = self.cax, orientation = self.colorbar_orientation )
        #self.colorbar.remove() ##initialize
        self.cax = self.fig.add_axes([self.cbar_pos_x, self.cbar_pos_y, self.cbar_width, self.cbar_height])
        self.cax.set_gid('colorbar')
        self.colorbar = self.fig.colorbar(self.im, cax = self.cax, orientation = self.colorbar_orientation )
        self.cax.tick_params(axis='y', which='both', left=self.colorbar_tick_left, right=self.colorbar_tick_right, labelleft=self.colorbar_tick_labelleft, labelright=(not self.colorbar_tick_labelleft),
                            width=self.colorbar_tick_width, length=self.colorbar_tick_length, color=self.colorbar_tick_color, direction=self.colorbar_tick_direction, labelcolor=self.colorbar_tick_labelcolor) 
        self.cax.tick_params(axis='x', which='both', top=self.colorbar_tick_top, bottom=self.colorbar_tick_bottom, labeltop= self.colorbar_tick_labeltop, labelbottom = (not self.colorbar_tick_labeltop),
                            width=self.colorbar_tick_width, length=self.colorbar_tick_length, color=self.colorbar_tick_color, direction=self.colorbar_tick_direction, labelcolor=self.colorbar_tick_labelcolor) 
        self.colorbar.outline.set_color(self.colorbar_tick_color)
        self.colorbar.outline.set_linewidth(self.colorbar_tick_width)
        self.colorbar.set_label(self.colorbar_label, fontsize=self.colorbar_label_fontsize, color=self.colorbar_label_color)
        self.colorbar.ax.minorticks_on()
        self.colorbar.ax.tick_params(which='minor', length=self.colorbar_mtick_length, color=self.colorbar_tick_color)
        self.colorbar.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(self.colorbar_mtick_freq))
        self.colorbar.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(self.colorbar_mtick_freq))
        

        #main_ticklabels = self.ax.get_xticklabels()
        #main_font_props = main_ticklabels[0].get_fontproperties()
        #for label in self.colorbar.ax.get_yticklabels():
        #    label.set_fontproperties(main_font_props)
        Common.update_colorbar(plane, self.colorbar)
        Common.update_cax(plane, self.cax)
        Common.update_colorbar(plane, self.colorbar)
        Common.update_fig(plane, self.fig)

        
        self.fig.set_facecolor(self.fig_background_color)
        self.ax.set_facecolor(self.ax_background_color)

        self.update_axes_format()

        disp_idx = tuple(True if i else False for i in self.slices)
        for idx in np.where(np.logical_not(disp_idx))[0]:
            self.ax.coords[idx].set_ticklabel_visible(False)
            self.ax.coords[idx].set_ticks_visible(False) 
        ax_xy = []
        for idx in np.where(disp_idx)[0]:
            self.ax.coords[idx].set_ticks_position(self.default_ticks_position)
            self.ax.coords[idx].set_ticklabel(exclude_overlapping=True)
            self.ax.coords[idx].set_ticklabel_visible(True)
            self.ax.coords[idx].display_minor_ticks(True)
            ax_xy.append(self.ax.coords[idx])        
      
        if self.plane == 'zy': ax_xy.reverse()
        ax_xy[0].set_ticklabel(rotation = self.tick_xlabelrotation)
        ax_xy[1].set_ticklabel(rotation = self.tick_ylabelrotation)

        ax_xy[0].set_ticklabel_position(self.xticklabel_position)
        ax_xy[0].set_axislabel_position(self.xticklabel_position)
        ax_xy[1].set_ticklabel_position(self.yticklabel_position)
        ax_xy[1].set_axislabel_position(self.yticklabel_position)
        
        if self.plane =='xy':
            ax_xy[0].set_minor_frequency(self.x_mtick_freq)
            ax_xy[1].set_minor_frequency(self.y_mtick_freq)
        elif self.plane =='xz':
            ax_xy[0].set_minor_frequency(self.x_mtick_freq)
            ax_xy[1].set_minor_frequency(self.z_mtick_freq)
        elif self.plane =='zy':
            ax_xy[0].set_minor_frequency(self.z_mtick_freq)
            ax_xy[1].set_minor_frequency(self.y_mtick_freq)
        
        
        Common.update_ax_coord(plane, ax_xy)
        
        for coord_name, axis_label in self.coords_dict.items():
            if coord_name in self.ax.coords: 
                self.ax.coords[coord_name].set_axislabel(axis_label, fontsize=self.axislabel_fontsize, fontfamily = self.axislabel_fontfamily, color = self.axislabel_color)
        
        self.ax.tick_params(axis='both', which = 'major', direction=self.tick_direction,length=self.tick_length,color=self.tick_color, width = self.tick_width, labelsize = self.tick_labelsize, labelcolor = self.tick_labelcolor)
        #self.ax.set_xticklabels(self.ax.get_xticks(), fontdict={'family': self.tick_font, 'size': self.tick_labelsize})

        self.ax.tick_params(axis='x', pad = self.tick_pad_x)
        self.ax.tick_params(axis='y', pad = self.tick_pad_y)
        self.ax.tick_params(which = 'minor', length=self.mtick_length)
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_zorder(5)
            spine.set_linewidth(self.tick_width)
            spine.set_color(self.tick_color)

        return self.im, self.ax
    
    def update_axes_format(self):
        if not self.wcs:
            return
        self.decimal = self.config.get('decimal', True)
        self.coord_wrap = self.config.get('coord_wrap', 180)

        axis_ctype = self.wcs.wcs.ctype
        axis_unit = [self.ax.coords[i].get_format_unit() for i in range(self.wcs.naxis)]

        for idx, coord in enumerate(self.ax.coords):
            ctype_str = (axis_ctype[idx] or '').upper()
            unit_str = str(axis_unit[idx] or '').lower()

            if unit_str == 'deg':
                if 'RA' in ctype_str:
                    coord.set_coord_type('longitude', coord_wrap=360)
                    coord.set_format_unit('hour', decimal=False) # hms
                elif 'DEC' in ctype_str:
                    coord.set_coord_type('latitude')
                    coord.set_format_unit('deg', decimal=False) # dms

                elif any(keyword in ctype_str for keyword in ['GLON', 'GLAT', 'OFFSET']):
                    if 'GLON' in ctype_str:
                        coord.set_coord_type('longitude', coord_wrap=self.coord_wrap)
                    else: 
                        coord.set_coord_type('latitude')
                    
                    coord.set_format_unit('deg', decimal=self.decimal)
