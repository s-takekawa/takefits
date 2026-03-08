from PySide6.QtCore import QObject, Signal as pyqtSignal

class Common(QObject):
    xpix = ypix = zpix = 0
    world_x = world_y = world_z = None
    world_x_str = world_y_str = world_z_str = ""
    xy_hline = xy_vline = xz_hline = xz_vline = zy_hline = zy_vline = None
    canvas_xy = canvas_xz = canvas_zy = None
    _background_xy = _background_xz = _background_zy = None
    im_xy = im_xz = im_zy = None
    ax_xy = ax_xz = ax_zy = None
    overlay_ax_xy = overlay_ax_xz = overlay_ax_zy = None
    plabel_xy = plabel_xz = plabel_zy = None
    chlabel_xy = chlabel_xz = chlabel_zy = None
    chval_box_xy = chval_box_xz = chval_box_zy = None 
    ax_xy_xunit = ax_xy_yunit = ax_xz_xunit = ax_xz_yunit = ax_zy_xunit = ax_zy_yunit = None
    ax_xy_xtype = ax_xy_ytype = ax_xz_xtype = ax_xz_ytype = ax_zy_xtype = ax_zy_ytype = None
    slider_xy = slider_xz = slider_zy = None
    clicked_xy = clicked_xz = clicked_zy = False
    colorbar_xy = colorbar_xz = colorbar_zy = None
    cax_xy = cax_xz = cax_zy = None
    fig_xy = fig_xz = fig_zy = None
    ax_coord_xy = ax_coord_xz = ax_coord_zy = None
    xmin_input_xy = xmax_input_xy = xmin_input_xz = xmax_input_xz = xmin_input_zy = xmax_input_zy = None
    ymin_input_xy = ymax_input_xy = ymin_input_xz = ymax_input_xz = ymin_input_zy = ymax_input_zy = None
    zmin_input_xy = zmax_input_xy = zmin_input_xz = zmax_input_xz = zmin_input_zy = zmax_input_zy = None
    integ_cax = []
    integ_colorbar = []
    ch_cax = None
    ch_colorbar = None
    position_updated = pyqtSignal(int, int, int)
    main_window = None
    
    @classmethod
    def update_pix(cls, x, y, z):
        cls.xpix, cls.ypix, cls.zpix = x, y, z
        try:
            from takefits.tools.spectrum import SpecWindow
            if SpecWindow.is_on:
                instance = cls._get_instance()
                instance.position_updated.emit(int(cls.xpix), int(cls.ypix), int(cls.zpix))
        except ImportError:
            pass

    @classmethod
    def _get_instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance
        
    @classmethod
    def update_lines(cls, plane, hline, vline):
        if plane == 'xy':
            cls.xy_hline = hline
            cls.xy_vline = vline
        elif plane == 'xz':
            cls.xz_hline = hline
            cls.xz_vline = vline
        elif plane == 'zy':
            cls.zy_hline = hline
            cls.zy_vline = vline

    @classmethod
    def update_background(cls, plane, background):
        if plane == 'xy':
            cls._background_xy = background
        elif plane == 'xz':
            cls._background_xz = background
        elif plane == 'zy':
            cls._background_zy = background

    @classmethod
    def copy_overlay_background(cls, plane):
        """Copies the overlay background, hiding regions in the main window when needed."""
        canvas = getattr(cls, f'canvas_{plane}', None)
        overlay_ax = getattr(cls, f'overlay_ax_{plane}', None)
        if canvas is None or overlay_ax is None:
            return None

        hidden_patches = []
        hidden_markers = []
        region_manager = None
        marker_manager = None
        viewer = None
        main = getattr(cls, 'main_window', None)
        if main is not None:
            if getattr(main, 'plane', 'xy') == plane or plane == 'xy':
                viewer = main
            else:
                for subwindow in getattr(main, 'subwindows', []):
                    if getattr(subwindow, 'plane', None) == plane:
                        viewer = subwindow
                        break
        if viewer is not None:
            region_manager = getattr(viewer, 'region_manager', None)
            marker_manager = getattr(viewer, 'marker_manager', None)
            if region_manager is not None:
                hidden_patches = region_manager.prepare_for_background_capture()
            if marker_manager is not None:
                hidden_markers = marker_manager.prepare_for_background_capture(plane)

        background = canvas.copy_from_bbox(overlay_ax.bbox)

        if region_manager is not None:
            region_manager.restore_after_background_capture(hidden_patches)
            region_manager.draw_regions_for_blit()
        if marker_manager is not None:
            if hidden_markers:
                marker_manager.restore_after_background_capture(hidden_markers)
            marker_manager.draw_markers_for_blit()
            if plane == 'xy':
                marker_manager.redraw_planes(['xz', 'zy'])

        return background

    @classmethod
    def update_im(cls, plane, im):
        if plane == 'xy': cls.im_xy = im
        elif plane == 'xz': cls.im_xz = im
        elif plane == 'zy': cls.im_zy = im
        
    @classmethod
    def update_ax(cls, plane, ax):
        if plane == 'xy': cls.ax_xy = ax
        elif plane == 'xz': cls.ax_xz = ax
        elif plane == 'zy': cls.ax_zy = ax

    @classmethod
    def update_overlay_ax(cls, plane, ax):
        if plane == 'xy': cls.overlay_ax_xy = ax
        elif plane == 'xz': cls.overlay_ax_xz = ax
        elif plane == 'zy': cls.overlay_ax_zy = ax

    @classmethod
    def update_canvas(cls, plane, canvas):
        if plane == 'xy': cls.canvas_xy = canvas
        elif plane == 'xz': cls.canvas_xz = canvas
        elif plane == 'zy': cls.canvas_zy = canvas

    @classmethod
    def update_poslabel(cls, plane, label):
        if plane == 'xy': cls.plabel_xy = label
        elif plane == 'xz': cls.plabel_xz = label
        elif plane == 'zy': cls.plabel_zy = label
        
    @classmethod
    def update_chlabel(cls, plane, label):
        if plane == 'xy': cls.chlabel_xy = label
        elif plane == 'xz': cls.chlabel_xz = label
        elif plane == 'zy': cls.chlabel_zy = label   

    @classmethod
    def update_hpbw(cls, plane, hpbw):
        if plane == 'xy': cls.hpbw_xy = hpbw
        elif plane == 'xz': cls.hpbw_xz = hpbw
        elif plane == 'zy': cls.hpbw_zy = hpbw
        
    @classmethod
    def update_slider(cls, plane, slider):
        if plane == 'xy': cls.slider_xy = slider
        elif plane == 'xz': cls.slider_xz = slider
        elif plane == 'zy': cls.slider_zy = slider       

    @classmethod
    def update_chval_box(cls, plane, textbox):
        if plane == 'xy': cls.chval_box_xy = textbox
        elif plane == 'xz': cls.chval_box_xz = textbox
        elif plane == 'zy': cls.chval_box_zy = textbox    
    
        
    @classmethod
    def update_ax_units(cls, plane, xunit, yunit):
        if plane == 'xy': cls.ax_xy_xunit, cls.ax_xy_yunit = xunit, yunit
        elif plane == 'xz': cls.ax_xz_xunit, cls.ax_xz_yunit = xunit, yunit
        elif plane == 'zy': cls.ax_zy_xunit, cls.ax_zy_yunit = xunit, yunit

    @classmethod
    def update_ax_types(cls, plane, xtype, ytype):
        if plane == 'xy': cls.ax_xy_xtype, cls.ax_xy_ytype = xtype, ytype
        elif plane == 'xz': cls.ax_xz_xtype, cls.ax_xz_ytype = xtype, ytype
        elif plane == 'zy': cls.ax_zy_xtype, cls.ax_zy_ytype = xtype, ytype

    @classmethod
    def update_world_xyz(cls, x, y, z):
        cls.world_x, cls.world_y, cls.world_z = x, y, z
    
    @classmethod
    def update_world_xyz_str(cls, x, y, z):
        cls.world_x_str, cls.world_y_str, cls.world_z_str = x, y, z
        
    @classmethod
    def update_colorbar(cls, plane, colorbar):
        if plane == 'xy':
            cls.colorbar_xy = colorbar
        elif plane == 'xz':
            cls.colorbar_xz = colorbar
        elif plane == 'zy':
            cls.colorbar_zy = colorbar

    @classmethod
    def update_cax(cls, plane, cax):
        if plane == 'xy':
            cls.cax_xy = cax
        elif plane == 'xz':
            cls.cax_xz = cax
        elif plane == 'zy':
            cls.cax_zy = cax
            
    @classmethod
    def update_fig(cls, plane, fig):
        if plane == 'xy':
            cls.fig_xy = fig
        elif plane == 'xz':
            cls.fig_xz = fig
        elif plane == 'zy':
            cls.fig_zy = fig

    @classmethod
    def update_ax_coord(cls, plane, ax):
        if plane == 'xy':
            cls.ax_coord_xy = ax
        elif plane == 'xz':
            cls.ax_coord_xz = ax
        elif plane == 'zy':
            cls.ax_coord_zy = ax
            
    @classmethod
    def update_xrange_input(cls, plane, xmin_input, xmax_input):
        if plane == 'xy':
            cls.xmin_input_xy = xmin_input
            cls.xmax_input_xy = xmax_input
        elif plane == 'xz':
            cls.xmin_input_xz = xmin_input
            cls.xmax_input_xz = xmax_input
        elif plane == 'zy':
            cls.xmin_input_zy = xmin_input
            cls.xmax_input_zy = xmax_input

    @classmethod
    def update_yrange_input(cls, plane, ymin_input, ymax_input):
        if plane == 'xy':
            cls.ymin_input_xy = ymin_input
            cls.ymax_input_xy = ymax_input
        elif plane == 'xz':
            cls.ymin_input_xz = ymin_input
            cls.ymax_input_xz = ymax_input
        elif plane == 'zy':
            cls.ymin_input_zy = ymin_input
            cls.ymax_input_zy = ymax_input
            
    @classmethod
    def update_zrange_input(cls, plane, zmin_input, zmax_input):
        if plane == 'xy':
            cls.zmin_input_xy = zmin_input
            cls.zmax_input_xy = zmax_input
        elif plane == 'xz':
            cls.zmin_input_xz = zmin_input
            cls.zmax_input_xz = zmax_input
        elif plane == 'zy':
            cls.zmin_input_zy = zmin_input
            cls.zmax_input_zy = zmax_input
    
    @classmethod
    def update_integ_cax(cls, cax):
        cls.integ_cax.append(cax)

    @classmethod
    def remove_integ_cax(cls, cax):
        if cax in cls.integ_cax:
            index = cls.integ_cax.index(cax)
            cls.integ_cax.pop(index)

            
    @classmethod
    def update_integ_colorbar(cls, colorbar):
        cls.integ_colorbar.append(colorbar)

    @classmethod
    def remove_integ_colorbar(cls, colorbar):
        if colorbar in cls.integ_colorbar:
            index = cls.integ_colorbar.index(colorbar)
            cls.integ_colorbar.pop(index)

    @classmethod
    def update_ch_colorbar(cls, colorbar):
        cls.ch_colorbar = colorbar
        
    @classmethod
    def update_ch_cax(cls, cax):
        cls.ch_cax = cax
