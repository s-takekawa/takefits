from ui.viewer import FITSViewer

class SubWindow(FITSViewer):
    def __init__(self, plane, windowtitle, parent):
        super().__init__(parent.data, parent.header, parent.wcs, parent.filename_path, parent.spectral_metadata)
        self.plane = plane
        self.initUI(plane)
        self.setWindowTitle(windowtitle)
        self.parent = parent 
        self.marker_manager = parent.marker_manager
        
    def closeEvent(self, event):
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.close()
        super().closeEvent(event)
        if self.plane == 'xz':
            self.parent.menu_bar.sub1_action.setChecked(False)
        elif self.plane == 'zy':
            self.parent.menu_bar.sub2_action.setChecked(False)
        
class SubWindow_control:
    def __init__(self):
        subwindow1 = None
        subwindow2 = None
        
    @classmethod
    def update_subwindow(cls, sub1, sub2):
            cls.subwindow1, cls.subwindow2 = sub1, sub2
