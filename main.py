#!/usr/bin/env python3

import sys
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication
import matplotlib as mpl
import matplotlib.style as mplstyle
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
from astropy.wcs import FITSFixedWarning

mplstyle.use('fast')
mpl.use('QtAgg')

from ui.main_window import MainWindow
from tools.color_scale import RegisterColor
from logic.fits_loader import FITSWorker  # Import the refactored worker

def main():
    # Specify warnings to ignore
    warnings.simplefilter('ignore', VerifyWarning)
    warnings.simplefilter('ignore', FITSFixedWarning)
    warnings.simplefilter('ignore', AstropyWarning)
    warnings.simplefilter('ignore', AstropyUserWarning)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Takefits v2.beta")
    
    RegisterColor()
    
    # Parse filename from command line arguments
    if len(sys.argv) < 2:
        print("Usage: main.py <FITS filename>")
        sys.exit(1)
    filename = sys.argv[1]
    # Create a QThread and start the FITSWorker
    thread = QThread()
    worker = FITSWorker(filename)
    worker.moveToThread(thread)
    
    thread.started.connect(worker.run)
    
    def on_finished(data, header, wcs, spectral_meta):
        main_win = MainWindow('xy', f"MainWindow: {filename}", data, header, wcs, filename, spectral_meta)
        print("*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*")
        main_win.show()
        thread.quit()
        thread.wait()

    def on_error(title, details):
        thread.quit()
        thread.wait()
        message = details if details else title
        if message:
            print(f"[takefits2] {message}", file=sys.stderr)
        app.exit(1)

    worker.finished.connect(on_finished)
    worker.progress.connect(lambda msg: print(msg))
    worker.error.connect(on_error)
    
    thread.start()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
