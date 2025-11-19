from PyQt6.QtWidgets import QFileDialog, QMessageBox
from astropy.io import fits


class SaveFITS:
    def __init__(self, data, header, original_filename, original_header=None):
        self.data = data
        self.header = header
        self.original_filename = original_filename
        self.original_header = original_header

    def save(self, suffix="takefits"):
        new_filename = self.generate_new_filename(suffix)

        filename, _ = QFileDialog.getSaveFileName(
            None, "Save FITS File", new_filename, "FITS Files (*.fits);;All Files (*)")

        #
        if filename:
            """
            if self.original_header:
                new_header = self.header
                original_naxis = self.original_header.get('NAXIS', 0)
                new_naxis = new_header.get('NAXIS', 0)

                for i in range(1, new_naxis + 1):
                    cunit_key = f'CUNIT{i}'
                    # If axis 'i' did not have CUNIT in original, or if original had fewer axes, remove it from new.
                    if i > original_naxis or cunit_key not in self.original_header:
                        if cunit_key in new_header:
                            del new_header[cunit_key]
            """
            
            fits.writeto(filename, self.data, self.header, overwrite=True)
            QMessageBox.information(None, "Save Successful", f"FITS successfully saved as: {filename}")
            print(f"File successfully saved as: {filename}\n")

    def generate_new_filename(self, suffix):
        base_filename = self.original_filename.rsplit('.', 1)[0]
        new_filename = f"{base_filename}.{suffix}.fits"
        return new_filename
