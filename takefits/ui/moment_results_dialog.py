from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

class MomentResultsDialog(QDialog):
    """
    A non-modal dialog to display text results in a selectable text box.
    It deletes itself upon closing to prevent memory leaks.
    """
    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        # Ensure the dialog is deleted when closed
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Main layout
        layout = QVBoxLayout(self)

        # Text edit area to display the content
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(content)
        # Use a standard system font for a cleaner look
        self.text_edit.setFont(QFont())
        
        layout.addWidget(self.text_edit)

        # OK button to close the dialog
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

        self.resize(500, 400)