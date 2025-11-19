from PyQt6.QtWidgets import QMainWindow, QTextBrowser, QScrollArea, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette

class ShowHeader(QMainWindow):
    def __init__(self, header):
        super().__init__()
        self.header = header
        self.initUI()

    def initUI(self):
        html = self.buildHeaderHTML()

        self.textBrowser = QTextBrowser()
        self.textBrowser.setHtml(html)
        self.textBrowser.setLineWrapMode(QTextBrowser.LineWrapMode.NoWrap)

        scroll = QScrollArea()
        scroll.setWidget(self.textBrowser)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.setCentralWidget(scroll)
        self.setWindowTitle("FITS Header")


    def buildHeaderHTML(self):
        # Get the application instance to access the palette
        app = QApplication.instance()
        
        # Default color for light mode
        value_color = "#333333" 
        if app:
            palette = app.palette()
            # Check the lightness of the base color to determine if it's a dark theme
            if palette.color(QPalette.ColorRole.Base).lightness() < 128:
                value_color = "#FFFFFF"  # White text for dark backgrounds
        
        keys = [card.keyword for card in self.header.cards if card.keyword != "COMMENT"]
        max_key_len = max((len(k) for k in keys), default=0)
        
        html = "<pre style='font-family: \"Menlo\", \"Fira Code\", \"Source Code Pro\", monospace; white-space: pre;'>"
        #html = "<pre style='font-family: \"Ubuntu Mono\", \"Inconsolata\", \"Courier New\", monospace; white-space: pre;'>"
        for card in self.header.cards:
            if card.keyword == "COMMENT":
                line = f"{card.keyword}  {card.value}"
            else:
                padded_key = card.keyword.ljust(max_key_len + 1)
                # Dynamically set the value color based on the theme
                line = (
                    f"<span style='color: #3366FF;'>{padded_key}</span>"
                    f"= <span style='color: {value_color};'>{card.value}</span>"
                )
                if getattr(card, 'comment', None):
                    line += f" / {card.comment}"
            html += line + "\n"
        html += "</pre>"
        return html
