from PySide6.QtWidgets import QPushButton, QSizePolicy


def fit_button_to_text(button: QPushButton, *, minimum_width: int = 52, padding: int = 22) -> None:
    text_width = button.fontMetrics().horizontalAdvance(button.text())
    button.setMinimumWidth(max(minimum_width, text_width + padding))
    button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
