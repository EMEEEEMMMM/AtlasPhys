from OptionsUI import MainWindow
import sys
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
