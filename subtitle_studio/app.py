from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .resources import resource_path
from .ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app_icon = QIcon(str(resource_path("assets/logo.ico")))
    app.setWindowIcon(app_icon)
    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()
    return app.exec()
