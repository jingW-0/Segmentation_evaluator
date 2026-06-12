import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# Ensure the project package can be imported when running main.py directly.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from gui.app import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
