"""Run the desktop app: python main.py [--video doubles.mp4]."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".runtime"))


def main():
    parser = argparse.ArgumentParser(description="Rally Vision desktop")
    parser.add_argument("--video", help="Video yang dibuka saat aplikasi dimulai")
    args = parser.parse_args()
    from PySide6.QtWidgets import QApplication
    from badminton_vision.ui import MainWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Rally Vision")
    app.setOrganizationName("Rally Vision")
    window = MainWindow(args.video)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
