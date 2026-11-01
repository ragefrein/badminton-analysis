"""Integration check of the real Qt app, worker, models and supplied videos."""
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from badminton_vision.ui import MainWindow


def main():
    from badminton_vision.storage import settings_path
    originals = {settings_path(ROOT/name): settings_path(ROOT/name).read_bytes()
                 for name in ("doubles.mp4", "single.mp4") if settings_path(ROOT/name).exists()}
    app = QApplication([])
    window = MainWindow()
    window.show()
    errors = []
    window.worker.error.connect(errors.append)

    def until(predicate, timeout=60):
        deadline = time.monotonic()+timeout
        while not predicate():
            app.processEvents()
            if errors:
                raise AssertionError(errors[-1])
            if time.monotonic() > deadline:
                raise TimeoutError("Desktop state did not arrive")
            time.sleep(.01)
        app.processEvents()

    try:
        until(lambda: window.packet is not None)
        assert len(window.cards) == 4
        assert window.packet["ready"] and not window.packet["confirmed"]
        window.grab().save(str(ROOT/"artifacts"/"desktop-preview.png"))
        name = window.cards[0].name
        name.setFocus()
        name.selectAll()
        QTest.keyClicks(name, "Atlet Satu")
        QTest.keyClick(name, Qt.Key.Key_Tab)
        until(lambda: window.packet["tracks"][0]["name"] == "Atlet Satu")
        assert name.text() == "Atlet Satu", "Spaces must work inside name input"
        print("PASS: editable name including spaces, automatic persistence", flush=True)

        # Exercise pause acknowledgement, coordinate mapping, and manual ID swap.
        other = window.packet["tracks"][1]["detection_index"]
        window.choose_player(1)
        until(lambda: window.canvas.selection == 1)
        detection = window.packet["detections"][other]
        x, y = (detection.box[:2]+detection.box[2:])/2
        rect = window.canvas.image_rect
        QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton,
                         pos=QPoint(int(rect.left()+x*rect.width()), int(rect.top()+y*rect.height())))
        until(lambda: window.packet["corrections"] == 1)
        assert window.packet["tracks"][0]["detection_index"] == other
        assert window.packet["tracks"][0]["name"] == "Atlet Satu"
        print("PASS: click binding swaps physical states and preserves names", flush=True)

        window.play_button.setFocus()
        QTest.mouseClick(window.play_button, Qt.MouseButton.LeftButton)
        until(lambda: window.packet["processed"] >= 8)
        window.worker.submit("pause")
        until(lambda: not window.packet["playing"])
        assert window.packet["confirmed"]
        print("PASS: confirmed tracking and responsive pause", flush=True)
        window.worker.submit("seek", index=960)
        until(lambda: window.packet["index"] == 960)
        assert not window.packet["ready"]
        assert all(t["state"] == "unbound" for t in window.packet["tracks"])
        print("PASS: seek invalidates all physical bindings", flush=True)

        window.open_video(ROOT/"single.mp4")
        until(lambda: window.packet["mode"] == "singles" and not window.busy)
        assert len(window.cards) == 2 and window.packet["ready"]
        window.grab().save(str(ROOT/"artifacts"/"desktop-singles.png"))
        window.worker.submit("play")
        until(lambda: window.packet["processed"] >= 8)
        window.worker.submit("export", folder=str(ROOT/"artifacts"/"desktop-export"))
        until(lambda: "Hasil tersimpan" in window.packet["notice"])
        assert (ROOT/"artifacts"/"desktop-export"/"tracking.csv").exists()
        print("PASS: singles mode and CSV/JSON export", flush=True)
        print("ALL DESKTOP CHECKS PASSED", flush=True)
    finally:
        window.close()
        until(lambda: not window.worker.isRunning(), timeout=30)
        for path, original in originals.items():
            path.write_bytes(original)


if __name__ == "__main__":
    main()
