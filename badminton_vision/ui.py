from __future__ import annotations

from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPolygonF, QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QMenu,
                               QHBoxLayout, QVBoxLayout, QLineEdit, QFileDialog,
                               QSlider, QComboBox, QScrollArea, QSizePolicy)

from .inference import ROOT
from .render import annotate
from .worker import VideoWorker

STYLE = """
QWidget { background: #111713; color: #e9eee8; font-family: 'Segoe UI'; font-size: 12px; }
QMainWindow { background: #111713; }
QLabel { background: transparent; border: none; }
QLabel[kind='muted'] { color: #87968b; }
QFrame#roster { background: #141c16; border-left: 1px solid #293229; }
QFrame#card { background: #1c261e; border: 1px solid #334035; border-radius: 9px; }
QPushButton { background: #202b23; border: 1px solid #3b493d; border-radius: 6px; padding: 9px 12px; font-weight: 500; }
QPushButton:hover { background: #2e3d30; border-color: #6b805d; }
QPushButton:disabled { color: #536057; background: #1b231d; border-color: #2b352d; }
QPushButton#primary { background: #bcf56a; color: #17230e; border-color: #bcf56a; font-weight: 700; }
QPushButton#primary:hover { background: #d0ff8a; }
QPushButton#primary:disabled { background: #4b5f36; color: #a1b38e; }
QPushButton:checked { border-color: #bcf56a; background: #26351f; }
QPushButton#small { font-size: 10px; padding: 5px 8px; }
QLineEdit { background: #131b15; border: 1px solid #3b493d; border-radius: 5px; padding: 7px 8px; selection-background-color: #4d672d; }
QLineEdit:focus { border-color: #bcf56a; }
QComboBox { background: #202b23; border: 1px solid #3b493d; border-radius: 5px; padding: 7px; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #202b23; selection-background-color: #405336; }
QMenu { background: #202b23; border: 1px solid #3b493d; padding: 5px; }
QMenu::item { padding: 7px 22px; }
QMenu::item:selected { background: #405336; }
QSlider::groove:horizontal { height: 4px; background: #344135; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #bcf56a; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -4px 0; border-radius: 6px; background: #d3f9a0; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #182019; width: 6px; }
QScrollBar::handle:vertical { background: #43513e; min-height: 25px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { color: #e9eee8; background: #26352a; border: 1px solid #53684a; padding: 5px; }
"""


def label(text, kind=None, wrap=False):
    widget = QLabel(text)
    if kind:
        widget.setProperty("kind", kind)
    widget.setWordWrap(wrap)
    return widget


def button(text, name=None):
    widget = QPushButton(text)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if name:
        widget.setObjectName(name)
    return widget


def clock_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds//60:02d}:{seconds%60:02d}"


class VideoCanvas(QWidget):
    picked = Signal(int)
    calibrated = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.packet = None
        self.image = None
        self.image_rect = QRectF()
        self.options = dict(skeleton=True, rings=True, trails=True)
        self.selection = None
        self.calibrating = False
        self.corners = []

    def set_packet(self, packet):
        self.packet = packet
        self.redraw()

    def redraw(self):
        if self.packet:
            raw = self.packet["frame"]
            h, w = raw.shape[:2]
            frame = cv2.resize(raw, (min(w, 1280), round(h*min(w, 1280)/w)))
            frame = annotate(frame, self.packet["tracks"], self.packet["detections"],
                             **self.options, candidates=self.selection is not None, identity_labels=False)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090e0b"))
        if self.image is None:
            painter.setPen(QColor("#6f8272"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Memuat video…")
            return
        ratio = min(self.width()/self.image.width(), self.height()/self.image.height())
        w, h = self.image.width()*ratio, self.image.height()*ratio
        self.image_rect = QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)
        painter.drawImage(self.image_rect, self.image)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        for track in self.packet["tracks"]:
            if track["state"] != "tracked" or track["box"] is None:
                continue
            x = self.image_rect.left()+track["box"][0]*w
            y = self.image_rect.top()+track["box"][1]*h-27
            name = painter.fontMetrics().elidedText(f"P{track['id']:02d}  ·  {track['name']}", Qt.TextElideMode.ElideRight, 200)
            text_width = painter.fontMetrics().horizontalAdvance(name)+16
            x = max(self.image_rect.left()+4, min(x, self.image_rect.right()-text_width-4))
            y = max(self.image_rect.top()+4, y)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(16, 28, 18, 225))
            painter.drawRoundedRect(QRectF(x, y, text_width, 23), 4, 4)
            painter.setPen(QColor(track["color"]))
            painter.drawText(QRectF(x+8, y, text_width-8, 23), Qt.AlignmentFlag.AlignVCenter, name)
        if self.calibrating:
            pts = [QPointF(self.image_rect.left()+x*w, self.image_rect.top()+y*h) for x, y in self.corners]
            painter.setPen(QPen(QColor("#bcf56a"), 2))
            if len(pts) > 1:
                painter.drawPolyline(QPolygonF(pts))
            for i, point in enumerate(pts):
                painter.setBrush(QColor("#bcf56a"))
                painter.drawEllipse(point, 5, 5)
                painter.drawText(point+QPointF(9, -8), str(i+1))
        if self.selection is not None or self.calibrating:
            painter.fillRect(QRectF(self.image_rect.left(), self.image_rect.top(), w, 33), QColor(12, 24, 14, 225))
            painter.setPen(QColor("#d8f8b6"))
            message = f"Klik tubuh pemain untuk P{self.selection:02d}  ·  Esc untuk batal" if self.selection else (
                ["1/4 · Klik sudut kiri jauh", "2/4 · Klik sudut kanan jauh", "3/4 · Klik sudut kanan dekat", "4/4 · Klik sudut kiri dekat"][min(3, len(self.corners))])
            painter.drawText(QRectF(self.image_rect.left()+12, self.image_rect.top(), w-20, 33), Qt.AlignmentFlag.AlignVCenter, message)

    def mousePressEvent(self, event):
        if not self.packet or not self.image_rect.contains(event.position()):
            return
        x = (event.position().x()-self.image_rect.left())/self.image_rect.width()
        y = (event.position().y()-self.image_rect.top())/self.image_rect.height()
        if self.calibrating:
            self.corners.append([x, y])
            if len(self.corners) == 4:
                self.calibrating = False
                self.calibrated.emit(self.corners)
            self.update()
        elif self.selection is not None:
            hits = [(i, (d.box[2]-d.box[0])*(d.box[3]-d.box[1])) for i, d in enumerate(self.packet["detections"])
                    if d.box[0] <= x <= d.box[2] and d.box[1] <= y <= d.box[3]]
            if hits:
                self.picked.emit(min(hits, key=lambda item: item[1])[0])


class PlayerCard(QFrame):
    rename = Signal(int, str)
    choose = Signal(int)

    def __init__(self, player_id, color):
        super().__init__()
        self.player_id = player_id
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 7, 11, 7)
        layout.setSpacing(4)
        row = QHBoxLayout()
        badge = label(f"●  P{player_id:02d}")
        badge.setStyleSheet(f"color:{color};font-weight:700;font-size:11px")
        self.state = label("Belum terikat", "muted")
        self.state.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.state.setStyleSheet("font-size:10px;color:#8f9d90")
        row.addWidget(badge)
        row.addWidget(self.state, 1)
        layout.addLayout(row)
        self.name = QLineEdit(f"Pemain {player_id}")
        self.name.setFixedHeight(28)
        self.name.setMaxLength(60)
        self.name.setPlaceholderText("Masukkan nama pemain")
        self.name.setAccessibleName(f"Nama pemain {player_id}")
        self.name.editingFinished.connect(lambda: self.rename.emit(self.player_id, self.name.text()))
        layout.addWidget(self.name)
        row = QHBoxLayout()
        choose = button("Pilih", "small")
        choose.setFixedHeight(22)
        choose.setAccessibleName(f"Pilih pemain {player_id} di video")
        choose.clicked.connect(lambda: self.choose.emit(self.player_id))
        row.addStretch()
        row.addWidget(choose)
        layout.addLayout(row)

    def update_track(self, track):
        if not self.name.hasFocus():
            self.name.setText(track["name"])
        states = {"tracked": ("● Terikat", "#bcf56a"), "uncertain": ("● Perlu cek", "#ffb86b"),
                  "lost": ("○ Tidak terlihat", "#ffb86b"), "unbound": ("○ Pasangkan ulang", "#a4b1a5")}
        text, color = states[track["state"]]
        self.state.setText(text)
        self.state.setStyleSheet(f"font-size:10px;color:{color}")


class MainWindow(QMainWindow):
    def __init__(self, video=None):
        super().__init__()
        # Explicit font loading also supports Qt's offscreen platform on Windows.
        for filename in ("segoeui.ttf", "segoeuib.ttf", "seguisym.ttf"):
            font = Path("C:/Windows/Fonts")/filename
            if font.exists():
                QFontDatabase.addApplicationFont(str(font))
        self.setWindowTitle("Rally Vision 0.2 — Analisis Badminton")
        self.resize(1320, 820)
        self.setMinimumSize(1000, 650)
        self.setStyleSheet(STYLE)
        self.packet = None
        self.cards = []
        self.busy = False
        self.generation = 0
        self.pending_selection = None
        self.pending_calibration = False
        self.worker = VideoWorker()
        self.worker.packet.connect(self.on_packet)
        self.worker.status.connect(self.on_status)
        self.worker.error.connect(self.on_error)
        self.worker.exported.connect(self.on_exported)
        self.build()
        self.worker.start()
        self.play_shortcut = self.shortcut("Space", self.toggle_play)
        self.shortcut("Ctrl+O", self.pick_video)
        self.shortcut("Ctrl+E", self.export)
        self.shortcut("Escape", self.cancel_selection)
        self.step_shortcut = self.shortcut("Right", lambda: self.worker.submit("step") if self.packet and not self.packet["playing"] else None)
        QApplication.instance().focusChanged.connect(self.on_focus_changed)
        initial = Path(video) if video else ROOT / "doubles.mp4"
        if initial.exists():
            QTimer.singleShot(100, lambda: self.open_video(initial))

    def shortcut(self, sequence, callback):
        shortcut = QShortcut(QKeySequence(sequence), self)
        shortcut.activated.connect(callback)
        return shortcut

    def on_focus_changed(self, old, current):
        editing = isinstance(current, QLineEdit)
        self.play_shortcut.setEnabled(not editing)
        self.step_shortcut.setEnabled(not editing)

    def build(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        toolbar = QHBoxLayout()
        title = label("Rally Vision")
        title.setStyleSheet("font-size:17px;font-weight:600")
        toolbar.addWidget(title)
        toolbar.addSpacing(16)
        self.import_button = button("Buka video")
        self.import_button.clicked.connect(self.pick_video)
        toolbar.addWidget(self.import_button)
        self.source_buttons = []
        for filename in ("doubles.mp4", "single.mp4"):
            if not (ROOT/filename).exists():
                continue
            source = button(filename, "small")
            source.setCheckable(True)
            source.clicked.connect(lambda checked=False, f=filename: self.open_video(ROOT/f))
            self.source_buttons.append(source)
            toolbar.addWidget(source)
        toolbar.addStretch()
        self.video_title = label("", "muted")
        toolbar.addWidget(self.video_title)
        options = button("Opsi")
        menu = QMenu(options)
        for text, key in (("Skeleton", "skeleton"), ("Lingkaran kaki", "rings"), ("Jejak gerak", "trails")):
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(lambda checked, k=key: self.change_overlay(k, checked))
        menu.addSeparator()
        self.calibrate_action = menu.addAction("Kalibrasi lapangan", self.calibrate)
        self.calibrate_action.setEnabled(False)
        menu.addAction("Simpan frame", self.save_frame)
        options.setMenu(menu)
        toolbar.addWidget(options)
        self.export_button = button("Ekspor")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        toolbar.addWidget(self.export_button)
        layout.addLayout(toolbar)
        body = QHBoxLayout()
        body.setSpacing(12)
        self.canvas = VideoCanvas()
        self.canvas.picked.connect(self.bind_player)
        self.canvas.calibrated.connect(self.apply_court)
        body.addWidget(self.canvas, 1)
        roster = QFrame()
        roster.setObjectName("roster")
        roster.setFixedWidth(240)
        right = QVBoxLayout(roster)
        right.setContentsMargins(12, 12, 12, 12)
        right.setSpacing(10)
        row = QHBoxLayout()
        row.addWidget(label("Pemain"))
        row.addStretch()
        self.mode = QComboBox()
        self.mode.addItem("Doubles", "doubles")
        self.mode.addItem("Singles", "singles")
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.mode.setAccessibleName("Mode pertandingan")
        row.addWidget(self.mode)
        right.addLayout(row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self.card_layout = QVBoxLayout(container)
        self.card_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.setSpacing(8)
        self.card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(container)
        right.addWidget(scroll, 1)
        self.detected_label = label("", "muted")
        right.addWidget(self.detected_label)
        body.addWidget(roster)
        layout.addLayout(body, 1)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.setAccessibleName("Posisi video")
        self.timeline.sliderPressed.connect(lambda: self.worker.submit("pause"))
        self.timeline.sliderReleased.connect(self.seek)
        layout.addWidget(self.timeline)
        playback = QHBoxLayout()
        self.play_button = button("Mulai", "primary")
        self.play_button.setMinimumWidth(100)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_play)
        playback.addWidget(self.play_button)
        self.time_label = label("00:00 / 00:00", "muted")
        playback.addWidget(self.time_label)
        playback.addStretch()
        self.notice = label("Memuat video…", "muted", True)
        playback.addWidget(self.notice, 1)
        layout.addLayout(playback)

    def open_video(self, path, mode=None):
        if self.busy:
            return
        self.busy = True
        self.generation += 1
        self.cancel_selection()
        self.set_busy(True)
        self.on_status("Memuat video…")
        self.worker.submit("open", path=str(path), mode=mode, generation=self.generation)

    def set_busy(self, busy):
        self.import_button.setEnabled(not busy)
        self.mode.setEnabled(not busy)
        self.play_button.setEnabled(not busy and self.packet is not None)
        self.timeline.setEnabled(not busy and self.packet is not None)
        self.calibrate_action.setEnabled(not busy and self.packet is not None)
        self.export_button.setEnabled(not busy and self.packet is not None)
        for source in self.source_buttons:
            source.setEnabled(not busy)

    def on_status(self, text):
        self.notice.setText(text)

    def on_error(self, text):
        self.busy = False
        self.set_busy(False)
        self.notice.setText(f"Perlu perhatian: {text}")

    def on_exported(self, folder):
        self.notice.setText(f"Hasil tersimpan: {folder}  ·  tracking.csv + session.json")

    def on_packet(self, packet):
        if packet["generation"] != self.generation:
            return
        different = self.packet is None or self.packet["path"] != packet["path"] or self.packet["mode"] != packet["mode"] or (
            [(t["id"], t["side"]) for t in self.packet["tracks"]] != [(t["id"], t["side"]) for t in packet["tracks"]])
        self.packet = packet
        if not packet["playing"]:
            if self.pending_selection is not None:
                self.canvas.selection = self.pending_selection
                self.pending_selection = None
                self.canvas.setCursor(Qt.CursorShape.CrossCursor)
            if self.pending_calibration:
                self.canvas.calibrating = True
                self.pending_calibration = False
                self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        if self.busy:
            self.busy = False
            self.set_busy(False)
        if different:
            while self.card_layout.count():
                item = self.card_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.cards = []
            for track in packet["tracks"]:
                card = PlayerCard(track["id"], track["color"])
                card.rename.connect(lambda player_id, name: self.worker.submit("name", player_id=player_id, name=name))
                card.choose.connect(self.choose_player)
                self.card_layout.addWidget(card)
                self.cards.append(card)
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0 if packet["mode"] == "doubles" else 1)
            self.mode.blockSignals(False)
        for card in self.cards:
            card.update_track(next(t for t in packet["tracks"] if t["id"] == card.player_id))
        self.canvas.set_packet(packet)
        self.video_title.setText(Path(packet["path"]).name)
        self.timeline.setRange(0, max(0, packet["total"]-1))
        if not self.timeline.isSliderDown():
            self.timeline.setValue(packet["index"])
        self.time_label.setText(f"{clock_text(packet['index']/packet['fps'])} / {clock_text(packet['total']/packet['fps'])}")
        seen = sum(t["state"] == "tracked" for t in packet["tracks"])
        self.detected_label.setText(f"{seen}/{len(packet['tracks'])} ID terikat")
        self.play_button.setText("Jeda" if packet["playing"] else "Lanjutkan" if packet["confirmed"] else "Mulai")
        self.play_button.setEnabled(True)
        for source in self.source_buttons:
            source.setChecked(Path(packet["path"]) == ROOT/source.text())
        if self.canvas.selection is None and not self.canvas.calibrating:
            uncertain = [f"P{t['id']:02d}" for t in packet["tracks"] if t["state"] in ("uncertain", "lost")]
            self.notice.setText(packet["notice"] or (f"Periksa {', '.join(uncertain)}" if uncertain else ""))

    def toggle_play(self):
        if not self.packet or self.busy:
            return
        # Space must remain a normal character while typing a player's name.
        if isinstance(QApplication.focusWidget(), QLineEdit):
            return
        self.cancel_selection()
        self.worker.submit("pause" if self.packet["playing"] else "play")

    def choose_player(self, player_id):
        if self.busy or not self.packet:
            return
        self.worker.submit("pause")
        self.canvas.calibrating = False
        self.pending_selection = player_id
        self.notice.setText(f"Klik pemain untuk P{player_id:02d}")

    def bind_player(self, index):
        player_id = self.canvas.selection
        self.cancel_selection()
        self.worker.submit("bind", player_id=player_id, index=index, frame_index=self.packet["index"])

    def cancel_selection(self):
        if not hasattr(self, "canvas"):
            return
        self.canvas.selection = None
        self.pending_selection = None
        self.pending_calibration = False
        self.canvas.calibrating = False
        self.canvas.corners = []
        self.canvas.unsetCursor()
        self.canvas.redraw()

    def calibrate(self):
        if not self.packet:
            return
        self.worker.submit("pause")
        self.canvas.selection = None
        self.pending_calibration = True
        self.canvas.corners = []
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        self.canvas.redraw()
        self.notice.setText("Klik empat sudut garis luar lapangan: kiri jauh → kanan jauh → kanan dekat → kiri dekat. Esc untuk batal.")

    def apply_court(self, corners):
        self.cancel_selection()
        self.worker.submit("court", corners=corners)

    def change_overlay(self, key, checked):
        if hasattr(self, "canvas"):
            self.canvas.options[key] = checked
            self.canvas.redraw()

    def change_mode(self):
        if self.packet:
            self.open_video(self.packet["path"], self.mode.currentData())

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Buka rekaman pertandingan", str(ROOT), "Video (*.mp4 *.mkv *.avi *.mov)")
        if path:
            self.open_video(path)

    def seek(self):
        if self.packet:
            self.cancel_selection()
            self.worker.submit("seek", index=self.timeline.value())
            self.notice.setText("Mencari frame…")

    def export(self):
        if not self.packet or self.busy:
            return
        self.worker.submit("pause")
        folder = QFileDialog.getExistingDirectory(self, "Pilih folder hasil", str(ROOT/"artifacts"))
        if folder:
            target = Path(folder)/f"{Path(self.packet['path']).stem}-{datetime.now():%Y%m%d-%H%M%S}"
            self.worker.submit("export", folder=str(target))

    def save_frame(self):
        if not self.packet:
            return
        folder = ROOT/"artifacts"
        folder.mkdir(exist_ok=True)
        target = folder/f"{Path(self.packet['path']).stem}-frame-{self.packet['index']}.png"
        self.canvas.grab().save(str(target))
        self.notice.setText(f"Frame dengan overlay tersimpan: {target.name}")

    def closeEvent(self, event):
        self.worker.submit("shutdown")
        if self.worker.wait(3000):
            event.accept()
        else:
            # Let current inference finish; never destroy/terminate a live QThread.
            event.ignore()
            self.notice.setText("Menunggu pemrosesan frame selesai sebelum menutup…")
            QTimer.singleShot(1000, self.close)
