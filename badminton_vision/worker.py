from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import time
import traceback
import cv2
from PySide6.QtCore import QThread, Signal

from .domain import Court, DEFAULT_CORNERS, make_roster
from .inference import CutDetector, PoseReID
from .storage import SessionLog, load_settings, save_settings
from .tracker import RosterTracker


class VideoWorker(QThread):
    packet = Signal(object)
    status = Signal(str)
    error = Signal(str)
    exported = Signal(str)

    def __init__(self):
        super().__init__()
        self.commands = Queue()
        self.running = True
        self.playing = False
        self.model = None
        self.capture = None
        self.log = None
        self.frame = None
        self.detections = []
        self.index = 0
        self.processing_fps = 0
        self.notice = ""
        self.generation = 0

    def submit(self, command, **payload):
        self.commands.put((command, payload))

    def run(self):
        try:
            while self.running:
                try:
                    command, args = self.commands.get(timeout=.001 if self.playing else .1)
                    self._command(command, args)
                except Empty:
                    if self.playing:
                        start = time.perf_counter()
                        self._next()
                        # Never skip frames for wall-clock playback speed.
                        self.msleep(max(0, int((1 / self.fps - (time.perf_counter()-start)) * 1000)))
                except Exception as exc:
                    self.playing = False
                    traceback.print_exc()
                    self.error.emit(str(exc))
        finally:
            if self.capture:
                self.capture.release()
            if self.log:
                self.log.close()

    def _command(self, command, args):
        if command == "shutdown":
            self.running = False
            self.playing = False
        elif command == "open":
            self.generation = args.get("generation", self.generation + 1)
            self._open(args["path"], args.get("mode"))
        elif command == "play" and self.frame is not None:
            if not self.tracker.ready():
                self.error.emit("Pasangkan semua ID dulu: klik tombol pilih pemain, lalu klik tubuhnya di video.")
                return
            self.tracker.confirmed = True
            self.playing = True
            self.notice = ""
            self._emit()
        elif command == "pause":
            self.playing = False
            if self.frame is not None:
                self._emit()
        elif command == "step" and self.frame is not None and not self.playing:
            if self.tracker.confirmed:
                self._next()
        elif command == "seek" and self.capture:
            self.playing = False
            self.index = min(max(0, args["index"]), self.total-1)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
            ok, self.frame = self.capture.read()
            if not ok:
                raise RuntimeError("Frame tidak dapat dibaca.")
            self.tracker.invalidate()
            for track in self.tracker.tracks.values():
                track.distance = 0
            self._new_log()
            self.cut = CutDetector()
            self.cut.update(self.frame)
            self.detections = self.model.detect(self.frame, self.court, len(self.players)//2)
            self.notice = "Pasangkan ulang ID pada frame ini."
            self._emit()
        elif command == "bind":
            self.playing = False
            if args.get("frame_index", self.index) != self.index:
                raise ValueError("Frame berubah sebelum identitas dipasangkan. Pilih pemain sekali lagi pada frame yang dijeda.")
            self.tracker.bind(args["player_id"], args["index"], self.detections, self.index/self.fps)
            self.log.event(self.index/self.fps, "manual_bind", f"P{args['player_id']:02d} -> D{args['index']+1}")
            self.notice = "ID diperbarui."
            self._emit()
        elif command == "name":
            for player in self.players:
                if player.id == args["player_id"]:
                    player.name = args["name"].strip()[:60] or f"Pemain {player.id}"
            self._save()
            self._emit()
        elif command == "court":
            self.playing = False
            court = Court(args["corners"])
            self.court = court
            self.tracker.court = court
            self.tracker.invalidate()
            self.detections = self.model.detect(self.frame, court, len(self.players)//2)
            for track in self.tracker.tracks.values():
                track.distance = 0
            self._new_log()
            self._save()
            self.notice = "Kalibrasi tersimpan. Pasangkan ulang ID."
            self._emit()
        elif command == "export":
            self.playing = False
            self.log.meta["manual_corrections"] = self.tracker.corrections
            folder = self.log.export(args["folder"], self.players)
            self.notice = f"Hasil tersimpan: {folder} · tracking.csv + session.json"
            self.exported.emit(str(folder))
            self._emit()

    def _save(self):
        save_settings(self.path, self.mode, self.players, self.court)

    def _new_log(self):
        if self.log:
            self.log.close()
        self.log = SessionLog(self.path, self.fps, self.mode, self.court)

    def _open(self, path, mode=None):
        self.playing = False
        if self.capture:
            self.capture.release()
        self.status.emit("Memuat video…")
        self.path = str(Path(path).resolve())
        settings = load_settings(path)
        self.mode = mode or settings.get("mode", "singles" if "single" in Path(path).stem.lower() else "doubles")
        self.players = make_roster(self.mode)
        for player in self.players:
            player.name = settings.get("names", {}).get(str(player.id), player.name)
        self.court = Court(settings.get("corners", DEFAULT_CORNERS))
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise ValueError("Video tidak dapat dibuka. Pilih file MP4 yang valid.")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.
        self.total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = min(int(30*self.fps), max(0, self.total-1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
        ok, self.frame = self.capture.read()
        if not ok:
            raise ValueError("Video tidak berisi frame yang dapat dibaca.")
        if self.model is None:
            self.model = PoseReID()
        self.detections = self.model.detect(self.frame, self.court, len(self.players)//2)
        self.tracker = RosterTracker(self.players, self.court)
        self.tracker.propose(self.detections, self.index/self.fps)
        self.cut = CutDetector()
        self.cut.update(self.frame)
        self.processing_fps = 0
        self._new_log()
        self._save()
        self.notice = "Periksa nama dan ID, lalu Mulai."
        self._emit()

    def _next(self):
        start = time.perf_counter()
        ok, frame = self.capture.read()
        if not ok:
            self.playing = False
            self.notice = "Video selesai. Hasil sesi siap diekspor."
            self._emit()
            return
        self.index += 1
        self.frame = frame
        self.detections = self.model.detect(frame, self.court, len(self.players)//2)
        if self.cut.update(frame):
            self.playing = False
            self.tracker.invalidate()
            self.log.event(self.index/self.fps, "scene_cut", "Manual identity confirmation required")
            self.notice = "Pergantian kamera terdeteksi. Cari tampilan lapangan dan pasangkan ulang ID."
        else:
            self.tracker.update(self.detections, self.index/self.fps)
            self.log.append(self.index, self.fps, self.tracker)
        duration = time.perf_counter()-start
        current = 1/max(.001, duration)
        self.processing_fps = current if not self.processing_fps else .85*self.processing_fps + .15*current
        self._emit()

    def _emit(self):
        self.packet.emit({"frame": self.frame, "detections": self.detections,
                          "generation": self.generation,
                          "tracks": self.tracker.snapshot(), "index": self.index,
                          "fps": self.fps, "total": self.total, "path": self.path,
                          "playing": self.playing, "confirmed": self.tracker.confirmed,
                          "ready": self.tracker.ready(), "notice": self.notice,
                          "mode": self.mode, "corners": self.court.corners.tolist(),
                          "device": self.model.device, "processing_fps": self.processing_fps,
                          "processed": self.log.count, "corrections": self.tracker.corrections})
