from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

ROOT = Path(__file__).resolve().parent.parent


def settings_path(video):
    path = Path(video).resolve()
    key = hashlib.sha256(f"{path}|{path.stat().st_size}".encode()).hexdigest()[:16]
    return ROOT / "sessions" / f"{key}.json"


def load_settings(video):
    path = settings_path(video)
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (ValueError, OSError):
        return {}


def save_settings(video, mode, players, court):
    path = settings_path(video)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"video": str(video), "mode": mode,
                                    "names": {str(p.id): p.name for p in players},
                                    "corners": court.corners.tolist()}, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class SessionLog:
    """Disk-backed frame log. Only processed frames are exported; missing IDs are explicit."""

    def __init__(self, video, fps, mode, court):
        folder = ROOT / "sessions"
        folder.mkdir(exist_ok=True)
        self.path = folder / f"run-{uuid.uuid4().hex[:12]}.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE observations (frame INTEGER, player INTEGER, timestamp REAL, name TEXT, state TEXT, score REAL, x REAL, y REAL, distance REAL, keypoints TEXT, source TEXT, detection_confidence REAL, PRIMARY KEY(frame, player))")
        self.db.execute("CREATE TABLE events (timestamp REAL, kind TEXT, detail TEXT)")
        self.meta = {"video": str(Path(video).resolve()), "fps": fps, "mode": mode,
                     "created_utc": datetime.now(timezone.utc).isoformat(),
                     "court_corners": court.corners.tolist(), "distance_unit": "estimated_meters",
                     "pose_model": "yolo11s-pose", "reid_model": "osnet_x0_25_msmt17",
                     "tracking_version": "0.2.0", "detection_rescue": "court_end_crop",
                     "scope": "processed_frames_only", "identity_ground_truth_validated": False}
        self.count = 0

    def append(self, frame, fps, tracker):
        for track in tracker.tracks.values():
            observed = track.state == "tracked" and track.detection is not None
            pos = tracker.court.project(track.detection.foot) if observed else (None, None)
            keypoints = json.dumps(track.detection.keypoints.tolist()) if observed else None
            self.db.execute("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (frame, track.player.id, frame / fps, track.player.name, track.state,
                             track.match_score, float(pos[0]) if observed else None,
                             float(pos[1]) if observed else None, track.distance, keypoints,
                             track.detection.source if observed else None,
                             track.detection.confidence if observed else None))
        self.count += 1
        if self.count % 60 == 0:
            self.db.commit()

    def event(self, timestamp, kind, detail):
        self.db.execute("INSERT INTO events VALUES (?,?,?)", (timestamp, kind, detail))

    def export(self, folder, players):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.db.commit()
        # Names are roster metadata; edits must also apply to earlier observations.
        for player in players:
            self.db.execute("UPDATE observations SET name=? WHERE player=?", (player.name, player.id))
        self.db.commit()
        cursor = self.db.execute("SELECT * FROM observations ORDER BY frame,player")
        with (folder / "tracking.csv").open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.writer(output)
            writer.writerow([col[0] for col in cursor.description])
            writer.writerows(cursor)
        summary = dict(self.meta)
        summary["players"] = [vars(player) for player in players]
        summary["processed_frames"] = self.db.execute("SELECT COUNT(DISTINCT frame) FROM observations").fetchone()[0]
        summary["events"] = [dict(timestamp=t, kind=k, detail=d) for t, k, d in self.db.execute("SELECT * FROM events")]
        summary["stats"] = [dict(player=i, observed=seen, total=total, estimated_distance_m=distance)
                            for i, seen, total, distance in self.db.execute(
                                "SELECT player,SUM(state='tracked'),COUNT(*),MAX(distance) FROM observations GROUP BY player")]
        (folder / "session.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return folder

    def close(self):
        self.db.commit()
        self.db.close()
