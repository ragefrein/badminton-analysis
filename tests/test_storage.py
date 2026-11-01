import csv
import json

from badminton_vision.storage import SessionLog
from tests.test_tracker import tracker_pair


def test_export_keeps_missing_rows_and_latest_names(tmp_path, monkeypatch):
    import badminton_vision.storage as storage
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    tracker, initial = tracker_pair()
    log = SessionLog("test.mp4", 30, "doubles", tracker.court)
    log.append(0, 30, tracker)
    tracker.update(initial[:2]+initial[3:], .1)
    log.append(3, 30, tracker)
    tracker.players[2].name = "Nama Pemain 日本"
    log.export(tmp_path/"export", tracker.players)
    log.close()
    with (tmp_path/"export"/"tracking.csv").open(encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 8
    p3 = [row for row in rows if row["player"] == "3"]
    assert all(row["name"] == "Nama Pemain 日本" for row in p3)
    assert p3[-1]["x"] == "" and p3[-1]["keypoints"] == ""
    assert json.loads((tmp_path/"export"/"session.json").read_text())["processed_frames"] == 2
