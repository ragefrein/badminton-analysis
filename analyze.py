"""Reproducible video smoke test; initial IDs are positional, not ground truth."""
from pathlib import Path
import argparse
import json
import time
import cv2

from badminton_vision.domain import Court, make_roster
from badminton_vision.inference import PoseReID, CutDetector
from badminton_vision.tracker import RosterTracker
from badminton_vision.render import annotate
from badminton_vision.storage import SessionLog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--mode", choices=("singles", "doubles"), default="doubles")
    parser.add_argument("--start", type=float, default=30)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--output", default="artifacts/smoke")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = PoseReID(args.device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    first = int(args.start*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    court = Court()
    tracker = RosterTracker(make_roster(args.mode), court)
    cut = CutDetector()
    log = SessionLog(args.video, fps, args.mode, court)
    writer = None
    counts = {p.id: 0 for p in tracker.players}
    frames = 0
    scene_cut = None
    start = time.perf_counter()
    try:
        for index in range(first, first+int(args.seconds*fps)):
            ok, frame = cap.read()
            if not ok:
                break
            detections = model.detect(frame, court, len(tracker.players)//2)
            if cut.update(frame):
                scene_cut = index
                log.event(index/fps, "scene_cut", "Stopped; manual confirmation needed")
                break
            if frames == 0:
                tracker.propose(detections, index/fps)
                if not tracker.ready():
                    raise RuntimeError(f"Roster awal tidak lengkap: {len(detections)} deteksi. Pilih waktu lain atau kalibrasi lapangan.")
                tracker.confirmed = True
                log.event(index/fps, "automatic_test_seed", "Initial left/right proposal; not manually verified")
            else:
                tracker.update(detections, index/fps)
            log.append(index, fps, tracker)
            tracks = tracker.snapshot()
            for track in tracks:
                counts[track["id"]] += int(track["state"] == "tracked")
            rendered = annotate(frame, tracks)
            rendered = cv2.resize(rendered, (1280, 720))
            if writer is None:
                writer = cv2.VideoWriter(str(output/"preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
                if not writer.isOpened():
                    raise RuntimeError("Encoder MP4 tidak tersedia.")
            writer.write(rendered)
            if frames == 0 or frames % 60 == 0:
                cv2.imwrite(str(output/f"frame-{index}.jpg"), rendered)
                print(f"frame={index} detections={len(detections)} states={[t['state'] for t in tracks]}", flush=True)
            frames += 1
        log.export(output, tracker.players)
        report = {"frames": frames, "source_start": args.start, "source_seconds": frames/fps,
                  "elapsed_seconds": time.perf_counter()-start, "device": model.device,
                  "observed_fraction": {p: seen/max(1, frames) for p, seen in counts.items()},
                  "scene_cut_frame": scene_cut, "id_switches": "not_measured_without_ground_truth"}
        (output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
        log.close()


if __name__ == "__main__":
    main()
