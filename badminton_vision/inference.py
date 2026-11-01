from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime"))


class PoseReID:
    """YOLO11 pose + pretrained OSNet/MSMT17 appearance embeddings, all local."""

    def __init__(self, device="auto"):
        import torch
        from ultralytics import YOLO
        from boxmot.reid.backbones.families.osnet import osnet_x0_25

        self.torch = torch
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        self.device = "cuda:0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        pose_path = ROOT / "models" / "yolo11s-pose.pt"
        reid_path = ROOT / "models" / "osnet_x0_25_msmt17.pt"
        for path in (pose_path, reid_path):
            if not path.is_file():
                raise FileNotFoundError(f"Model belum tersedia: {path.name}. Jalankan scripts/setup.ps1.")
        self.pose = YOLO(str(pose_path))
        weights = torch.load(reid_path, map_location="cpu", weights_only=True)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
        self.reid = osnet_x0_25(num_classes=weights["classifier.weight"].shape[0], pretrained=False)
        # A mismatch is fatal: never silently run a random/untrained ReID encoder.
        self.reid.load_state_dict(weights, strict=True)
        self.reid.eval().to(self.device)
        self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)

    def _predict(self, image, frame_shape, offset=(0, 0), source="full"):
        from .domain import Detection
        h, w = frame_shape[:2]
        result = self.pose.predict(image, imgsz=960, conf=.12, iou=.72, classes=[0],
                                   device=self.device, half=self.device.startswith("cuda"), verbose=False)[0]
        detections = []
        if result.keypoints is None:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for box, keypoints, score in zip(boxes, points, scores):
            box = box + np.array([*offset, *offset])
            keypoints = keypoints.copy()
            keypoints[:, :2] += offset
            norm_box = box / np.array([w, h, w, h])
            norm_points = keypoints / np.array([w, h, 1.])
            detection = Detection(norm_box, norm_points, float(score), source=source)
            if box[3] - box[1] < 24:
                continue
            detections.append(detection)
        return detections

    def detect(self, frame, court, expected_per_side=None):
        h, w = frame.shape[:2]
        detections = [d for d in self._predict(frame, frame.shape) if court.accepts(d)]
        detections = merge_detections(detections)
        # A second, larger view of the affected end rescues small/overlapping players.
        # Low-score observations are retained for existing tracks, never used to create IDs.
        if expected_per_side:
            for side in ("far", "near"):
                reliable = [d for d in detections if court.side(d.foot) == side and d.confidence >= .40]
                if len(reliable) >= expected_per_side:
                    continue
                x1, y1, x2, y2 = (court.region(side)*np.array([w, h, w, h])).astype(int)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    recovered = [d for d in self._predict(crop, frame.shape, (x1, y1), "crop")
                                 if court.accepts(d) and court.side(d.foot) == side]
                    detections = merge_detections(detections+recovered)
        crops = []
        valid = []
        for detection in detections:
            box = detection.box*np.array([w, h, w, h])
            x1, y1, x2, y2 = np.round(box).astype(int)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
            crops.append(cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB))
            valid.append(detection)
        if crops:
            torch = self.torch
            batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(self.device).float() / 255.
            with torch.inference_mode():
                features = self.reid((batch - self.mean) / self.std)
                if not isinstance(features, torch.Tensor):
                    raise RuntimeError("Encoder ReID tidak menghasilkan embedding tensor.")
                features = torch.nn.functional.normalize(features, dim=1).cpu().numpy()
            for detection, feature in zip(valid, features):
                detection.feature = feature
        return valid


def same_person(a, b):
    """Merge repeated poses of one body without suppressing overlapping teammates."""
    intersection = np.maximum(0, np.minimum(a.box[2:], b.box[2:])-np.maximum(a.box[:2], b.box[:2])).prod()
    area_a, area_b = np.maximum(0, a.box[2:]-a.box[:2]).prod(), np.maximum(0, b.box[2:]-b.box[:2]).prod()
    if intersection/max(1e-8, min(area_a, area_b)) < .65:
        return False
    height = min(a.box[3]-a.box[1], b.box[3]-b.box[1])
    # During occlusion the model can hallucinate shared hips/shoulders for two
    # people. Distinct visible heads must override that apparent agreement.
    if min(a.keypoints[0, 2], b.keypoints[0, 2]) >= .45:
        head_distance = np.linalg.norm(a.keypoints[0, :2]-b.keypoints[0, :2])
        if head_distance > height*.12:
            return False
    joints = [j for j in (0, 5, 6, 11, 12) if min(a.keypoints[j, 2], b.keypoints[j, 2]) >= .4]
    if len(joints) >= 2:
        distances = np.linalg.norm(a.keypoints[joints, :2]-b.keypoints[joints, :2], axis=1)
        return bool(float(np.median(distances)) < height*.10)
    return intersection/max(1e-8, area_a+area_b-intersection) > .85


def merge_detections(detections):
    result = []
    for detection in sorted(detections, key=lambda d: d.confidence+.25*d.pose_quality, reverse=True):
        if not any(same_person(detection, other) for other in result):
            result.append(detection)
    return result


class CutDetector:
    def __init__(self):
        self.previous = None

    def update(self, frame):
        small = cv2.resize(frame, (64, 36)).astype(np.float32) / 255.
        difference = 0 if self.previous is None else float(np.abs(small - self.previous).mean())
        self.previous = small
        return difference > .19
