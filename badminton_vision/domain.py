from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np

COLORS = ("#bcf56a", "#56d5ed", "#ffb86b", "#c4a1ff")
DEFAULT_CORNERS = ((.331, .406), (.698, .410), (.831, .939), (.179, .925))
LIMBS = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4))


@dataclass
class Player:
    id: int
    name: str
    side: str
    color: str


def make_roster(mode: str = "doubles") -> list[Player]:
    count = 2 if mode == "singles" else 4
    return [Player(i + 1, f"Pemain {i + 1}", "far" if i < count // 2 else "near", COLORS[i])
            for i in range(count)]


@dataclass
class Detection:
    box: np.ndarray
    keypoints: np.ndarray
    confidence: float
    feature: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    source: str = "full"

    @property
    def anchor(self) -> np.ndarray:
        """Use the hips for identity motion; ankle visibility changes during lunges."""
        hips = self.keypoints[11:13]
        if np.all(hips[:, 2] >= .35):
            return hips[:, :2].mean(axis=0)
        return np.array([(self.box[0]+self.box[2])/2, self.box[1]+.60*(self.box[3]-self.box[1])])

    @property
    def pose_quality(self) -> float:
        return float(np.mean(self.keypoints[5:, 2]))

    @property
    def foot(self) -> np.ndarray:
        ankles = self.keypoints[15:17]
        visible = ankles[ankles[:, 2] >= .35]
        if len(visible):
            return visible[:, :2].mean(axis=0)
        return np.array([(self.box[0] + self.box[2]) / 2, self.box[3]])


class Court:
    """Perspective calibration, ordered far-left, far-right, near-right, near-left."""

    def __init__(self, corners=DEFAULT_CORNERS):
        points = np.asarray(corners, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Kalibrasi memerlukan empat sudut yang valid.")
        contour = points.reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < .025:
            raise ValueError("Sudut harus membentuk lapangan cembung dan berurutan.")
        self.corners = points
        self.matrix = cv2.getPerspectiveTransform(points, np.float32([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]]))

    def project(self, point) -> np.ndarray:
        return cv2.perspectiveTransform(np.float32(point).reshape(1, 1, 2), self.matrix)[0, 0]

    def accepts(self, detection: Detection) -> bool:
        x, y = self.project(detection.foot)
        return -1.0 <= x <= 7.1 and -1.0 <= y <= 14.4

    def side(self, point) -> str:
        return "far" if self.project(point)[1] < 6.7 else "near"

    def region(self, side):
        """Image-space crop of one court end, including bodies above the floor."""
        y1, y2 = (0, 6.7) if side == "far" else (6.7, 13.4)
        points = np.float32([[[0, y1], [6.1, y1], [6.1, y2], [0, y2]]])
        image_points = cv2.perspectiveTransform(points, np.linalg.inv(self.matrix))[0]
        low, high = image_points.min(axis=0), image_points.max(axis=0)
        return np.clip([low[0]-.08, low[1]-.26, high[0]+.08, high[1]+.06], 0, 1)
