"""Shared video overlays for the desktop viewer and annotated clip export."""
import cv2
import numpy as np
from .domain import LIMBS


def bgr(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (5, 3, 1))


def annotate(frame, tracks, detections=(), skeleton=True, rings=True, trails=True, candidates=False, identity_labels=True):
    output = frame.copy()
    h, w = output.shape[:2]
    scale = np.array([w, h])
    thickness = max(2, round(w / 700))
    if candidates:
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = (detection.box * np.array([w, h, w, h])).astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(output, f"D{i + 1}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 2, cv2.LINE_AA)
    for track in tracks:
        if track["state"] != "tracked" or track["foot"] is None:
            continue  # Do not render a guessed skeleton or ring for an uncertain identity.
        color = bgr(track["color"])
        foot = tuple((track["foot"] * scale).astype(int))
        box = track["box"] * np.array([w, h, w, h])
        radius = int(np.clip((box[2] - box[0]) * .65, 18, w * .045))
        if trails and len(track["trail"]) > 1:
            pts = (np.array(track["trail"]) * scale).astype(np.int32).reshape(-1, 1, 2)
            layer = output.copy()
            cv2.polylines(layer, [pts], False, color, thickness, cv2.LINE_AA)
            output = cv2.addWeighted(layer, .55, output, .45, 0)
        if rings:
            # Perspective ellipse, translucent fill, lower rim and highlight = 3D-like ground ring.
            layer = output.copy()
            center = (foot[0], foot[1] + 4)
            axes = (radius, max(6, int(radius * .24)))
            cv2.ellipse(layer, center, axes, 0, 0, 360, color, -1, cv2.LINE_AA)
            output = cv2.addWeighted(layer, .14, output, .86, 0)
            cv2.ellipse(output, (center[0], center[1] + 5), axes, 0, 0, 180, tuple(int(c*.45) for c in color), thickness + 2, cv2.LINE_AA)
            cv2.ellipse(output, center, axes, 0, 0, 360, color, thickness, cv2.LINE_AA)
            cv2.ellipse(output, center, axes, 0, 195, 330, (240, 255, 244), max(1, thickness-1), cv2.LINE_AA)
        if skeleton:
            points = track["keypoints"]
            for a, b in LIMBS:
                if min(points[a, 2], points[b, 2]) >= .35:
                    cv2.line(output, tuple((points[a, :2]*scale).astype(int)), tuple((points[b, :2]*scale).astype(int)), (20, 30, 24), thickness + 2, cv2.LINE_AA)
                    cv2.line(output, tuple((points[a, :2]*scale).astype(int)), tuple((points[b, :2]*scale).astype(int)), color, thickness, cv2.LINE_AA)
            for x, y, confidence in points[5:]:
                if confidence >= .35:
                    cv2.circle(output, (int(x*w), int(y*h)), thickness+1, color, -1, cv2.LINE_AA)
        # Unicode names are painted by Qt in the desktop; OpenCV export uses stable numeric IDs.
        if identity_labels:
            label = f"P{track['id']:02d}"
            x, y = int(box[0]), max(25, int(box[1])-10)
            cv2.rectangle(output, (x-4, y-23), (x+64, y+5), (20, 29, 24), -1)
            cv2.putText(output, label, (x+3, y-2), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    return output
