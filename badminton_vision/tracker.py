"""Fixed-roster association. Ambiguous observations never update appearance memory.

IDs belong to the roster, never to a detector or left/right screen position.
The global alternative-assignment margin rejects ambiguous crossings.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

from .domain import Court, Detection, Player


@dataclass
class Track:
    player: Player
    detection: Detection | None = None
    detection_index: int | None = None
    last_seen: float = -1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    head_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    gallery: deque = field(default_factory=lambda: deque(maxlen=24))
    trail: deque = field(default_factory=lambda: deque(maxlen=75))
    state: str = "unbound"
    match_score: float = 0
    distance: float = 0
    recovery_anchor: np.ndarray | None = None
    recovery_hits: int = 0
    recovery_time: float = -1


class RosterTracker:
    def __init__(self, players: list[Player], court: Court):
        self.players = players
        self.court = court
        self.tracks = {p.id: Track(p) for p in players}
        self.confirmed = False
        self.corrections = 0

    def propose(self, detections: list[Detection], timestamp: float):
        """Initial proposal only; GUI requires explicit user confirmation to play."""
        for side in ("far", "near"):
            players = [p for p in self.players if p.side == side]
            candidates = [i for i, d in enumerate(detections) if self.court.side(d.foot) == side and d.confidence >= .40]
            candidates = sorted(candidates, key=lambda i: detections[i].confidence, reverse=True)[:len(players)]
            candidates.sort(key=lambda i: detections[i].foot[0])
            for player, index in zip(players, candidates):
                self.bind(player.id, index, detections, timestamp, correction=False)

    def bind(self, player_id, index, detections, timestamp, correction=True):
        detection = detections[index]
        owner = next((t for t in self.tracks.values() if t.detection_index == index
                      and t.state == "tracked" and t.player.id != player_id), None)
        target = self.tracks[player_id]
        if owner:
            # Manual rebinding swaps complete physical states while names/IDs stay fixed.
            a, b = target.player, owner.player
            self.tracks[a.id], self.tracks[b.id] = owner, target
            owner.player, target.player = a, b
            target = owner
        else:
            target.gallery.clear()
        target.detection = detection
        target.player.side = self.court.side(detection.foot)
        if owner:
            for track in self.tracks.values():
                if track.detection is not None:
                    track.player.side = self.court.side(track.detection.foot)
        target.detection_index = index
        target.last_seen = timestamp
        target.velocity = np.zeros(2)
        target.head_velocity = np.zeros(2)
        target.state = "tracked"
        target.match_score = 1
        target.trail.clear()
        target.recovery_hits = 0
        target.trail.append(detection.foot.copy())
        if detection.feature.size:
            target.gallery.append(detection.feature.copy())
        if correction:
            self.corrections += 1

    def invalidate(self):
        """Seek/cut: never reuse old coordinates as evidence in the new scene."""
        self.confirmed = False
        for track in self.tracks.values():
            track.state = "unbound"
            track.detection_index = None
            track.velocity[:] = 0
            track.head_velocity[:] = 0
            track.trail.clear()
            track.recovery_hits = 0

    def ready(self):
        return all(t.state == "tracked" for t in self.tracks.values())

    def _appearance(self, track: Track, detection: Detection):
        if not track.gallery or not detection.feature.size:
            return .25
        distances = 1 - np.asarray(track.gallery) @ detection.feature
        return float(np.mean(np.sort(distances)[:min(3, len(distances))]))

    def _costs(self, detections, timestamp):
        tracks = list(self.tracks.values())
        n, m = len(tracks), len(detections)
        # Independent dummy columns allow every identity to be withheld.
        costs = np.full((n, m + n), .72, dtype=float)
        for row, track in enumerate(tracks):
            if track.detection is None or track.state == "unbound":
                costs[row, :m] = 100
                continue
            elapsed = max(.001, timestamp - track.last_seen)
            predicted = track.detection.anchor + track.velocity * min(elapsed, .5)
            for col, detection in enumerate(detections):
                foot = detection.foot
                court_pos = self.court.project(foot)
                # Do not carry identities across ends; an end change requires rebinding.
                wrong_end = (track.player.side == "far" and court_pos[1] > 7.4) or (
                    track.player.side == "near" and court_pos[1] < 6.0)
                motion = float(np.linalg.norm(detection.anchor - predicted))
                old_head, new_head = track.detection.keypoints[0], detection.keypoints[0]
                if min(old_head[2], new_head[2]) >= .45:
                    head_prediction = old_head[:2]+track.head_velocity*min(elapsed, .5)
                    head_motion = float(np.linalg.norm(new_head[:2]-head_prediction))
                    motion = .8*head_motion+.2*motion
                appearance = self._appearance(track, detection)
                height = track.detection.box[3]-track.detection.box[1]
                max_motion = min(.30, max(.025, height*.24) + elapsed * .35)
                if wrong_end or motion > max_motion or appearance > .55:
                    costs[row, col] = 100
                    continue
                if elapsed > 1.2 and (appearance > .32 or not track.gallery):
                    costs[row, col] = 100
                    continue
                old = track.detection.box
                size_a = max(1e-5, (old[2] - old[0]) * (old[3] - old[1]))
                size_b = max(1e-5, (detection.box[2] - detection.box[0]) * (detection.box[3] - detection.box[1]))
                size_cost = min(1., abs(np.log(size_b / size_a)))
                costs[row, col] = .52 * motion / max_motion + .40 * appearance / .55 + .08 * size_cost
        return costs

    @staticmethod
    def _assign(costs, minimum_margin=.075):
        n, columns = costs.shape
        m = columns-n
        rows, cols = linear_sum_assignment(costs)
        best_total = costs[rows, cols].sum()
        assignments = dict(zip(rows, cols))
        accepted = {}
        for row, col in assignments.items():
            if col >= m or costs[row, col] >= .68:
                continue
            alternative = costs.copy()
            alternative[row, col] = 100
            ar, ac = linear_sum_assignment(alternative)
            margin = float(alternative[ar, ac].sum() - best_total)
            # Small global margins indicate equally plausible identity assignments.
            if margin < minimum_margin:
                continue
            accepted[row] = (col, margin)
        return accepted

    def update(self, detections: list[Detection], timestamp: float):
        tracks = list(self.tracks.values())
        costs = self._costs(detections, timestamp)
        high = [i for i, d in enumerate(detections) if d.confidence >= .40]
        first = np.concatenate([costs[:, high], costs[:, len(detections):]], axis=1)
        accepted = {row: (high[col], margin) for row, (col, margin) in self._assign(first).items()}
        missing = [row for row in range(len(tracks)) if row not in accepted]
        used = [detections[col] for col, _ in accepted.values()]
        low = [i for i, d in enumerate(detections) if .12 <= d.confidence < .40 and not any(
            _iou(d.box, other.box) > .50 and np.linalg.norm(d.anchor-other.anchor) < .4*(other.box[3]-other.box[1])
            for other in used)]
        if missing and low:
            second = np.concatenate([costs[np.ix_(missing, low)], np.full((len(missing), len(missing)), .72)], axis=1)
            # Low-confidence boxes may continue a recent ID, never resurrect a long-lost one.
            for row, original in enumerate(missing):
                if timestamp-tracks[original].last_seen > .5:
                    second[row, :len(low)] = 100
            accepted.update({missing[row]: (low[col], margin) for row, (col, margin) in self._assign(second, .10).items()})
        for row, (col, _) in list(accepted.items()):
            track, detection = tracks[row], detections[col]
            if timestamp-track.last_seen > 1.2 and self._appearance(track, detection) > .18:
                consistent = (track.recovery_anchor is not None and timestamp-track.recovery_time < .2
                              and np.linalg.norm(detection.anchor-track.recovery_anchor) < .035)
                track.recovery_hits = track.recovery_hits+1 if consistent else 1
                track.recovery_anchor = detection.anchor.copy()
                track.recovery_time = timestamp
                if track.recovery_hits < 3:
                    del accepted[row]
        for row, track in enumerate(tracks):
            track.detection_index = None
            if row not in accepted:
                if track.state != "unbound":
                    track.state = "uncertain" if timestamp - track.last_seen < .8 else "lost"
                track.match_score = 0
                continue
            col, margin = accepted[row]
            detection = detections[col]
            elapsed = max(.001, timestamp - track.last_seen)
            delta = detection.anchor - track.detection.anchor
            if elapsed < .5:
                track.velocity = .60 * track.velocity + .40 * delta / elapsed
                old_head, new_head = track.detection.keypoints[0], detection.keypoints[0]
                if min(old_head[2], new_head[2]) >= .45:
                    velocity = (new_head[:2]-old_head[:2])/elapsed
                    velocity *= min(1., 1.5/max(1e-6, np.linalg.norm(velocity)))
                    track.head_velocity = .60*track.head_velocity+.40*velocity
                else:
                    track.head_velocity *= .8
                distance = float(np.linalg.norm(self.court.project(detection.foot) - self.court.project(track.detection.foot)))
                if distance / elapsed < 12 and distance > .015:
                    track.distance += distance
            else:
                track.velocity[:] = 0
                track.head_velocity[:] = 0
                track.trail.clear()
            # Avoid contaminating the identity gallery with overlapping bodies.
            overlapping = any(j != col and _iou(detection.box, d.box) > .18 for j, d in enumerate(detections))
            if margin > .15 and not overlapping and detection.confidence > .65 and detection.feature.size and self._appearance(track, detection) < .32:
                track.gallery.append(detection.feature.copy())
            track.detection = detection
            track.detection_index = col
            track.last_seen = timestamp
            track.state = "tracked"
            track.recovery_hits = 0
            track.match_score = float(np.clip(1 - costs[row, col], 0, 1))
            track.trail.append(detection.foot.copy())

    def snapshot(self):
        return [{"id": t.player.id, "name": t.player.name, "color": t.player.color,
                 "side": t.player.side, "state": t.state, "detection_index": t.detection_index,
                 "score": t.match_score, "distance": t.distance,
                 "source": t.detection.source if t.detection and t.state == "tracked" else None,
                 "box": t.detection.box.copy() if t.detection else None,
                 "keypoints": t.detection.keypoints.copy() if t.detection else None,
                 "foot": t.detection.foot.copy() if t.detection else None,
                 "trail": list(t.trail)} for t in self.tracks.values()]


def _iou(a, b):
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    return intersection / max(1e-8, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)
