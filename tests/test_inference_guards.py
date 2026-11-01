import numpy as np
from badminton_vision.inference import CutDetector


def test_scene_cut_requires_large_full_frame_change():
    detector = CutDetector()
    dark = np.zeros((120, 200, 3), dtype=np.uint8)
    assert not detector.update(dark)
    moving_player = dark.copy()
    moving_player[30:90, 50:75] = 255
    assert not detector.update(moving_player)
    assert detector.update(np.full_like(dark, 180))


def test_pose_deduplication_preserves_distinct_visible_heads():
    from tests.test_tracker import detection
    from badminton_vision.inference import merge_detections
    a, b = detection(.5), detection(.505)
    a.keypoints[0] = [.49, .50, .9]
    b.keypoints[0] = [.53, .53, .9]
    a.keypoints[11:13] = b.keypoints[11:13] = [[.49, .65, .9], [.51, .65, .9]]
    assert len(merge_detections([a, b])) == 2
    duplicate = detection(.501)
    duplicate.keypoints = a.keypoints.copy()
    assert len(merge_detections([a, duplicate])) == 1
