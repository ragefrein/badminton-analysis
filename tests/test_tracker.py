import numpy as np
import pytest

from badminton_vision.domain import Court, Detection, make_roster
from badminton_vision.tracker import RosterTracker


def detection(x, y=.75, identity=0, shared=False):
    keypoints = np.zeros((17, 3))
    keypoints[15] = [x-.01, y, .95]
    keypoints[16] = [x+.01, y, .95]
    feature = np.eye(4)[0 if shared else identity]
    return Detection(np.array([x-.035, y-.22, x+.035, y+.01]), keypoints, .95, feature)


def tracker_pair(shared=False):
    court = Court(((0, 0), (1, 0), (1, 1), (0, 1)))
    tracker = RosterTracker(make_roster(), court)
    initial = [detection(.3, .25, 0), detection(.7, .25, 1),
               detection(.3, identity=2, shared=shared), detection(.7, identity=3, shared=shared)]
    tracker.propose(initial, 0)
    return tracker, initial


def test_ids_survive_detector_order_changes_and_crossing():
    tracker, _ = tracker_pair()
    for step in range(1, 31):
        t = step/30
        detections = [detection(.3+.4*t, identity=2), detection(.7-.4*t, identity=3),
                      detection(.3, .25, 0), detection(.7, .25, 1)]
        tracker.update(list(reversed(detections)), t)
        assert tracker.tracks[3].state == "tracked"
        assert np.argmax(tracker.tracks[3].detection.feature) == 2
        assert np.argmax(tracker.tracks[4].detection.feature) == 3
    assert tracker.tracks[3].detection.foot[0] > tracker.tracks[4].detection.foot[0]


def test_indistinguishable_crossing_withholds_identity():
    tracker, _ = tracker_pair(shared=True)
    for step in range(1, 31):
        x = .3+.2*step/30
        tracker.update([detection(x, shared=True), detection(1-x, shared=True),
                        detection(.3, .25, 0), detection(.7, .25, 1)], step/30)
    assert tracker.tracks[3].state != "tracked"
    assert tracker.tracks[4].state != "tracked"
    assert tracker.tracks[3].detection_index is None


def test_missing_player_is_not_replaced_by_opponent():
    tracker, initial = tracker_pair()
    tracker.update(initial[:2] + initial[3:], .1)
    assert tracker.tracks[3].state == "uncertain"
    tracker.update(initial[:2]+initial[3:], 1.5)
    assert tracker.tracks[3].state == "lost"
    tracker.update(initial, 1.6)
    assert tracker.tracks[3].state == "tracked"
    assert len(tracker.tracks) == 4


def test_explicit_rebind_swaps_states_and_keeps_names():
    tracker, initial = tracker_pair()
    tracker.players[2].name = "Ayu"
    tracker.players[3].name = "Sari"
    tracker.bind(3, 3, initial, .1)
    assert tracker.tracks[3].player.name == "Ayu"
    assert tracker.tracks[4].player.name == "Sari"
    assert tracker.tracks[3].detection_index == 3
    assert tracker.tracks[4].detection_index == 2
    assert np.argmax(tracker.tracks[3].gallery[0]) == 3


def test_cut_or_seek_requires_new_explicit_bindings():
    tracker, initial = tracker_pair()
    tracker.invalidate()
    tracker.update(initial, 1)
    assert not tracker.ready()
    assert all(t.state == "unbound" for t in tracker.tracks.values())
    for index in range(4):
        tracker.bind(index+1, index, initial, 1)
    assert tracker.ready()


def test_opposite_end_is_gated_even_with_identical_appearance():
    tracker, _ = tracker_pair()
    tracker.update([detection(.3, .25, 2)], .1)
    assert tracker.tracks[3].state != "tracked"


def test_no_distance_added_over_long_gap():
    tracker, initial = tracker_pair()
    tracker.update([], .2)
    shifted = initial[:2]+[detection(.4, identity=2), initial[3]]
    tracker.update(shifted, 1.5)
    assert tracker.tracks[3].distance == 0


def test_court_projection_and_validation():
    court = Court()
    assert np.allclose(court.project(court.corners[0]), [0, 0], atol=1e-4)
    assert np.allclose(court.project(court.corners[2]), [6.1, 13.4], atol=1e-4)
    with pytest.raises(ValueError):
        Court(((0, 0), (1, 1), (1, 0), (0, 1)))


def test_unobserved_tracks_are_not_rendered():
    from badminton_vision.render import annotate
    tracker, _ = tracker_pair()
    tracker.update([], .1)
    raw = np.zeros((100, 100, 3), dtype=np.uint8)
    assert np.array_equal(raw, annotate(raw, tracker.snapshot()))


def test_low_confidence_body_continues_existing_id_without_learning_its_appearance():
    tracker, initial = tracker_pair()
    gallery_size = len(tracker.tracks[3].gallery)
    initial[2].confidence = .19
    tracker.update(initial, 1/30)
    assert tracker.tracks[3].state == "tracked"
    assert len(tracker.tracks[3].gallery) == gallery_size
    fresh, _ = tracker_pair()
    fresh.invalidate()
    fresh.update(initial, 1/30)
    assert all(t.state == "unbound" for t in fresh.tracks.values())


def test_ankle_jitter_does_not_break_identity_when_hips_stay_still():
    tracker, initial = tracker_pair()
    for index, d in enumerate(initial):
        x = float(d.foot[0])
        d.keypoints[11:13] = [[x-.01, d.foot[1]-.1, .9], [x+.01, d.foot[1]-.1, .9]]
    moved = detection(.3, identity=2)
    moved.keypoints[11:13] = initial[2].keypoints[11:13]
    moved.keypoints[15:17, :2] += [.10, -.1]
    tracker.update(initial[:2]+[moved, initial[3]], 1/30)
    assert tracker.tracks[3].state == "tracked"


def test_visible_heads_separate_teammates_with_shared_occluded_hips():
    tracker, initial = tracker_pair(shared=True)
    for p, x in ((2, .3), (3, .7)):
        initial[p].keypoints[0] = [x, .5, .95]
    for step in range(1, 31):
        fraction = step/30
        pair = [detection(.3+.15*fraction, identity=2, shared=True),
                detection(.7-.15*fraction, identity=3, shared=True)]
        for d, x in zip(pair, (.3+.15*fraction, .7-.15*fraction)):
            d.keypoints[0] = [x, .5, .95]
            d.keypoints[11:13] = [[.49, .65, .9], [.51, .65, .9]]
        tracker.update(initial[:2]+pair, step/30)
    assert tracker.tracks[3].state == "tracked"
    assert tracker.tracks[4].state == "tracked"
    assert tracker.tracks[3].detection.keypoints[0, 0] < tracker.tracks[4].detection.keypoints[0, 0]


def test_long_gap_with_moderate_reid_needs_three_consistent_frames():
    tracker, initial = tracker_pair()
    tracker.update(initial[:2]+initial[3:], 2)
    changed = detection(.3, identity=2)
    changed.feature = np.array([0, 0, .73, np.sqrt(1-.73**2)])
    for timestamp in (2.1, 2.133):
        tracker.update(initial[:2]+[changed, initial[3]], timestamp)
        assert tracker.tracks[3].state == "lost"
    tracker.update(initial[:2]+[changed, initial[3]], 2.166)
    assert tracker.tracks[3].state == "tracked"
