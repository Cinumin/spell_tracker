from collections import deque

import pytest
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

from spell_tracker.hand_tracker import (
    PINCH_SUSTAIN_THRESHOLD,
    StrokeTracker,
    is_pinch,
    smooth_stroke,
)


def make_hand_result(landmark_xy: dict) -> HandLandmarkerResult:
    """Build a minimal fake HandLandmarkerResult with 21 zeroed landmarks,
    overriding only the indices needed for is_pinch()'s distance math."""
    landmarks = [NormalizedLandmark(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    for idx, (x, y) in landmark_xy.items():
        landmarks[idx] = NormalizedLandmark(x=x, y=y, z=0.0)
    return HandLandmarkerResult(handedness=[[]], hand_landmarks=[landmarks], hand_world_landmarks=[[]])


# thumb_ip=3, thumb_tip=4, index_dip=7, index_tip=8 chosen so relativeDistance
# == 6.8 -- between PINCH_ENGAGE_THRESHOLD (6.0) and PINCH_SUSTAIN_THRESHOLD
# (7.5), i.e. a "borderline noisy frame" that reads as pinching only under
# the looser sustain threshold.
BORDERLINE_RESULT = make_hand_result({3: (0.1, 0.0), 4: (0.0, 0.0), 7: (0.168, 0.0), 8: (0.068, 0.0)})

# Same landmark shape, clearly within both thresholds (relativeDistance == 3.0).
CLEAR_PINCH_RESULT = make_hand_result({3: (0.1, 0.0), 4: (0.0, 0.0), 7: (0.13, 0.0), 8: (0.03, 0.0)})


def test_is_pinch_uses_engage_threshold_by_default():
    assert not is_pinch(BORDERLINE_RESULT)


def test_is_pinch_respects_explicit_sustain_threshold():
    assert is_pinch(BORDERLINE_RESULT, threshold=PINCH_SUSTAIN_THRESHOLD)


def test_update_sustains_pinch_through_transient_noisy_frame():
    tracker = StrokeTracker()

    tracker.latest_result = CLEAR_PINCH_RESULT
    tracker.update(640, 480)
    tracker.latest_result = BORDERLINE_RESULT
    tracker.update(640, 480)
    tracker.latest_result = CLEAR_PINCH_RESULT
    tracker.update(640, 480)

    assert len(tracker.current_stroke) == 3
    assert tracker.strokes == []


def test_smooth_stroke_empty_returns_empty():
    assert smooth_stroke([]) == []


def test_smooth_stroke_single_point_returns_that_point():
    assert smooth_stroke([(5, 5)]) == [(5, 5)]


def test_smooth_stroke_two_points_starts_and_ends_at_inputs():
    curve = smooth_stroke([(0, 0), (100, 0)])
    assert curve[0] == (0, 0)
    assert curve[-1] == (100, 0)


def test_smooth_stroke_upsamples_multi_point_stroke():
    points = [(0, 0), (50, 0), (50, 50), (0, 50)]
    curve = smooth_stroke(points)
    assert len(curve) > len(points)


def test_smooth_stroke_passes_through_original_points():
    points = [(0, 0), (50, 0), (50, 50), (0, 50)]
    curve = smooth_stroke(points)
    for point in points:
        assert point in curve


def test_end_stroke_moves_buffer_to_strokes_list():
    tracker = StrokeTracker()
    tracker.add_point((1, 2))
    tracker.add_point((3, 4))
    tracker.end_stroke()
    assert list(tracker.strokes[0]) == [(1, 2), (3, 4)]
    assert len(tracker.current_stroke) == 0


def test_end_stroke_is_noop_when_buffer_empty():
    tracker = StrokeTracker()
    tracker.end_stroke()
    assert tracker.strokes == []


def test_clear_resets_all_state():
    tracker = StrokeTracker()
    tracker.add_point((1, 2))
    tracker.end_stroke()
    tracker.add_point((5, 6))
    tracker.clear()
    assert tracker.strokes == []
    assert len(tracker.current_stroke) == 0


def test_compare_to_wraps_geometry_compare_to_reference():
    tracker = StrokeTracker()
    tracker.strokes = [deque([(0, 0), (1, 1)])]
    total, per_stroke = tracker.compare_to([[(0, 0), (1, 1)]])
    assert total == pytest.approx(0.0, abs=1e-6)
