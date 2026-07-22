from collections import deque

import pytest

from spell_tracker.hand_tracker import StrokeTracker


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
