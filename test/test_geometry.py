from pathlib import Path

import pytest

from spell_tracker import geometry

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "spell_tracker" / "data" / "huo.json"


def test_load_reference_and_normalize():
    medians = geometry.load_reference("火", path=str(FIXTURE_PATH))
    normalized = geometry.normalize_strokes(medians)
    assert len(normalized) == len(medians)
    for stroke in normalized:
        assert stroke.shape[1] == 2


def test_compare_to_reference_self_similarity_is_zero():
    medians = geometry.load_reference("火", path=str(FIXTURE_PATH))
    total, per_stroke = geometry.compare_to_reference(medians, medians)
    assert total == pytest.approx(0.0, abs=1e-6)
    assert len(per_stroke) == len(medians)


def test_compare_to_reference_raises_on_stroke_count_mismatch():
    medians = geometry.load_reference("火", path=str(FIXTURE_PATH))
    with pytest.raises(ValueError):
        geometry.compare_to_reference(medians[:-1], medians)
