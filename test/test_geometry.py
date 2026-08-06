from pathlib import Path

import pytest

from spell_tracker import geometry

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "spell_tracker" / "data" / "characters.json"


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


def test_compare_to_reference_score_is_stable_across_point_densities():
    """The same shape traced at very different point densities (e.g. a
    slow vs. fast drawing on a fixed-framerate webcam) should score
    similarly, since strokes are resampled to a fixed point count before
    comparison."""
    medians = geometry.load_reference("火", path=str(FIXTURE_PATH))
    sparse = [geometry.resample_strokes(stroke, 5) for stroke in medians]
    dense = [geometry.resample_strokes(stroke, 200) for stroke in medians]

    sparse_total, _ = geometry.compare_to_reference(sparse, medians)
    dense_total, _ = geometry.compare_to_reference(dense, medians)

    assert sparse_total == pytest.approx(0.0, abs=0.05)
    assert dense_total == pytest.approx(0.0, abs=0.05)


def test_load_references_loads_multiple_characters_in_one_pass():
    refs = geometry.load_references(["火", "土"], path=str(FIXTURE_PATH))
    assert set(refs.keys()) == {"火", "土"}
    assert refs["火"] == geometry.load_reference("火", path=str(FIXTURE_PATH))
    assert refs["土"] == geometry.load_reference("土", path=str(FIXTURE_PATH))


def test_load_references_raises_on_missing_character():
    with pytest.raises(ValueError):
        geometry.load_references(["火", "not-a-real-character"], path=str(FIXTURE_PATH))


def test_identify_character_ranks_exact_match_first():
    candidates = geometry.load_references(["火", "土"], path=str(FIXTURE_PATH))
    huo_medians = candidates["火"]

    ranked = geometry.identify_character(huo_medians, candidates)

    assert ranked[0][0] == "火"
    assert ranked[0][1] == pytest.approx(0.0, abs=1e-6)


def test_identify_character_skips_candidates_with_wrong_stroke_count():
    candidates = geometry.load_references(["火", "土"], path=str(FIXTURE_PATH))
    tu_medians = candidates["土"]  # 3 strokes; 火 has 4, so it should be skipped

    ranked = geometry.identify_character(tu_medians, candidates)

    assert [character for character, _ in ranked] == ["土"]


def test_identify_character_empty_when_no_candidate_matches_stroke_count():
    candidates = geometry.load_references(["火", "土"], path=str(FIXTURE_PATH))
    one_stroke = [[(0, 0), (1, 1)]]

    assert geometry.identify_character(one_stroke, candidates) == []
