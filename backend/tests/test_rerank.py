import numpy as np

from app.retrieval.rerank import (
    RankingWeights,
    _descriptor_similarity,
    _tag_similarity,
    mmr_select,
    score_candidates,
)


def _cand(track_id, *, audio, artist="a1", tags=None, listens=0, **descriptors):
    return {
        "track_id": track_id,
        "artist_id": artist,
        "tags": tags or [],
        "audio_score": audio,
        "listens": listens,
        **descriptors,
    }


def test_tag_similarity_is_jaccard():
    assert _tag_similarity(["rock", "guitar"], ["rock", "guitar"]) == 1.0
    assert _tag_similarity(["rock"], ["jazz"]) == 0.0
    # |∩| = 1, |∪| = 3
    assert _tag_similarity(["rock", "guitar"], ["rock"]) == 1 / 2


def test_tag_similarity_handles_missing_tags():
    assert _tag_similarity([], ["rock"]) == 0.0
    assert _tag_similarity(["rock"], []) == 0.0


def test_descriptor_similarity_peaks_on_identical_tracks():
    seed = {"tempo_bpm": 120.0, "loudness_db": -12.0}
    assert _descriptor_similarity(seed, dict(seed)) == 1.0


def test_descriptor_similarity_falls_off_with_distance():
    seed = {"tempo_bpm": 120.0}
    near = _descriptor_similarity(seed, {"tempo_bpm": 125.0})
    far = _descriptor_similarity(seed, {"tempo_bpm": 200.0})
    assert near > far


def test_descriptor_similarity_without_overlap_is_zero():
    """No shared descriptor keys means no evidence, not a perfect match."""
    assert _descriptor_similarity({"tempo_bpm": 120.0}, {"loudness_db": -5.0}) == 0.0


def test_score_candidates_orders_by_fused_relevance():
    seed = {"tempo_bpm": 120.0, "tags": ["rock"]}
    weights = RankingWeights(name="t", audio=1.0, descriptor=0.0, tag=0.0, novelty=0.0)

    scored = score_candidates(
        seed,
        [_cand("low", audio=0.1), _cand("high", audio=0.9), _cand("mid", audio=0.5)],
        weights,
    )
    assert [c["track_id"] for c in scored] == ["high", "mid", "low"]


def test_tag_weight_can_overturn_audio_ranking():
    """The point of keeping signals separate: a weight change reorders results."""
    seed = {"tags": ["jazz", "piano"]}
    candidates = [
        _cand("audio_win", audio=0.9, tags=["metal"]),
        _cand("tag_win", audio=0.6, tags=["jazz", "piano"]),
    ]

    audio_only = RankingWeights(name="a", audio=1.0, descriptor=0.0, tag=0.0)
    tag_heavy = RankingWeights(name="b", audio=1.0, descriptor=0.0, tag=1.0)

    assert score_candidates(seed, candidates, audio_only)[0]["track_id"] == "audio_win"
    assert score_candidates(seed, candidates, tag_heavy)[0]["track_id"] == "tag_win"


def test_novelty_rewards_the_long_tail():
    """The discovery arm must be able to outrank a more popular, better-matching
    track — that is the whole claim against play-count-driven baselines."""
    seed = {}
    candidates = [
        _cand("popular", audio=0.80, listens=1_000_000, artist="a1"),
        _cand("obscure", audio=0.78, listens=3, artist="a2"),
    ]

    neutral = RankingWeights(name="n", audio=1.0, descriptor=0.0, tag=0.0, novelty=0.0)
    discovery = RankingWeights(name="d", audio=1.0, descriptor=0.0, tag=0.0, novelty=0.35)

    assert score_candidates(seed, candidates, neutral)[0]["track_id"] == "popular"
    assert score_candidates(seed, candidates, discovery)[0]["track_id"] == "obscure"


def test_signals_are_reported_per_candidate():
    """The API exposes these so a recommendation can be explained, not just made."""
    scored = score_candidates(
        {"tags": ["rock"], "tempo_bpm": 120.0},
        [_cand("x", audio=0.5, tags=["rock"], tempo_bpm=118.0)],
        RankingWeights(name="t"),
    )
    assert set(scored[0]["signals"]) == {"audio", "descriptor", "tag", "novelty"}


# --- MMR ---------------------------------------------------------------


def test_mmr_lambda_one_is_pure_relevance():
    scored = score_candidates(
        {},
        [_cand("a", audio=0.9), _cand("b", audio=0.8), _cand("c", audio=0.7)],
        RankingWeights(name="t", audio=1.0, descriptor=0.0, tag=0.0),
    )
    picked = mmr_select(scored, {}, top_k=3, mmr_lambda=1.0, max_per_artist=99)
    assert [c["track_id"] for c in picked] == ["a", "b", "c"]


def test_mmr_demotes_a_near_duplicate():
    """`dup` scores higher than `diff` but is nearly collinear with the top pick,
    so a diversity-weighted selection should prefer `diff` second."""
    scored = score_candidates(
        {},
        [
            _cand("top", audio=0.90, artist="a1"),
            _cand("dup", audio=0.89, artist="a2"),
            _cand("diff", audio=0.70, artist="a3"),
        ],
        RankingWeights(name="t", audio=1.0, descriptor=0.0, tag=0.0),
    )
    vectors = {
        "top": np.array([1.0, 0.0], dtype=np.float32),
        "dup": np.array([0.999, 0.0447], dtype=np.float32),
        "diff": np.array([0.0, 1.0], dtype=np.float32),
    }

    picked = mmr_select(scored, vectors, top_k=2, mmr_lambda=0.5, max_per_artist=99)
    assert [c["track_id"] for c in picked] == ["top", "diff"]


def test_artist_cap_is_enforced():
    """Without this, CLAP's nearest neighbours are usually the same album."""
    scored = score_candidates(
        {},
        [
            _cand("s1", audio=0.99, artist="same"),
            _cand("s2", audio=0.98, artist="same"),
            _cand("s3", audio=0.97, artist="same"),
            _cand("other", audio=0.10, artist="other"),
        ],
        RankingWeights(name="t", audio=1.0, descriptor=0.0, tag=0.0),
    )

    picked = mmr_select(scored, {}, top_k=3, mmr_lambda=1.0, max_per_artist=2)
    ids = [c["track_id"] for c in picked]
    assert ids.count("s3") == 0
    assert "other" in ids
    assert len(ids) == 3


def test_artist_cap_applies_under_mmr_too():
    scored = score_candidates(
        {},
        [
            _cand("s1", audio=0.99, artist="same"),
            _cand("s2", audio=0.98, artist="same"),
            _cand("s3", audio=0.97, artist="same"),
            _cand("other", audio=0.10, artist="other"),
        ],
        RankingWeights(name="t", audio=1.0, descriptor=0.0, tag=0.0),
    )
    vectors = {k: np.array([1.0, 0.0], dtype=np.float32) for k in ("s1", "s2", "s3")}
    vectors["other"] = np.array([0.0, 1.0], dtype=np.float32)

    picked = mmr_select(scored, vectors, top_k=4, mmr_lambda=0.7, max_per_artist=2)
    artists = [c["artist_id"] for c in picked]
    assert artists.count("same") <= 2


def test_mmr_stops_when_the_pool_is_exhausted():
    scored = score_candidates(
        {}, [_cand("only", audio=0.5)], RankingWeights(name="t")
    )
    picked = mmr_select(scored, {}, top_k=10, mmr_lambda=0.5)
    assert len(picked) == 1
