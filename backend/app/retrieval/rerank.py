"""Stage 2: rerank the ANN candidates.

v1 collapsed everything into `meta_mse + lambda * (1 - cos)` with lambda frozen
at 0.1 by an offline same-artist proxy. Two problems: one scalar can't express
"prefer acoustically close *but* tonally varied", and the proxy it was tuned on
is not the objective anyone actually cares about.

Here each ranking signal stays separate, gets combined by an explicit weight
vector, and then goes through MMR so the returned list isn't six near-duplicates
by the same artist. The weight vectors are the bandit arms (see
`app.experiments`), so which one is right becomes an online question rather
than a hyperparameter someone hardcoded.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.embeddings.descriptors import DESCRIPTOR_KEYS, normalize_descriptors


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """Weights over the reranking signals. One instance == one bandit arm."""

    name: str
    audio: float = 1.0
    descriptor: float = 0.3
    tag: float = 0.2
    novelty: float = 0.0
    # MMR tradeoff: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr_lambda: float = 0.75

    def as_dict(self) -> dict[str, float]:
        return {
            "audio": self.audio,
            "descriptor": self.descriptor,
            "tag": self.tag,
            "novelty": self.novelty,
            "mmr_lambda": self.mmr_lambda,
        }


def _descriptor_similarity(seed: dict[str, Any], cand: dict[str, Any]) -> float:
    """1 - mean absolute difference over normalized descriptors.

    L1 rather than the squared error v1 used: with only six bounded dimensions,
    MSE lets a single outlier dimension (usually tempo, which doubles/halves on
    beat-tracking errors) dominate the whole comparison.
    """
    a = normalize_descriptors({k: seed.get(k, 0.0) for k in DESCRIPTOR_KEYS if k in seed})
    b = normalize_descriptors({k: cand.get(k, 0.0) for k in DESCRIPTOR_KEYS if k in cand})
    shared = [k for k in DESCRIPTOR_KEYS if k in a and k in b]
    if not shared:
        return 0.0
    diff = np.mean([abs(a[k] - b[k]) for k in shared])
    return float(1.0 - diff)


def _tag_similarity(seed_tags: list[str], cand_tags: list[str]) -> float:
    """Jaccard overlap on Jamendo tags."""
    if not seed_tags or not cand_tags:
        return 0.0
    s, c = set(seed_tags), set(cand_tags)
    union = s | c
    return float(len(s & c) / len(union)) if union else 0.0


def score_candidates(
    seed: dict[str, Any],
    candidates: list[dict[str, Any]],
    weights: RankingWeights,
) -> list[dict[str, Any]]:
    """Attach per-signal scores and a fused relevance score to each candidate."""
    seed_tags = seed.get("tags") or []
    scored = []

    for cand in candidates:
        audio = float(cand.get("audio_score", 0.0))
        descriptor = _descriptor_similarity(seed, cand)
        tag = _tag_similarity(seed_tags, cand.get("tags") or [])
        # Novelty rewards the long tail: log-damped inverse popularity, so a
        # positive novelty weight surfaces tracks collaborative filtering can't
        # reach at all. This is the differentiator against play-count systems.
        listens = float(cand.get("listens") or 0.0)
        novelty = 1.0 / (1.0 + np.log1p(listens))

        relevance = (
            weights.audio * audio
            + weights.descriptor * descriptor
            + weights.tag * tag
            + weights.novelty * novelty
        )

        scored.append(
            {
                **cand,
                "signals": {
                    "audio": audio,
                    "descriptor": descriptor,
                    "tag": tag,
                    "novelty": float(novelty),
                },
                "relevance": float(relevance),
            }
        )

    scored.sort(key=lambda c: c["relevance"], reverse=True)
    return scored


def mmr_select(
    scored: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    *,
    top_k: int,
    mmr_lambda: float,
    max_per_artist: int = 2,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance selection with a per-artist cap.

    Greedy: repeatedly take the candidate maximizing
        lambda * relevance - (1 - lambda) * max_similarity_to_already_selected.
    Without this, a seed track's nearest neighbours in CLAP space are routinely
    other tracks from the same album, which reads as a broken recommender even
    though every individual score is correct.
    """
    if mmr_lambda >= 1.0:
        return _apply_artist_cap(scored, top_k, max_per_artist)

    selected: list[dict[str, Any]] = []
    selected_vecs: list[np.ndarray] = []
    artist_counts: dict[str, int] = {}
    pool = list(scored)

    while pool and len(selected) < top_k:
        best_idx, best_score = None, -np.inf

        for i, cand in enumerate(pool):
            artist = cand.get("artist_id") or ""
            if artist and artist_counts.get(artist, 0) >= max_per_artist:
                continue

            vec = vectors.get(cand["track_id"])
            if selected_vecs and vec is not None:
                redundancy = max(float(vec @ sv) for sv in selected_vecs)
            else:
                redundancy = 0.0

            score = mmr_lambda * cand["relevance"] - (1.0 - mmr_lambda) * redundancy
            if score > best_score:
                best_idx, best_score = i, score

        if best_idx is None:
            break

        chosen = pool.pop(best_idx)
        selected.append(chosen)
        artist = chosen.get("artist_id") or ""
        if artist:
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        vec = vectors.get(chosen["track_id"])
        if vec is not None:
            selected_vecs.append(vec)

    return selected


def _apply_artist_cap(
    scored: list[dict[str, Any]], top_k: int, max_per_artist: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for cand in scored:
        artist = cand.get("artist_id") or ""
        if artist and counts.get(artist, 0) >= max_per_artist:
            continue
        out.append(cand)
        if artist:
            counts[artist] = counts.get(artist, 0) + 1
        if len(out) >= top_k:
            break
    return out
