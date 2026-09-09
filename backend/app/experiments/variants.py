"""Ranking variants under test.

Each variant is one weight vector for the stage-2 reranker and one arm of the
bandit. `control` reproduces v1's behaviour — audio similarity dominant, no
diversity term — so every other arm is measured against the shipped baseline
rather than against nothing.
"""
from app.retrieval.rerank import RankingWeights

VARIANTS: dict[str, RankingWeights] = {
    # v1 parity: pure similarity, no diversification, no novelty.
    "control": RankingWeights(
        name="control", audio=1.0, descriptor=0.1, tag=0.0, novelty=0.0, mmr_lambda=1.0
    ),
    # Leans on the interpretable descriptors — tests whether tempo/loudness
    # continuity matters more than raw timbral similarity.
    "descriptor_heavy": RankingWeights(
        name="descriptor_heavy", audio=1.0, descriptor=0.5, tag=0.2, novelty=0.0, mmr_lambda=0.9
    ),
    # Diversity-first: same relevance signals, aggressive MMR.
    "diverse": RankingWeights(
        name="diverse", audio=1.0, descriptor=0.3, tag=0.2, novelty=0.0, mmr_lambda=0.6
    ),
    # The long-tail bet: explicitly rewards low-play-count tracks. This is the
    # arm that tests our core claim against popularity-driven baselines.
    "discovery": RankingWeights(
        name="discovery", audio=1.0, descriptor=0.3, tag=0.2, novelty=0.35, mmr_lambda=0.7
    ),
}

DEFAULT_VARIANT = "control"


def get_variant(name: str | None) -> RankingWeights:
    return VARIANTS.get(name or DEFAULT_VARIANT, VARIANTS[DEFAULT_VARIANT])
