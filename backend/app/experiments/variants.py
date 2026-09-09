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
    # The long-tail bet: rewards low-play-count tracks. This is the arm that
    # tests our core claim against popularity-driven baselines.
    #
    # Novelty is weighted far below the other signals on purpose. Weights act
    # on a signal's spread *within a candidate pool*, not its nominal 0-1
    # range, and those differ sharply here: ANN has already selected similar
    # tracks, so audio cosines arrive clustered (measured spread 0.127), while
    # novelty still spans almost the whole range (0.937). At 0.35 its effective
    # pull was 2.6x audio's and it stopped being a tie-breaker — every result
    # was a zero-play track regardless of how well it matched. At 0.08 the
    # pulls are comparable, so novelty breaks ties toward the long tail instead
    # of replacing the ranking.
    "discovery": RankingWeights(
        name="discovery", audio=1.0, descriptor=0.3, tag=0.2, novelty=0.08, mmr_lambda=0.7
    ),
}

DEFAULT_VARIANT = "control"


def get_variant(name: str | None) -> RankingWeights:
    return VARIANTS.get(name or DEFAULT_VARIANT, VARIANTS[DEFAULT_VARIANT])
