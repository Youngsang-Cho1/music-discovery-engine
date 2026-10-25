"""Pydantic request/response models."""
from typing import Literal

from pydantic import BaseModel, Field


class TrackOut(BaseModel):
    id: str
    name: str
    artist_name: str | None = None
    album_name: str | None = None
    duration_sec: int | None = None
    audio_url: str | None = None
    image_url: str | None = None
    license_url: str | None = None
    share_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)
    # Exposed so the UI can show what the `novelty` signal is reacting to. A
    # zero here is the claim this project is built on: a track with no
    # listening history is still retrievable.
    listens: int = 0
    favorites: int = 0

    model_config = {"from_attributes": True}


class RecommendationOut(BaseModel):
    track: TrackOut
    relevance: float
    signals: dict[str, float]


class RecommendationResponse(BaseModel):
    variant: str
    session_id: str
    seed_track_id: str | None = None
    query: str | None = None
    results: list[RecommendationOut]


class EventIn(BaseModel):
    """One interaction with one recommended track.

    `variant` must be the value echoed by the recommendation response that
    produced this track — that is what attributes the outcome to a ranking arm.

    Reward semantics: `complete` and `like` count positively. `skip` counts
    against the arm only when `dwell_ms` is under the skip threshold — a
    listener who skips at 2:30 heard the track and did not reject the pick.
    """

    session_id: str = Field(description="Stable per-browser id, for grouping events")
    variant: str = Field(description="Ranking arm echoed by the recommendation response")
    track_id: str
    action: Literal["impression", "play", "skip", "complete", "like"]
    seed_track_id: str | None = None
    position: int | None = Field(
        default=None, description="0-based rank in the returned list; corrects position bias"
    )
    dwell_ms: int | None = Field(default=None, description="Milliseconds actually listened")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "6f1c0d2e-6a7b-4f3e-9c11-2b8a5d0e4f77",
                    "variant": "discovery",
                    "track_id": "1215804",
                    "seed_track_id": "525910",
                    "action": "complete",
                    "position": 2,
                    "dwell_ms": 187000,
                }
            ]
        }
    }


class ArmStat(BaseModel):
    variant: str
    impressions: int
    rewards: int
    posterior_mean: float
    ci_low: float
    ci_high: float
    probability_best: float | None = None


class ExperimentReport(BaseModel):
    arms: list[ArmStat]
    total_impressions: int
    total_rewards: int


class TagCount(BaseModel):
    tag: str
    count: int


class CatalogStats(BaseModel):
    tracks: int
    artists: int
    total_duration_sec: int
    top_tags: list[TagCount] = Field(default_factory=list)
