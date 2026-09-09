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
    session_id: str
    variant: str
    track_id: str
    action: Literal["impression", "play", "skip", "complete", "like"]
    seed_track_id: str | None = None
    position: int | None = None
    dwell_ms: int | None = None


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
