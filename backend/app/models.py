"""ORM models: catalog, interaction events, and bandit state.

Postgres is the source of truth for track metadata and every interaction event.
Audio embeddings live in Qdrant (keyed by the same track id) because ANN search
is the one access pattern Postgres is bad at.
"""
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Track(Base):
    """A CC-licensed track sourced from Jamendo.

    `features` holds audio-derived scalar descriptors (tempo, rms energy,
    spectral centroid, ...). Unlike the v1 pipeline these are computed from the
    waveform rather than pulled from Spotify, so the catalog has no dependency
    on a proprietary metadata provider.
    """

    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(512))
    artist_id: Mapped[str | None] = mapped_column(String(32), index=True)
    artist_name: Mapped[str | None] = mapped_column(String(512), index=True)
    album_name: Mapped[str | None] = mapped_column(String(512))

    duration_sec: Mapped[int | None] = mapped_column(Integer)
    release_date: Mapped[str | None] = mapped_column(String(32))

    # Streaming + licensing. audio_url is a Jamendo-hosted stream we are
    # licensed to play; license_url points at the specific CC deed.
    audio_url: Mapped[str | None] = mapped_column(String(1024))
    image_url: Mapped[str | None] = mapped_column(String(1024))
    license_url: Mapped[str | None] = mapped_column(String(512))
    share_url: Mapped[str | None] = mapped_column(String(1024))

    # Jamendo musicinfo: tags plus coarse categorical descriptors.
    tags: Mapped[list | None] = mapped_column(JSON, default=list)
    speed: Mapped[str | None] = mapped_column(String(16))
    acoustic_electric: Mapped[str | None] = mapped_column(String(16))
    vocal_instrumental: Mapped[str | None] = mapped_column(String(16))
    lang: Mapped[str | None] = mapped_column(String(16))

    # Audio-derived scalar descriptors, computed in the ingestion pipeline.
    features: Mapped[dict | None] = mapped_column(JSON, default=dict)

    embedded: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_tracks_embedded_artist", "embedded", "artist_id"),)


class InteractionEvent(Base):
    """One user action on one recommended track.

    This is the online signal that replaces the offline same-artist proxy used
    to tune lambda in v1. `variant` records which ranking arm produced the
    recommendation, which is what makes the bandit updateable.
    """

    __tablename__ = "interaction_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    variant: Mapped[str] = mapped_column(String(64), index=True)

    seed_track_id: Mapped[str | None] = mapped_column(String(32), index=True)
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id"), index=True)

    # skip | complete | like | play | impression
    action: Mapped[str] = mapped_column(String(32), index=True)
    # Where in the returned list this track sat (0-based); lets us correct for
    # position bias when evaluating arms.
    position: Mapped[int | None] = mapped_column(Integer)
    dwell_ms: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    __table_args__ = (Index("ix_events_variant_action", "variant", "action"),)


class BanditArm(Base):
    """Beta posterior for one ranking variant.

    Durable counterpart to the Redis copy: Redis serves the hot read path during
    assignment, Postgres survives a cache flush.
    """

    __tablename__ = "bandit_arms"

    variant: Mapped[str] = mapped_column(String(64), primary_key=True)
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    rewards: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
