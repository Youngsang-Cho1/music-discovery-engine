"""Jamendo API client.

Replaces the v1 yt-dlp pipeline. Jamendo publishes Creative-Commons licensed
tracks with an official API and hosted stream URLs, so both the ingestion and
the playback path in the deployed service are licensed — unlike scraping
YouTube audio, which violates its Terms of Service and has no redistribution
right attached.
"""
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

# Jamendo caps a single page at 200 rows.
MAX_LIMIT = 200


@dataclass(slots=True)
class JamendoTrack:
    """Flattened view of one Jamendo track row."""

    id: str
    name: str
    artist_id: str | None
    artist_name: str | None
    album_name: str | None
    duration_sec: int | None
    release_date: str | None
    audio_url: str | None
    image_url: str | None
    license_url: str | None
    share_url: str | None
    tags: list[str] = field(default_factory=list)
    speed: str | None = None
    acoustic_electric: str | None = None
    vocal_instrumental: str | None = None
    lang: str | None = None

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> "JamendoTrack":
        musicinfo = row.get("musicinfo") or {}
        tag_groups = musicinfo.get("tags") or {}
        # musicinfo.tags splits into genres / instruments / vartags; for
        # retrieval we treat them as one bag of descriptors.
        tags: list[str] = []
        for group in ("genres", "instruments", "vartags"):
            tags.extend(tag_groups.get(group) or [])

        return cls(
            id=str(row["id"]),
            name=row.get("name") or "",
            artist_id=str(row["artist_id"]) if row.get("artist_id") else None,
            artist_name=row.get("artist_name"),
            album_name=row.get("album_name"),
            duration_sec=int(row["duration"]) if row.get("duration") else None,
            release_date=row.get("releasedate"),
            audio_url=row.get("audio"),
            image_url=row.get("image") or row.get("album_image"),
            license_url=row.get("license_ccurl"),
            share_url=row.get("shareurl"),
            tags=sorted(set(tags)),
            speed=musicinfo.get("speed"),
            acoustic_electric=musicinfo.get("acousticelectric"),
            vocal_instrumental=musicinfo.get("vocalinstrumental"),
            lang=musicinfo.get("lang"),
        )


class JamendoError(RuntimeError):
    pass


class JamendoClient:
    """Thin async wrapper over the Jamendo v3.0 REST API."""

    def __init__(self, client_id: str | None = None, base_url: str | None = None):
        settings = get_settings()
        self.client_id = client_id or settings.jamendo_client_id
        self.base_url = (base_url or settings.jamendo_api_base).rstrip("/")
        if not self.client_id:
            raise JamendoError(
                "JAMENDO_CLIENT_ID is not set — register a free app at "
                "https://devportal.jamendo.com/ and put the client id in .env"
            )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        query = {"client_id": self.client_id, "format": "json", **params}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self.base_url}/{path.lstrip('/')}", params=query)
            resp.raise_for_status()
            payload = resp.json()

        headers = payload.get("headers", {})
        if headers.get("status") != "success":
            raise JamendoError(f"Jamendo returned {headers.get('error_message', headers)}")
        return payload.get("results", [])

    async def fetch_tracks(
        self,
        *,
        limit: int = MAX_LIMIT,
        offset: int = 0,
        tags: str | None = None,
        order: str = "popularity_total",
        audioformat: str = "mp32",
    ) -> list[JamendoTrack]:
        """Fetch one page of tracks.

        Only rows that actually carry a stream URL are returned — a track we
        can't play is useless both for embedding and for the player.
        """
        params: dict[str, Any] = {
            "limit": min(limit, MAX_LIMIT),
            "offset": offset,
            "order": order,
            "audioformat": audioformat,
            "include": "musicinfo licenses",
            "imagesize": 300,
        }
        if tags:
            params["fuzzytags"] = tags

        rows = await self._get("tracks", params)
        tracks = [JamendoTrack.from_api(r) for r in rows]
        return [t for t in tracks if t.audio_url]

    async def iter_tracks(
        self,
        *,
        target: int,
        tags: str | None = None,
        order: str = "popularity_total",
    ):
        """Page through the catalog until `target` tracks have been yielded."""
        seen = 0
        offset = 0
        while seen < target:
            page = await self.fetch_tracks(
                limit=min(MAX_LIMIT, target - seen), offset=offset, tags=tags, order=order
            )
            if not page:
                return
            for track in page:
                yield track
                seen += 1
            offset += MAX_LIMIT
