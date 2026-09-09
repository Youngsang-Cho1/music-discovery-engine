"""Jamendo API client.

Replaces the v1 yt-dlp pipeline. Jamendo publishes Creative-Commons licensed
tracks with an official API and hosted stream URLs, so both the ingestion and
the playback path in the deployed service are licensed — unlike scraping
YouTube audio, which violates its Terms of Service and has no redistribution
right attached.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)

# Jamendo caps a single page at 200 rows.
MAX_LIMIT = 200

# ...but a multi-value parameter (e.g. `id`) is capped at 50 values, which is a
# separate and much lower limit. Exceeding it is a hard API error, not a
# truncated response.
MAX_MULTI_VALUE = 50

# Jamendo intermittently answers a valid query with `status: success` and zero
# rows — measured at roughly 20-80% of requests depending on how hard we have
# been hitting it, independent of limit and offset. Since an empty page is also
# the legitimate end-of-catalog signal, the two are indistinguishable from one
# response; only a repeat request tells them apart.
EMPTY_PAGE_RETRIES = 4
EMPTY_PAGE_BACKOFF = 2.0


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
    # Play count. The `novelty` ranking signal is defined against this, so
    # without it the discovery arm scores every track identically and the
    # long-tail claim quietly stops being tested.
    listens: int = 0
    favorites: int = 0

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
            listens=int((row.get("stats") or {}).get("rate_listened_total") or 0),
            favorites=int((row.get("stats") or {}).get("favorited") or 0),
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
            "include": "musicinfo licenses stats",
            "imagesize": 300,
        }
        if tags:
            params["fuzzytags"] = tags

        # Re-ask on an empty page before believing it. Without this the very
        # first page can come back empty and the whole ingestion exits having
        # done nothing, reporting success.
        rows: list[dict[str, Any]] = []
        for attempt in range(EMPTY_PAGE_RETRIES):
            rows = await self._get("tracks", params)
            if rows:
                break
            if attempt < EMPTY_PAGE_RETRIES - 1:
                logger.warning(
                    "empty page at offset=%s (attempt %d/%d), retrying",
                    offset,
                    attempt + 1,
                    EMPTY_PAGE_RETRIES,
                )
                await asyncio.sleep(EMPTY_PAGE_BACKOFF * (attempt + 1))

        if not rows:
            logger.info("offset=%s empty after %d attempts", offset, EMPTY_PAGE_RETRIES)

        tracks = [JamendoTrack.from_api(r) for r in rows]
        return [t for t in tracks if t.audio_url]

    async def fetch_stats(self, track_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Play counts for specific track ids, keyed by id.

        Metadata-only: no audio is fetched, so refreshing popularity across the
        whole catalog costs one request per 200 tracks.
        """
        out: dict[str, dict[str, Any]] = {}

        # Chunked to MAX_MULTI_VALUE: the `id` parameter takes at most 50
        # values regardless of the row limit.
        for start in range(0, len(track_ids), MAX_MULTI_VALUE):
            chunk = track_ids[start : start + MAX_MULTI_VALUE]
            if not chunk:
                continue
            rows = await self._get(
                "tracks",
                {
                    "id": " ".join(chunk),
                    "limit": MAX_MULTI_VALUE,
                    "include": "stats",
                },
            )
            out.update({str(r["id"]): (r.get("stats") or {}) for r in rows})

        return out

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
