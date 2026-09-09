"""Jamendo client tests, focused on its response quirks.

The API intermittently answers a perfectly valid query with `status: success`
and zero rows. An empty page is also how the end of the catalog looks, so a
single response cannot distinguish "transient blip" from "no more data" — and
guessing wrong the first way makes an entire ingestion run exit reporting
success having stored nothing.
"""
import pytest

from app.ingestion.jamendo import JamendoClient, JamendoTrack

ROW = {
    "id": 42,
    "name": "Test Track",
    "artist_id": 7,
    "artist_name": "Test Artist",
    "album_name": "Test Album",
    "duration": 210,
    "releasedate": "2020-01-01",
    "audio": "https://example.invalid/42.mp3",
    "image": "https://example.invalid/42.jpg",
    "license_ccurl": "https://creativecommons.org/licenses/by/3.0/",
    "shareurl": "https://example.invalid/share/42",
    "musicinfo": {
        "speed": "medium",
        "acousticelectric": "electric",
        "vocalinstrumental": "vocal",
        "lang": "en",
        "tags": {
            "genres": ["rock", "pop"],
            "instruments": ["guitar"],
            "vartags": ["energetic", "rock"],
        },
    },
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JAMENDO_CLIENT_ID", "test-client-id")
    from app.config import get_settings

    get_settings.cache_clear()
    yield JamendoClient(client_id="test-client-id")
    get_settings.cache_clear()


# --- parsing -----------------------------------------------------------


def test_parses_a_row():
    track = JamendoTrack.from_api(ROW)
    assert track.id == "42"
    assert track.artist_name == "Test Artist"
    assert track.duration_sec == 210
    assert track.speed == "medium"


def test_tag_groups_are_merged_and_deduplicated():
    """genres/instruments/vartags are three lists; retrieval wants one bag,
    and `rock` appears in two of them."""
    track = JamendoTrack.from_api(ROW)
    assert track.tags == ["energetic", "guitar", "pop", "rock"]


def test_missing_musicinfo_is_tolerated():
    row = {k: v for k, v in ROW.items() if k != "musicinfo"}
    track = JamendoTrack.from_api(row)
    assert track.tags == []
    assert track.speed is None


def test_ids_are_coerced_to_strings():
    """Jamendo sends ints; our primary key is a string, and a mismatch would
    create duplicate rows for one track."""
    track = JamendoTrack.from_api(ROW)
    assert isinstance(track.id, str)
    assert isinstance(track.artist_id, str)


# --- the empty-page quirk ---------------------------------------------


@pytest.mark.asyncio
async def test_a_transient_empty_page_is_retried(client, monkeypatch):
    responses = [[], [], [ROW]]
    calls = 0

    async def fake_get(_path, _params):
        nonlocal calls
        calls += 1
        return responses.pop(0)

    monkeypatch.setattr(client, "_get", fake_get)
    monkeypatch.setattr("app.ingestion.jamendo.EMPTY_PAGE_BACKOFF", 0)

    tracks = await client.fetch_tracks(limit=10)

    assert calls == 3
    assert [t.id for t in tracks] == ["42"]


@pytest.mark.asyncio
async def test_a_genuinely_empty_page_gives_up(client, monkeypatch):
    """End of catalog: retries are bounded, not infinite."""
    calls = 0

    async def fake_get(_path, _params):
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr(client, "_get", fake_get)
    monkeypatch.setattr("app.ingestion.jamendo.EMPTY_PAGE_BACKOFF", 0)

    from app.ingestion.jamendo import EMPTY_PAGE_RETRIES

    assert await client.fetch_tracks(limit=10) == []
    assert calls == EMPTY_PAGE_RETRIES


@pytest.mark.asyncio
async def test_unplayable_tracks_are_dropped(client, monkeypatch):
    """A row with no stream URL can be neither embedded nor played."""
    no_audio = {**ROW, "id": 43, "audio": ""}

    async def fake_get(_path, _params):
        return [ROW, no_audio]

    monkeypatch.setattr(client, "_get", fake_get)
    tracks = await client.fetch_tracks(limit=10)
    assert [t.id for t in tracks] == ["42"]


@pytest.mark.asyncio
async def test_iter_tracks_stops_at_the_target(client, monkeypatch):
    async def fake_get(_path, params):
        return [{**ROW, "id": params["offset"] + i} for i in range(params["limit"])]

    monkeypatch.setattr(client, "_get", fake_get)

    seen = [t async for t in client.iter_tracks(target=5)]
    assert len(seen) == 5


def test_missing_client_id_fails_loudly(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("JAMENDO_CLIENT_ID", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(Exception, match="JAMENDO_CLIENT_ID"):
            JamendoClient(client_id="")
    finally:
        get_settings.cache_clear()
