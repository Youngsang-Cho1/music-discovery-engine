"""End-to-end API tests over the real database with a faked vector store."""
import pytest

from tests.conftest import requires_postgres

pytestmark = requires_postgres


# --- catalog -----------------------------------------------------------


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_list_and_fetch_tracks(client, seeded):
    listed = client.get("/tracks?limit=10").json()
    assert {t["id"] for t in listed} == set(seeded)

    one = client.get("/tracks/a1").json()
    assert one["artist_name"] == "artist-1"
    assert one["tags"] == ["rock"]


def test_unknown_track_is_404(client, seeded):
    assert client.get("/tracks/does-not-exist").status_code == 404


def test_track_search_filters_by_name_and_artist(client, seeded):
    assert len(client.get("/tracks?q=artist-6").json()) == 4
    assert len(client.get("/tracks?q=Track a1").json()) == 1


# --- recommendations ---------------------------------------------------


def test_recommend_returns_same_cluster(client, seeded):
    body = client.get("/recommend/track/a1?force_variant=control&top_k=3").json()

    assert body["seed_track_id"] == "a1"
    ids = [r["track"]["id"] for r in body["results"]]
    assert "a1" not in ids, "the seed must not recommend itself"
    assert all(i.startswith("a") for i in ids), f"cross-cluster leakage: {ids}"


def test_recommend_reports_every_signal(client, seeded):
    body = client.get("/recommend/track/a1?force_variant=control&top_k=1").json()
    signals = body["results"][0]["signals"]
    assert set(signals) == {"audio", "descriptor", "tag", "novelty"}


def test_recommend_for_unknown_track_is_404(client, seeded):
    assert client.get("/recommend/track/nope").status_code == 404


def test_artist_cap_limits_one_artist(client, seeded):
    """Seeding from c1 leaves three same-artist candidates ahead of everything
    else; without the cap all three would fill the list."""
    body = client.get("/recommend/track/c1?force_variant=control&top_k=3").json()
    artists = [r["track"]["artist_name"] for r in body["results"]]
    assert artists.count("artist-6") == 2
    assert len(body["results"]) == 3


def test_tag_filter_restricts_candidates(client, seeded):
    body = client.get("/recommend/track/a1?force_variant=control&tags=jazz").json()
    ids = [r["track"]["id"] for r in body["results"]]
    assert ids and all(i.startswith("b") for i in ids)


def test_discovery_arm_surfaces_the_long_tail(client, seeded):
    """a4 has 5 plays against a2's 500k and sits slightly further away.

    Under `control` only distance matters, so a4 ranks last of the cluster;
    under `discovery` the novelty term should pull it up. This is the project's
    core claim reduced to one assertion — if a popularity-blind index can't
    surface the long tail, nothing else here matters.
    """
    control = client.get("/recommend/track/a1?force_variant=control&top_k=3").json()
    discovery = client.get("/recommend/track/a1?force_variant=discovery&top_k=3").json()

    control_ids = [r["track"]["id"] for r in control["results"]]
    discovery_ids = [r["track"]["id"] for r in discovery["results"]]

    assert control_ids == ["a2", "a3", "a4"], "control should rank purely by distance"
    assert discovery_ids[0] == "a4", f"novelty failed to surface a4: {discovery_ids}"


def test_text_search_returns_503_when_disabled(client, seeded, monkeypatch):
    """A deployment without the model must degrade to one unavailable route,
    not a 500 and not a broken service."""
    from app.embeddings.text_encoder import TextSearchUnavailable
    from app.services import recommender

    def unavailable(_query):
        raise TextSearchUnavailable("Text search is disabled on this deployment")

    monkeypatch.setattr(recommender, "encode_query", unavailable)

    resp = client.get("/recommend/text?q=rainy%20piano")
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"]

    # Everything else keeps working.
    assert client.get("/recommend/track/a1").status_code == 200


def test_variant_is_echoed_for_attribution(client, seeded):
    body = client.get("/recommend/track/a1?force_variant=diverse").json()
    assert body["variant"] == "diverse"


def test_vectors_are_only_fetched_when_mmr_needs_them(client, seeded, store):
    """`control` has mmr_lambda == 1.0, so pulling vectors would be wasted
    bandwidth on every request."""
    store.search_calls.clear()

    client.get("/recommend/track/a1?force_variant=control")
    assert store.search_calls[-1]["with_vectors"] is False

    client.get("/recommend/track/a1?force_variant=diverse")
    assert store.search_calls[-1]["with_vectors"] is True


def test_internal_vector_never_reaches_the_response(client, seeded):
    body = client.get("/recommend/track/a1?force_variant=diverse&top_k=2").json()
    for rec in body["results"]:
        assert set(rec) == {"track", "relevance", "signals"}
        assert "_vector" not in rec["track"]


# --- events and the bandit --------------------------------------------


def test_event_is_accepted(client, seeded, session_id):
    resp = client.post(
        "/events",
        json={
            "session_id": session_id,
            "variant": "control",
            "track_id": "a2",
            "action": "impression",
            "position": 0,
        },
    )
    assert resp.status_code == 202


def test_invalid_action_is_rejected(client, seeded, session_id):
    resp = client.post(
        "/events",
        json={
            "session_id": session_id,
            "variant": "control",
            "track_id": "a2",
            "action": "teleported",
        },
    )
    assert resp.status_code == 422


def _post_outcome(client, session_id, variant, action, dwell_ms):
    return client.post(
        "/events",
        json={
            "session_id": session_id,
            "variant": variant,
            "track_id": "a2",
            "action": action,
            "dwell_ms": dwell_ms,
        },
    )


def _arm(client, variant):
    report = client.get("/experiments/report").json()
    return next(a for a in report["arms"] if a["variant"] == variant)


def test_completion_raises_the_posterior(client, seeded, session_id):
    before = _arm(client, "control")["posterior_mean"]
    _post_outcome(client, session_id, "control", "complete", 60_000)
    assert _arm(client, "control")["posterior_mean"] > before


def test_fast_skip_lowers_the_posterior(client, seeded, session_id):
    before = _arm(client, "control")["posterior_mean"]
    _post_outcome(client, session_id, "control", "skip", 2_000)
    assert _arm(client, "control")["posterior_mean"] < before


def test_late_skip_is_not_counted_against_the_arm(client, seeded, session_id):
    """Skipping at 2:30 means the listener heard the track, not that the
    recommendation was wrong."""
    before = _arm(client, "control")["posterior_mean"]
    _post_outcome(client, session_id, "control", "skip", 150_000)
    assert _arm(client, "control")["posterior_mean"] == pytest.approx(before)


def test_probability_best_is_a_distribution(client, seeded):
    report = client.get("/experiments/report").json()
    total = sum(a["probability_best"] for a in report["arms"])
    assert total == pytest.approx(1.0, abs=1e-6)


def test_untouched_arms_start_uniform(client, seeded):
    """Beta(1,1) priors — every arm equally likely before any evidence."""
    report = client.get("/experiments/report").json()
    assert report["total_impressions"] == 0
    for arm in report["arms"]:
        assert arm["posterior_mean"] == pytest.approx(0.5)


def test_impressions_are_counted_per_request(client, seeded):
    client.get("/recommend/track/a1?force_variant=control&top_k=3")
    assert _arm(client, "control")["impressions"] == 3


def test_event_summary_groups_by_variant_and_action(client, seeded, session_id):
    _post_outcome(client, session_id, "control", "complete", 60_000)
    _post_outcome(client, session_id, "control", "skip", 1_000)
    _post_outcome(client, session_id, "diverse", "complete", 60_000)

    rows = client.get("/experiments/events/summary").json()
    counts = {(r["variant"], r["action"]): r["count"] for r in rows}
    assert counts[("control", "complete")] == 1
    assert counts[("control", "skip")] == 1
    assert counts[("diverse", "complete")] == 1
