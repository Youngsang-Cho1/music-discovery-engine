"""Load profile for the recommendation API.

    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 \
           --headless -u 50 -r 10 -t 60s

Task weights approximate a real session rather than hammering one endpoint:
a listener browses, opens a few similar-track pages, occasionally types a
query, and emits a stream of impression/skip/complete events throughout.

Two details make the numbers mean something:

* Seed ids are sampled from the live catalog at start, so every request hits a
  track that actually exists — 404s are cheap and would flatter the results.
* Each simulated user carries its own forwarded address, so "50 users" means 50
  clients rather than one client hammering a single per-IP budget. Otherwise
  the run mostly measures how fast the service returns 429.
* Text search stays a minority of the mix regardless, because it is the one
  route whose cold path is a model forward pass — over-weighting it would
  measure the embedder rather than the API.
"""
import random

from locust import HttpUser, between, events, task

SEED_TRACK_IDS: list[str] = []

# Text search is the tightest budget (20/min/IP) and the most expensive route.
# Keeping it a small share of the task mix reflects real usage — listeners
# browse and skip far more often than they type a new query.
SEARCH_WEIGHT = 3

QUERIES = [
    "rainy day lo-fi with soft piano",
    "aggressive distorted metal guitar",
    "calm gentle acoustic fingerpicking",
    "upbeat electronic dance music",
    "tense cinematic strings",
    "warm jazz brass at night",
]


@events.test_start.add_listener
def _load_seed_ids(environment, **_kwargs):
    """Pull real track ids once, before the run starts.

    Sends its own forwarded address so setup doesn't spend the budget the
    simulated users need — and so a previous run's leftover budget can't leave
    this one with an empty seed list and no similar-track traffic at all.
    """
    import time

    import httpx

    headers = {"X-Forwarded-For": "10.255.255.254"}
    last_error = None

    for attempt in range(5):
        try:
            resp = httpx.get(
                f"{environment.host}/tracks?limit=200", headers=headers, timeout=30
            )
            if resp.status_code == 429:
                last_error = "rate limited"
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            SEED_TRACK_IDS.extend(t["id"] for t in resp.json())
            break
        except Exception as exc:
            last_error = exc
            time.sleep(1)

    if not SEED_TRACK_IDS:
        raise RuntimeError(
            f"could not load seed track ids from {environment.host}: {last_error}. "
            "Is the catalog populated?"
        )
    print(f"loaded {len(SEED_TRACK_IDS)} seed track ids")


class Listener(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.session_id = f"load-{random.randint(0, 10**9)}"
        self.last_results = []
        self.variant = "control"

        # Rate limiting is per client IP, but a load generator has one. Giving
        # each simulated user a distinct forwarded address models N separate
        # listeners; without it the run mostly measures how fast the service
        # returns 429, which is a different question.
        octets = ".".join(str(random.randint(1, 254)) for _ in range(3))
        self.client.headers["X-Forwarded-For"] = f"10.{octets}"

    @task(10)
    def browse(self):
        with self.client.get(
            "/tracks/random?limit=12", name="/tracks/random", catch_response=True
        ) as resp:
            if resp.status_code == 429:
                resp.success()

    @task(20)
    def similar_tracks(self):
        """The hot path: ANN + rerank, and the one the cache serves."""
        seed = random.choice(SEED_TRACK_IDS)
        with self.client.get(
            f"/recommend/track/{seed}?top_k=20&session_id={self.session_id}",
            name="/recommend/track/{id}",
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                resp.success()  # rate limiting working as designed, not a failure
                return
            if resp.status_code == 200:
                body = resp.json()
                self.variant = body["variant"]
                self.last_results = [r["track"]["id"] for r in body["results"]]

    @task(SEARCH_WEIGHT)
    def text_search(self):
        """Cheap on a cache hit, a model forward pass on a miss."""
        with self.client.get(
            "/recommend/text",
            params={"q": random.choice(QUERIES), "top_k": 20, "session_id": self.session_id},
            name="/recommend/text",
            catch_response=True,
        ) as resp:
            if resp.status_code in (429, 503):
                resp.success()  # throttled, or no encoder on this deployment

    @task(30)
    def log_event(self):
        """Impression/skip/complete traffic — the highest-volume endpoint."""
        if not self.last_results:
            return
        track_id = random.choice(self.last_results)
        skipped = random.random() < 0.6
        dwell_ms = random.randint(500, 8000) if skipped else random.randint(60_000, 200_000)
        with self.client.post(
            "/events",
            json={
                "session_id": self.session_id,
                "variant": self.variant,
                "track_id": track_id,
                "action": "skip" if skipped else "complete",
                "dwell_ms": dwell_ms,
            },
            name="POST /events",
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                resp.success()

    @task(2)
    def experiment_report(self):
        self.client.get("/experiments/report", name="/experiments/report")
