# Music Discovery Engine

Search Creative-Commons music by **how it sounds**, not by how many times it has been played.

Type *"rainy day lo-fi with soft piano"* and get tracks whose audio matches that description —
including tracks with three plays, which no collaborative-filtering system can reach.

> **Status.** v2 is under active development. The v1 coursework version (NumPy PCA / K-Means over
> hand-crafted librosa features, Streamlit UI) is preserved under [`legacy/`](legacy/) — see
> [What changed from v1](#what-changed-from-v1) for why each piece was replaced.

---

## Why this exists

Recommenders built on play counts and co-listen matrices have a structural blind spot: a track
with no listening history has no neighbours, so it cannot be recommended, so it never accumulates
history. Genre tags are the usual patch, and they are far too coarse — "electronic" spans an
ambient drone and a hardstyle kick.

This engine indexes the **audio itself**. A track's position in the index is determined entirely by
what it sounds like, so a brand-new upload is as retrievable as a catalog staple on day one. That
is the differentiator, and the `discovery` ranking arm exists specifically to test whether it holds
up against listeners.

---

## Architecture

```
                  ┌──────────────────────── ingestion (offline) ─────────────────────────┐
   Jamendo API ──►  fetch CC-licensed tracks ──► decode 3 × 10s windows ──► CLAP
   (official,                                          │                      │
    licensed)                                          ▼                      ▼
                                        librosa scalar descriptors      mean-pooled
                                          (middle window)               512-d vector
                                                       │                      │
                                                       ▼                      ▼
                                                   Postgres              Qdrant (HNSW)
                                              (catalog + events)        (audio vectors)
                                          audio is discarded after embedding
                  └──────────────────────────────────────────────────────────────────────┘

                  ┌───────────────────────── serving (online) ───────────────────────────┐
   text query ──►  CLAP text encoder ──┐
                                       ├──► Qdrant ANN ──► ~200 candidates
   seed track ──►  stored vector ──────┘                        │
                                                                ▼
                                          stage 2: weighted signal fusion + MMR
                                          (audio · descriptor · tag · novelty)
                                                                │
                                                                ▼
                                                             top-K ──► Next.js player
                                                                              │
                                    Thompson-sampling bandit ◄── skip/complete events
                  └──────────────────────────────────────────────────────────────────────┘
```

**Stack** — FastAPI · Postgres · Qdrant · Redis · CLAP (`laion/larger_clap_music_and_speech`) ·
librosa · Next.js 16 / React 19 / Tailwind 4 · Docker Compose · GitHub Actions

### Two-stage retrieval

Scoring the whole catalog per request is `O(n)` and stops working the moment the catalog grows.
Stage 1 is an HNSW approximate-nearest-neighbour lookup in Qdrant that narrows the catalog to 200
candidates in sublinear time, with genre and tempo filters pushed into the index rather than
applied as a post-hoc mask. Stage 2 does the expensive part — per-signal scoring and diversity —
on those 200 only. This is the standard candidate-generation / reranking split used in production
recommenders, and it is what makes latency independent of catalog size.

### Why CLAP, and not the v1 features

The claim that a pretrained embedding beats hand-crafted statistics needed
checking, because CLAP is the only reason the deployment needs a 5.92 GB image
and a separately-scaled service. Both representations were built from the same
three windows of the same 119 tracks, with v1's z-score normalization applied
to the librosa vector so its "cosine collapse" fix was in place
(`scripts/compare_representations.py`):

| Representation | Same-artist discrimination | Tag correlation |
| --- | --- | --- |
| **CLAP (512-d, pooled)** | **90.8%** | **0.221** |
| librosa (56-d, v1) | 71.0% | 0.058 |

*Same-artist discrimination: does a same-artist pair rank closer than a random
pair? 50% is chance. Tag correlation: Spearman ρ between embedding similarity
and Jaccard overlap on human-assigned Jamendo tags, which neither representation
ever saw.*

The tag correlation is the telling one: at ρ = 0.058 the librosa vector has
essentially no relationship to how people actually group music. CLAP earns its
cost on retrieval quality alone — natural-language search is then free, since
the model is already loaded.

### One window is not a song

CLAP's feature extractor is hard-capped at 480,000 samples — exactly 10s at
48 kHz — and its default truncation picks a *random* window. So "embed the
whole track" is not an option the model offers; the only question is which 10
seconds, and how many.

Measured on real tracks, two 10s windows of the *same* song score as low as
**0.0 cosine** against each other — the same magnitude as the gap between
unrelated tracks. A single window is a high-variance sample of a song, not a
description of it. Embedding several windows and mean-pooling fixes it:

| Windows | Self-agreement across halves of the same track (mean) | Worst case |
| --- | --- | --- |
| 1 (middle) | 0.710 | **0.394** |
| 3 (pooled) | 0.853 | **0.821** |

Mean rather than max pooling: the windows are alternative views of one track,
so averaging unit vectors moves the result toward whatever they agree on while
the parts they disagree about cancel. Max would just select one window back out
and reintroduce the variance.

**Changing this splits the index.** Vectors built the old way and the new way
still occupy the same space and still return a cosine, so nothing errors — the
similarities are simply wrong across the boundary, and the same track's two
representations agree at only ~0.56–0.89. Re-ingesting doesn't fix it either,
since that only touches tracks the Jamendo query happens to return.
`scripts/reembed.py` walks Postgres instead, so every stored track is
refreshed. Any future change to the model, window count, or pooling has to be
followed by a full re-embed for the same reason.

### Catalog diversity

An unfiltered `popularity_total` ingest lands almost entirely in a couple of
genres. Once that head is stored, re-querying it with a tag returns nothing new
— but only for the genres the head already covered. Measured against a
1,477-track catalog built that way, per-genre popularity still had this much
untouched:

| Genre | New in top 199 | New in the next 199 |
| --- | --- | --- |
| metal, blues | 197 | 199 |
| reggae, world, classical, funk | 185–189 | 199 |
| jazz, hiphop, ambient, folk, soundtrack | 151–185 | 198–199 |
| **electronic, pop** | **0** | 74–190 |

Only `electronic` and `pop` were saturated, because those dominate the global
popularity ranking the first pass drew from. So the fix is to make popularity
*per genre* rather than to abandon popularity — going by release date instead
trades a well-produced catalog for whatever happened to be uploaded recently:

```bash
python -m app.ingestion.pipeline \
  --target 400 \
  --tags "metal,blues,reggae,world,classical,jazz,funk,hiphop,ambient,folk,soundtrack,rock,electronic,pop" \
  --orders "popularity_total"
```

`--orders` accepts several sorts and sweeps every tag × order pair, so
`releasedate_desc,id_asc` is available for reaching further into the long tail
once the popular head of each genre is in.

The sweep runs in one process because CLAP takes ~26s to load; a shell loop
over the CLI paid that once per combination, which dominated the actual work.
Downloads within a batch run concurrently — that alone took ingestion from
3.8 s/track to **1.9 s/track** while tripling the number of windows embedded.

### Caching

Two things in the request path are expensive and highly repeatable, and both are
memoized in Redis:

| Cached | TTL | Why |
| --- | --- | --- |
| CLAP text-query embeddings | 24h | A transformer forward pass per query. Queries are normalized for case and whitespace first, so `"Rainy Day "` and `"rainy day"` share one entry. The model never changes, so entries survive ingestion. |
| Ranked results per (seed/query, arm, filters) | 5m | Same inputs produce the same ranking. Invalidated wholesale when ingestion adds tracks. |

Measured on the 798-track catalog, `diverse` arm (MMR active), three cold/warm pairs:
**7–11 ms warm against 76–128 ms cold**, roughly a 10× reduction. Track rows are always re-read
from Postgres rather than cached with the ranking, so a metadata fix appears immediately instead
of waiting out the TTL. Every cache path degrades to a miss on Redis failure — a cache that can
take the API down with it is worse than no cache.

### Ranking signals

Stage 2 keeps four signals separate instead of collapsing them into one number:

| Signal | What it measures |
| --- | --- |
| `audio` | CLAP cosine — timbral and semantic similarity |
| `descriptor` | L1 similarity over tempo, loudness, brightness, percussiveness, dynamic range, onset density |
| `tag` | Jaccard overlap on Jamendo genre/instrument tags |
| `novelty` | log-damped inverse play count — the long-tail term |

Selection then runs **Maximal Marginal Relevance** with a per-artist cap. Without it, a seed
track's nearest CLAP neighbours are routinely other tracks from the same album — every individual
score correct, the list useless.

Each response carries its per-signal breakdown, so the UI can say *why* a track was picked
("similar tempo & energy", "hidden gem") instead of presenting an unexplained ranking.

### Online experimentation

v1 fixed its fusion weight at `λ = 0.1`, chosen offline by asking whether same-artist pairs rank
above random pairs. That proxy is not the objective — nobody listens to a pair-discrimination rate.

v2 defines four ranking arms (`control`, `descriptor_heavy`, `diverse`, `discovery`) and assigns
them by **Thompson sampling**. Reward is binary per impression: did the listener keep the track past
the skip threshold? Beta posteriors update per outcome and traffic shifts toward winning arms
automatically — a fixed 25/25/25/25 split would keep paying full price for arms already visibly
losing. `/experiments` reports each arm's posterior mean, 95% credible interval, and Monte-Carlo
**P(best)**, which is the quantity a ship decision should rest on; comparing raw conversion rates
would call a 5/10 arm a winner over a 400/1000 arm.

`control` reproduces v1's behaviour exactly, so every other arm is measured against the shipped
baseline rather than against nothing.

---

## What changed from v1

| | v1 (coursework) | v2 (this) | Why |
| --- | --- | --- | --- |
| **Audio source** | YouTube via `yt-dlp`, with randomized sleeps to avoid bot detection | Jamendo API | Scraping YouTube audio violates its ToS and carries no redistribution right. Tolerable for a local class project; not something you can deploy publicly. Jamendo serves CC-licensed audio through an official API with hosted stream URLs. |
| **Audio storage** | every `.m4a` kept on disk | nothing retained — decode, embed, discard | No copyrightable artifact is stored; playback points at Jamendo's CDN. |
| **Representation** | 56 hand-picked librosa statistics | CLAP embeddings (512-d) | Hand-crafted statistics are only meaningful relative to each other. CLAP shares one space with text, which is what makes natural-language search possible at all. |
| **Metadata** | Spotify `danceability` / `energy` / `valence` | descriptors computed from the waveform | Spotify features are undefined for every track outside their catalog, including all of ours. |
| **Fusion** | `meta_mse + λ·(1 − cos)`, λ frozen offline | four separate signals, weights as bandit arms | One scalar can't express "acoustically close *but* tonally varied", and the proxy λ was tuned on isn't the goal. |
| **Search index** | full NumPy scan of the catalog | Qdrant HNSW | Linear scan is fine at 16k tracks and hopeless past that. |
| **Grouping** | K-Means++ with `K` hand-set to 8 | ANN neighbourhoods + MMR | `K` was a guess, and Voronoi boundaries split genuinely adjacent tracks. |
| **Interface** | Streamlit | Next.js + FastAPI | Streamlit is a prototyping tool — no custom UX, no real session model, no path to horizontal scaling. |
| **Evaluation** | offline same-artist pair discrimination (83.3%) | online skip/completion with bandit posteriors | An offline proxy can't tell you whether anyone wanted the recommendation. |

The v1 implementation stands on its own terms — PCA and K-Means++ were written from scratch in
NumPy to satisfy a no-black-box requirement, and that code is intact in [`legacy/`](legacy/).
What changed is the target: a graded artifact and a deployed service have different constraints.

---

## Running locally

```bash
# 1. Infrastructure
docker compose up -d postgres qdrant redis

# 2. Backend
cd backend
cp .env.example .env               # add your Jamendo client id
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt   # ML deps for local text search
uvicorn app.main:app --reload      # http://localhost:8000/docs

# 3. Populate the catalog (downloads audio, embeds it, discards the audio)
python -m app.ingestion.pipeline --target 1000

# 4. Frontend
cd ../frontend
cp .env.example .env.local
npm install && npm run dev         # http://localhost:3000
```

A free Jamendo `client_id` takes about a minute to obtain at
[devportal.jamendo.com](https://devportal.jamendo.com/).

Or bring the whole stack up at once:

```bash
JAMENDO_CLIENT_ID=... docker compose up --build
```

Without a Jamendo key, `python -m scripts.seed_demo --tracks 300` fills the catalog with synthetic
tracks. It exercises the wiring but the vectors carry no acoustic meaning, so text search returns
nothing sensible against it.

### Helper scripts

| Script | Purpose |
| --- | --- |
| `scripts/seed_demo.py` | Synthetic catalog, for running without a Jamendo key |
| `scripts/simulate_traffic.py` | Drives synthetic listeners so the bandit has outcomes to learn from — the experiments page shows four identical priors until something generates traffic |
| `scripts/reembed.py` | Re-embeds every stored track after an embedding change |
| `scripts/compare_representations.py` | CLAP vs. the v1 librosa vector on the same audio |

`simulate_traffic.py` gives each arm a hidden true reward rate the bandit is never told, then
reports whether the posteriors found the best one. It paces itself under the API's own rate limit
rather than rotating addresses to slip past it.

## Deployment shape

Text search is the only route that needs CLAP resident at request time — seed-track
recommendation reads vectors computed during ingestion. That asymmetry drives the split:

| Image | Contains | Size (measured) | Runs |
| --- | --- | --- | --- |
| `Dockerfile` | API only, no torch | **739 MB** | The whole API |
| `Dockerfile.ml` | + torch (CPU build), transformers, librosa, baked CLAP weights | **5.92 GB** | Embedder service, ingestion jobs |

`TEXT_ENCODER_MODE` picks how the API resolves a query:

* `local` — load CLAP in-process (development, or an instance with the RAM to spare)
* `remote` — POST to the embedder service, scaled independently
* `disabled` — `/recommend/text` returns **503** and every other route keeps working

Loading torch into the API just to serve one endpoint is what forces an otherwise-small service
onto a large instance. Splitting it means the API fits a 1 GB instance, and the embedder can scale
to zero between queries.

The API image is **8× smaller**. Verified end-to-end against both built images:

* API image: `import torch` → `ModuleNotFoundError`, and `/recommend/track/{id}` serves normally.
* `TEXT_ENCODER_MODE=disabled` → `/recommend/text` returns `503 Text search is disabled on this
  deployment`, every other route unaffected.
* `TEXT_ENCODER_MODE=remote` → the torch-free API answers a text query correctly by calling the
  embedder container, returning L2-normalized 512-d vectors from `POST /embed`.

### Load testing

`backend/loadtest/locustfile.py` models a listener session — browsing, opening similar-track pages,
occasional text queries, and a steady stream of impression/skip/complete events. Each simulated
user carries a distinct forwarded address, so the run measures service capacity rather than how
fast the rate limiter can return 429.

50 concurrent users, 60s, against a 798-track catalog. Single uvicorn process on a laptop that is
simultaneously running Postgres, Qdrant and Redis in Docker — so these are a floor, not a ceiling:

| Endpoint | p50 | p95 | p99 |
| --- | --- | --- | --- |
| `/recommend/track/{id}` | 84 ms | 270 ms | 1600 ms |
| `/recommend/text` | 15 ms | 110 ms | 1200 ms |
| `POST /events` | 15 ms | 90 ms | 1400 ms |
| **Aggregate** (2132 reqs, 35.6 req/s) | **19 ms** | **200 ms** | 1400 ms |

**0 failures.**

**The first run did not look like this.** It produced 773 `ConnectionRefused` errors because the
process died partway through, and the load test is the only thing that found out why: `_load_model`
was memoized with `@lru_cache`, which caches the *result* but holds no lock while the function
runs. FastAPI serves sync handlers from a threadpool, so a burst of cold text-search requests all
missed the cache and each began loading its own ~1.5GB copy of CLAP. Reproduced in isolation: 32
concurrent callers construct the model **32 times**. It is now double-checked locking plus a
startup warm-up, with a regression test that fails if concurrent callers ever load it more than
once.

### Rate limiting

Sliding-window counters in Redis, per client IP, with per-route budgets — a seed-track
recommendation is two Qdrant calls, an uncached text query is a transformer forward pass:

| Route | Budget |
| --- | --- |
| `/tracks/*`, `/recommend/track/{id}` | 120 / min |
| `/recommend/text` | 20 / min |
| `POST /events` | 600 / min (one impression per rendered row) |

A sorted set rather than a fixed-window counter, because a fixed window lets a caller spend a full
budget at the end of one window and again at the start of the next. Verified against the running
API: requests 1–20 return 200, 21+ return 429 with `Retry-After`, and a second IP is unaffected.
Like the cache, it fails open — Redis being down must not become an API outage.

### Deploying

**API** — Railway service, root directory `backend`, reads `backend/railway.json`. Add the Postgres
and Redis plugins (Railway injects `DATABASE_URL` and `REDIS_URL`), then set:

```
QDRANT_URL=https://<cluster>.qdrant.io:6333   # Qdrant Cloud free tier
QDRANT_API_KEY=...
CORS_ORIGINS=https://<your-app>.vercel.app
TEXT_ENCODER_MODE=remote                      # or `disabled` to skip the embedder entirely
TEXT_ENCODER_URL=https://<embedder>.up.railway.app
```

**Embedder** — a second Railway service on the same repo, config path `railway.embedder.json`.
Needs ~2 GB; the healthcheck timeout is 300s because the image warms CLAP at startup.

**Ingestion** — a Railway cron job or one-off run on the embedder image:
`python -m app.ingestion.pipeline --target 5000`.

**Frontend** — Vercel, root directory `frontend`, with
`NEXT_PUBLIC_API_URL=https://<api>.up.railway.app`.

## Tests

```bash
docker compose up -d postgres      # the API tests need a real database
cd backend  && pytest tests -q     # 119 tests
cd frontend && npm test            # 11 tests
cd frontend && npm run build       # typecheck + production build
```

**Unit** (90) — signal fusion, MMR diversity and the artist cap, descriptor extraction against
synthetic signals with known ground truth, cache key construction and failure degradation, the
Jamendo client's empty-page retry, waveform decoding from truncated downloads, and the
Beta-posterior math including the small-sample case a naive rate comparison gets wrong.

**Integration** (29) — the full API against real Postgres, with the vector store faked in-process.
Postgres is real because the ingestion and bandit paths use `ON CONFLICT`, so testing on SQLite
would skip the SQL most worth checking. Covered: cluster recovery, the artist cap, tag filtering,
`_vector` never leaking into a response, vectors being requested only when the selected arm
actually diversifies, text search degrading to 503 rather than 500, and each event type moving the
posterior the right way — including that a late skip is *not* counted against the arm.

The frontend suite covers the display logic that has real branching — duration formatting, the
"why this track" label, and descriptor translation including clamping, since a beat-tracking octave
error would otherwise render a bar wider than its container.

The backend suite runs against a dedicated `mde_test` database and refuses to start if pointed at the
development one, because it truncates between tests. It skips automatically if Postgres isn't
reachable.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /recommend/text?q=...` | Natural-language mood search |
| `GET /recommend/track/{id}` | Seed-track recommendations |
| `GET /tracks?q=&tags=` | Catalog listing, name search, tag filter |
| `GET /tracks/random`, `/tracks/{id}` | Seed picks and single-track lookup |
| `GET /catalog/stats` | Corpus size, artist count, tag distribution |
| `POST /events` | Log an impression / play / skip / complete / like |
| `GET /experiments/report` | Per-arm posteriors and P(best) |

### Pages

| Route | What it does |
| --- | --- |
| `/` | Natural-language search, with example prompts and seed tracks |
| `/browse` | Catalog listing with name search and tag filters |
| `/track/[id]` | Seed track, its audio profile, and what sounds like it |
| `/experiments` | Live bandit posteriors, credible intervals, P(best) |

Every result row can be expanded to show its four-signal breakdown, so a
ranking is explainable rather than asserted — the same numbers the reranker
used, not a post-hoc rationalization.

## Licensing

All audio is Creative-Commons licensed and streamed from Jamendo, with attribution and a link to
the specific CC deed shown in the player. No audio is redistributed or stored by this service.
