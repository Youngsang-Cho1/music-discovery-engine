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
   Jamendo API ──►  fetch CC-licensed tracks ──► decode 10s window ──► CLAP encoder
   (official,                                          │                     │
    licensed)                                          ▼                     ▼
                                        librosa scalar descriptors     512-d embedding
                                                       │                     │
                                                       ▼                     ▼
                                                   Postgres              Qdrant (HNSW)
                                              (catalog + events)        (audio vectors)
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
pip install -r requirements.txt
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

## Tests

```bash
cd backend  && pytest tests -q     # ranking, descriptors, bandit statistics
cd frontend && npm run build       # typecheck + production build
```

The backend suite covers the pure logic — signal fusion, MMR diversity and the artist cap,
descriptor extraction against synthetic signals with known ground truth, and the Beta-posterior
math including the small-sample case a naive rate comparison gets wrong.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /recommend/text?q=...` | Natural-language mood search |
| `GET /recommend/track/{id}` | Seed-track recommendations |
| `GET /tracks`, `/tracks/random`, `/tracks/{id}` | Catalog browsing |
| `POST /events` | Log an impression / play / skip / complete / like |
| `GET /experiments/report` | Per-arm posteriors and P(best) |

## Licensing

All audio is Creative-Commons licensed and streamed from Jamendo, with attribution and a link to
the specific CC deed shown in the player. No audio is redistributed or stored by this service.
