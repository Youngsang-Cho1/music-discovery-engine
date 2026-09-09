"""FastAPI entrypoint."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_catalog, routes_events, routes_recommend, routes_tracks
from app.api.deps import get_store
from app.config import get_settings
from app.db import Base, SessionLocal, engine
from app.experiments import bandit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema creation is fine at this size; swap to Alembic migrations once the
    # catalog table starts changing under live traffic.
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        bandit.ensure_arms(db)

    try:
        get_store().ensure_collection()
    except Exception as exc:  # Qdrant down shouldn't block catalog endpoints
        logger.warning("qdrant unavailable at startup: %s", exc)

    # Load CLAP before accepting traffic when it runs in-process. Deferring it
    # to the first request means that request waits ~13s and every concurrent
    # one queues behind the same lock.
    if settings.text_encoder_mode == "local":
        try:
            from app.embeddings.clap import warm_up

            logger.info("warming CLAP (TEXT_ENCODER_MODE=local)")
            warm_up()
        except Exception as exc:
            logger.warning("CLAP warm-up failed, text search will load on demand: %s", exc)

    yield


DESCRIPTION = """
Search Creative-Commons music by **how it sounds**, not by how many times it
has been played.

Tracks are indexed by a CLAP embedding of their audio, so a track with three
plays is exactly as retrievable as one with a million. Retrieval runs in two
stages: an ANN lookup narrows the catalog to ~200 candidates, then a reranker
fuses four signals and diversifies the result.

**Every recommendation response carries the `variant` that produced it.** Echo
that value back on the events you log — it is what lets the bandit attribute
outcomes to ranking arms.

### Notes for callers

* Rate limits are per client IP: 120/min on reads, **20/min on
  `/recommend/text`** (an uncached query costs a model forward pass), 600/min
  on events. Exceeding one returns `429` with `Retry-After`.
* `/recommend/text` returns **503** on deployments that run without a text
  encoder. Every other route still works — treat it as a feature being off,
  not as an outage.
"""

TAGS_METADATA = [
    {"name": "recommend", "description": "Retrieval by text query or seed track."},
    {"name": "tracks", "description": "Catalog listing, name search, and tag filtering."},
    {"name": "catalog", "description": "Corpus-level aggregates."},
    {
        "name": "events",
        "description": "Interaction logging and the live bandit report. Events are the only "
        "ground truth the ranking is evaluated against.",
    },
]

app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_catalog.router)
app.include_router(routes_tracks.router)
app.include_router(routes_recommend.router)
app.include_router(routes_events.router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
