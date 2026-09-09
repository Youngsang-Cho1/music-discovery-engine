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


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)

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
