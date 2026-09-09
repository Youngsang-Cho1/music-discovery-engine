"""Standalone CLAP text-embedding service.

Exists so the API can stay small. Text search is the only route that needs the
model resident, and a torch + CLAP working set is roughly 2.5GB — enough to
force an otherwise-1GB service onto a much larger instance for one endpoint.

Deployed separately, this can be scaled (or scaled to zero) on its own, and the
API talks to it with TEXT_ENCODER_MODE=remote. Running everything in one
process is still the default for local development.

    uvicorn app.embedder.service:app --port 8100
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class EmbedRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model at startup rather than on the first request: a cold CLAP
    # load takes ~13s, which would otherwise land on a user.
    from app.embeddings.clap import embed_text

    logger.info("warming CLAP")
    embed_text(["warmup"])
    logger.info("ready")
    yield


app = FastAPI(title="CLAP text embedder", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    from app.embeddings.clap import embed_text

    vector = embed_text([request.query])[0]
    return EmbedResponse(embedding=vector.tolist(), dim=int(vector.size))
