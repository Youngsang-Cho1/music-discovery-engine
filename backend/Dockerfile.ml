# ML image: CLAP embedder service, and the ingestion pipeline.
#
# Same code, heavier dependency set. Build target chooses which one runs.
FROM python:3.12-slim

# ffmpeg + libsndfile are what librosa uses to decode the mp3 streams Jamendo
# serves; without them every decode fails at runtime rather than at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./

# CPU-only torch: the default wheel bundles CUDA and adds several GB that a
# CPU host can never use.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements-ml.txt

COPY app ./app
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models
EXPOSE 8100

# Bake the weights into the image so a cold start doesn't pull ~1.5GB from the
# Hugging Face hub before serving its first request.
RUN python -c "from transformers import ClapModel, ClapProcessor; \
    m='laion/larger_clap_music_and_speech'; \
    ClapModel.from_pretrained(m); ClapProcessor.from_pretrained(m)"

CMD ["sh", "-c", "uvicorn app.embedder.service:app --host 0.0.0.0 --port ${PORT:-8100}"]
