# API image. Deliberately excludes torch/transformers/librosa.
#
# Only text search needs the model at request time, and it is pluggable
# (TEXT_ENCODER_MODE). Keeping ML out of this image is the difference between
# an image that fits a small instance and one that does not.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    TEXT_ENCODER_MODE=remote
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
