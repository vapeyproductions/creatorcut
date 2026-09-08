FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY models ./models
RUN python -m pip install --no-cache-dir ".[semantic]" \
    && useradd --create-home --uid 10001 creatorcut \
    && mkdir -p /app/runtime /app/artifacts \
    && chown -R creatorcut:creatorcut /app/runtime /app/artifacts

USER creatorcut
CMD ["creatorcut-worker", "--database", "/app/runtime/creatorcut.sqlite", "--work-dir", "/app/runtime/work", "--model-cache", "/app/artifacts/whisper", "--semantic-cache", "/app/artifacts/huggingface"]
