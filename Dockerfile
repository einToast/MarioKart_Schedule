FROM python:3.15.0-slim AS build

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --require-hashes --only-binary :all: -r requirements.txt && \
    useradd --system --no-create-home --uid 10001 schedule

COPY src/ ./src/

USER schedule

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD curl --fail http://localhost:8000/healthcheck || exit 1

EXPOSE 8000

CMD ["gunicorn", "--workers", "1", "--chdir", "/app/src", "--bind", "0.0.0.0:8000", "webserver:app"]