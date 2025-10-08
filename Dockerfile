FROM python:3.14.0-slim AS build

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m compileall .

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD curl --fail http://localhost:8000/healthcheck || exit 1

EXPOSE 8000

ENTRYPOINT ["python"]
CMD ["src/webserver.py"]
