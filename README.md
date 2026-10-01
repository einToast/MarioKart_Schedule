# MarioKart Tournament Schedule Service

Flask webserver that generates balanced tournament schedules for 4-player Mario Kart matches. It exposes HTTP endpoints for health checks and schedule generation. The schedulers try to minimize repeated duels between teams and to balance appearances on field A (Main Switch) across teams.

## Tech Stack

- Python 3.14
- Flask, served by Gunicorn in Docker
- Docker

## Project Structure

```
src/
  webserver.py              Flask app: /healthcheck and /schedule
  schedulers/
    v1/                     Original scheduler (randomized search with curated seeds)
    v2/                     Current scheduler (time-boxed optimizer), default
tests/                      unittest suite (schedulers and API)
Dockerfile                  Production image (Gunicorn)
.github/workflows/          Tests, webserver smoke test, Docker build
```

Both schedulers return the same plan shape, so the API treats them interchangeably.

| Version | Approach |
|---------|----------|
| `1` | Repeated randomized search. Teams are paired by fewest prior meetings, and rounds are then reordered to balance Main Switch appearances. Uses curated starting seeds for 15 to 25 teams. |
| `2` (default) | Builds a plan greedily with a cost function, then improves it with pairwise swaps for a fixed time budget (0.35 s by default). Output is deterministic for the same inputs. |

## Getting Started

### Prerequisites

- Python 3.14

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/webserver.py
```

The service listens on `http://127.0.0.1:8000` by default. To change that, set `FLASK_RUN_HOST` and `FLASK_RUN_PORT`.

### Run the tests

```bash
python -m unittest discover -s tests
```

### Dev container

The repo includes a VS Code dev container (Python 3.14). It creates `.venv`, installs the requirements and forwards port 8000.

## API

### `GET /healthcheck`

Returns `200` with the body `OK`.

### `POST /schedule`

Request body (JSON):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_teams` | int | required | Number of teams. |
| `version` | int | `2` | Scheduler version, `1` or `2`. |
| `num_fields` | int | `4` | Fields (games) per round. |
| `num_rounds` | int | `8` | Number of rounds. |
| `num_teams_per_game` | int | `4` | Teams per field. Ignored by version `1`, which always uses 4. |

Example:

```bash
curl -X POST http://localhost:8000/schedule \
  -H 'Content-Type: application/json' \
  -d '{"num_teams": 25}'
```

Response (truncated):

```json
{
  "plan": [
    [[0, 13, 6, 5], [23, 3, 9, 15], [14, 8, 24, 12], [20, 10, 11, 1]]
  ],
  "max_games_count": 6,
  "version": 2,
  "num_teams": 25,
  "num_fields": 4,
  "num_rounds": 8,
  "num_teams_per_game": 4
}
```

- `plan` is a nested list `[round][field][team]`. Team IDs are 0-based, so `num_teams=25` gives IDs `0` to `24`. Field index 0 is the Main Switch field.
- `max_games_count` is the number of games each team plays that count towards the rating. Games a team plays beyond the minimum are excluded.
- The other fields echo the parameters used.

Errors return `400` with a JSON body:

- `{"error": "num_teams is required"}`
- `{"error": "version must be one of: 1, 2"}`
- `{"error": "Invalid input parameters"}`, for example when the teams do not fit into the available game slots.

## Benchmarking (v2)

Measure schedule quality across team counts. Run from the repository root:

```bash
PYTHONPATH=src python -m schedulers.v2.benchmark --team-min 15 --team-max 25
```

Add `--compare-old` to compare against v1. Other options are `--fields`, `--rounds`, `--teams-per-game` and `--max-seconds`.

## Docker

Build and run the image:

```bash
docker build -t mariokart-schedule .
docker run --rm -p 8000:8000 mariokart-schedule
```

The container runs Gunicorn (one worker) on port 8000 as a non-root user, with a health check on `/healthcheck`.

For the full stack, see [MarioKart_Deployment](https://github.com/einToast/MarioKart_Deployment).
