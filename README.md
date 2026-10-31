# MarioKart Tournament Schedule Service

Flask webserver that generates balanced tournament schedules for 4-player Mario Kart matches. It exposes HTTP endpoints for health checks and schedule generation. The schedulers try to minimize repeated duels between teams and to balance appearances on field A (Main Switch) across teams.

## Tech Stack

- Python 3.15
- Flask, served by Gunicorn in Docker
- Docker

## Project Structure

```
src/
  webserver.py              Flask app: /healthcheck and /schedule
  schedulers/
    v1/                     Original scheduler (randomized search with curated seeds)
    v2/                     Fewest repeated duels first, default
    v3/                     Simulated annealing with a fixed iteration count
    v4/                     Like v3, with a number of teams per game for every field
    v5/                     Simulated annealing from a random start, like v4 in its options
tests/                      unittest suite (schedulers and API)
Dockerfile                  Production image (Gunicorn)
.github/workflows/          Tests, webserver smoke test, Docker build
```

All schedulers return the same plan shape, so the API treats them interchangeably.

| Version | Approach |
|---------|----------|
| `1` | Repeated randomized search. Teams are paired by fewest prior meetings, and rounds are then reordered to balance Main Switch appearances. Uses curated starting seeds for 15 to 25 teams. |
| `2` (default) | Puts few repeated duels first and even field use second. With at least as many teams as one round has seats it uses the same kind of search as version `5`, with a fixed number of iterations, and gives the same plan for the same inputs on every machine. Smaller tournaments, where some teams play twice in a round, keep the older greedy build and swap search with a time budget (1 s by default). |
| `3` | Starts like version `2`, then improves the plan with simulated annealing over team and field swaps for a fixed number of iterations. Puts even field use first: every team plays on every field the same number of times, give or take one, at the cost of a few more repeated duels than version `2`. Gives the same plan for the same inputs on every machine. |
| `4` | Version `3` with a number of teams per game for every field, for example two Switches with 4 controllers and one with 2. A team plays on a field in proportion to its size: with 4 and 2 teams per game, about two thirds of its games are on the larger field. |
| `5` | Takes the same options as version `4` and follows the same priorities (even field use first, then few repeated duels), but uses a new search: simulated annealing from a random start that prices every move before applying it, followed by a short tabu search. In the same time it repeats fewer duels than versions `3` and `4` in most setups, for example 20 teams on 4 Switches. Needs at least as many teams as one round has seats. Gives the same plan for the same inputs on every machine. |

## Getting Started

### Prerequisites

- Python 3.15

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/webserver.py
```

 `requirements.txt` is generated from `requirements.in`. Regenerate it with:

```bash
pip install --require-hashes --only-binary :all: -r requirements-pip-tools.txt
pip-compile --generate-hashes --output-file=requirements.txt --strip-extras requirements.in
```

The service listens on `http://127.0.0.1:8000` by default. To change that, set `FLASK_RUN_HOST` and `FLASK_RUN_PORT`.

### Run the tests

```bash
python -m unittest discover -s tests
```

### Dev container

The repo includes a VS Code dev container (Python 3.15). It creates `.venv`, installs the requirements and forwards port 8000.

## API

### `GET /healthcheck`

Returns `200` with the body `OK`.

### `POST /schedule`

Request body (JSON):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_teams` | int | required | Number of teams. |
| `version` | int | `3` | Scheduler version, `1`, `2`, `3`, `4` or `5`. |
| `num_fields` | int | `4` | Fields (games) per round. |
| `num_rounds` | int | `8` | Number of rounds. |
| `num_teams_per_game` | int, or list of int for versions `4` and `5` | `4` | Teams per field. Ignored by version `1`, which always uses 4. Versions `4` and `5` also accept a list with one entry per field, such as `[4, 4, 2]`; the first entry is the Main Switch field. |

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

Example with a different number of teams on every field (version `4`):

```bash
curl -X POST http://localhost:8000/schedule \
  -H 'Content-Type: application/json' \
  -d '{"num_teams": 20, "version": 4, "num_fields": 3, "num_teams_per_game": [4, 4, 2]}'
```

Every round of the plan then has fields with 4, 4 and 2 teams, and `num_teams_per_game` is echoed as the list.

Errors return `400` with a JSON body:

- `{"error": "num_teams is required"}`
- `{"error": "version must be one of: 1, 2, 3, 4, 5"}`
- `{"error": "Invalid input parameters"}`, for example when the teams do not fit into the available game slots, when `num_teams_per_game` is a list for a version below `4` or does not have one entry per field, or when version `5` gets fewer teams than one round has seats.

## Benchmarking (v2 to v5)

Measure schedule quality across team counts. Run from the repository root:

```bash
PYTHONPATH=src python -m schedulers.v2.benchmark --team-min 15 --team-max 25
```

Use `schedulers.v3.benchmark` for version `3` and `schedulers.v4.benchmark` or `schedulers.v5.benchmark` for versions `4` and `5`, which also take one size per field, as in `--fields 3 --teams-per-game 4,4,2`. Add `--compare-old` to compare against v1. Other options are `--fields`, `--rounds`, `--teams-per-game` and `--max-seconds`.

## Docker

Build and run the image:

```bash
docker build -t mariokart-schedule .
docker run --rm -p 8000:8000 mariokart-schedule
```

The container runs Gunicorn (one worker) on port 8000 as a non-root user, with a health check on `/healthcheck`.

For the full stack, see [MarioKart_Deployment](https://github.com/einToast/MarioKart_Deployment).
