# MarioKart Tournament – Schedule Service

Flask webserver that generates balanced tournament schedules for 4-player Mario Kart matches. It exposes simple HTTP endpoints for health and schedule generation and can be run locally or via Docker. The scheduling core tries to minimize repeated duels and balance "field A" (Main Switch) appearances across teams.

## Tech Stack

- Language: Python 3.13
- Framework: Flask
- Deployment: Docker + Docker Compose

- Randomized schedule generation with curated RNG seeds for common team counts


## Project Structure

- `src/webserver.py`: Flask app exposing `/healthcheck` and `/schedule`.
- `src/generate_gameplay_lists.py`: schedule generation core (`generate_plan`, `create_plan`).
- `src/helper.py`, `src/team.py`: utilities and data model.
- `src/test.py`: ad-hoc script for schedule testing.

## Getting Started

### Prerequisites

- Python 3.13

### Run

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/webserver.py
```

Service runs at `http://localhost:8000`.

Generate a schedule (example for 25 teams):

```
curl -X POST http://localhost:8000/schedule \
  -H 'Content-Type: application/json' \
  -d '{"num_teams":25}'
```

### API

- `GET /healthcheck` → `200 OK` with body `OK`.
- `POST /schedule`
  - Request JSON: `{ "num_teams": <int> }`
  - Response JSON: `{ "plan": <list>, "max_games_count": <int> }`

### Response Shape

- `plan`: Nested list of shape `[round][field][team]`.
  - A "team" is an integer ID (0-based; e.g., with `num_teams=25`, teams are `0-24`).
  - Each round contains multiple fields (default 4), each with a group of teams (default 4 per field).
- `max_games_count`: Maximum number of games the least active team has played.

Example (truncated):

```
{
  "plan": [
    [ [0, 13, 6, 5], [23, 3, 9, 15], [14, 8, 24, 12], [20, 10, 11, 1] ],
    ...
  ],
  "max_games_count": 6
}
```

Notes:
- Currently fixes `num_fields=4` and `num_rounds=8` in `webserver.py`.
- For fast generation curated RNG seeds are used for common team counts; otherwise generation may take longer.

## Docker

- See [MarioKart_Deployment](https://github.com/einToast/MarioKart_Deployment)