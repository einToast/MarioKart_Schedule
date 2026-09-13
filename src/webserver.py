from flask import Flask, Response, jsonify, request

from schedulers.v1.generate_gameplay_lists import (
    generate_plan as generate_plan_v1,
)
from schedulers.v2.generate_gameplay_lists import (
    generate_plan as generate_plan_v2,
)

app = Flask(__name__)
SCHEDULERS = {
    1: generate_plan_v1,
    2: generate_plan_v2,
}


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return Response("OK", status=200)


@app.route("/schedule", methods=["POST"])
def schedule():
    print("Request received")
    data = request.get_json(silent=True) or {}
    version = int(data.get("version", 2))

    if "num_teams" not in data:
        return jsonify({"error": "num_teams is required"}), 400
    if version not in SCHEDULERS:
        return jsonify({"error": "version must be one of: 1, 2"}), 400

    try:
        num_teams = int(data["num_teams"])
        num_fields = int(data.get("num_fields", 4))
        num_rounds = int(data.get("num_rounds", 8))
        num_teams_per_game = int(data.get("num_teams_per_game", 4))

        plan, max_games_count = SCHEDULERS[version](num_teams, num_fields, num_rounds)
    except (TypeError, ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400

    response_dict = {
        "plan": plan,
        "max_games_count": next(iter(max_games_count)),
        "version": version,
    }
    return jsonify(response_dict)
    # return jsonify(plan)


if __name__ == "__main__":
    app.run(debug=False, port=8000, host="0.0.0.0")
