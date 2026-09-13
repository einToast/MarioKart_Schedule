import itertools
import contextlib
import io
import argparse
import time
from collections import Counter, defaultdict

from schedulers.v2.generate_gameplay_lists import (
    DEFAULT_MAX_SECONDS,
    create_plan,
    get_unrated_games,
)


def flatten(values):
    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(flatten(value))
        else:
            result.append(value)
    return result


def pair_counts(plan, teams):
    counts = Counter(dict.fromkeys(itertools.combinations(teams, 2), 0))
    for round_plan in plan:
        for field_plan in round_plan:
            for team_a, team_b in itertools.combinations(sorted(field_plan), 2):
                counts[(team_a, team_b)] += 1
    return counts


def rating_counts(plan, rate_plan):
    counts = defaultdict(int)
    for round_idx, round_plan in enumerate(plan):
        for field_idx, field_plan in enumerate(round_plan):
            for team_idx, team in enumerate(field_plan):
                if rate_plan[round_idx][field_idx][team_idx]:
                    counts[team] += 1
    return Counter(counts)


def collect_metrics(plan, num_teams, num_fields=4, num_rounds=8, teams_per_game=4):
    teams = list(range(num_teams))
    played = Counter(flatten(plan))
    pairs = pair_counts(plan, teams)
    rate_plan = get_unrated_games(plan)
    rated = rating_counts(plan, rate_plan)
    field_counts = {team: Counter() for team in teams}
    switch_teams = set()
    duplicate_round_teams = 0
    shape_ok = len(plan) == num_rounds

    for round_plan in plan:
        shape_ok = shape_ok and len(round_plan) == num_fields
        round_teams = flatten(round_plan)
        duplicate_round_teams += len(round_teams) - len(set(round_teams))
        for field_idx, field_plan in enumerate(round_plan):
            shape_ok = shape_ok and len(field_plan) == teams_per_game
            if field_idx == 0:
                switch_teams.update(field_plan)
            for team in field_plan:
                field_counts[team][field_idx] += 1

    max_duel = max(pairs.values()) if pairs else 0
    repeated_pairs = sum(1 for count in pairs.values() if count >= 2)
    pairs_at_max = sum(1 for count in pairs.values() if count == max_duel)

    return {
        "shape_ok": shape_ok,
        "play-count set": sorted(set(played.values())),
        "rating-count set": sorted(set(rated.values())),
        "missing switch-1 teams": sorted(set(teams) - switch_teams),
        "max duel repeat": max_duel,
        "pairs at max duel": pairs_at_max,
        "duel pairs repeated 2+ times": repeated_pairs,
        "same-round duplicates": duplicate_round_teams,
        "per-team distinct fields": sorted(
            Counter(len(field_counts[team]) for team in teams).items()
        ),
        "max same-field repeats": max(
            max(field_counts[team].values()) if field_counts[team] else 0
            for team in teams
        ),
    }


def try_old_scheduler(num_teams, num_fields, num_rounds):
    try:
        from schedulers.v1.generate_gameplay_lists import (
            create_plan as old_create_plan,
        )
        from schedulers.v1.helper import create_team_liste
    except Exception as exc:
        return f"old scheduler unavailable: {exc}"

    try:
        started_at = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            plan = old_create_plan(create_team_liste(num_teams), num_fields, num_rounds)
        metrics = collect_metrics(plan, num_teams, num_fields, num_rounds)
        metrics["elapsed seconds"] = round(time.perf_counter() - started_at, 3)
        return metrics
    except Exception as exc:
        return f"old scheduler failed: {exc}"


def print_metrics(label, metrics):
    print(f"{label}:")
    if isinstance(metrics, str):
        print(f"  {metrics}")
        return
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark v2 schedule generation.")
    parser.add_argument("--team-min", type=int, default=15)
    parser.add_argument("--team-max", type=int, default=25)
    parser.add_argument("--fields", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--teams-per-game", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--compare-old", action="store_true")
    args = parser.parse_args()

    for num_teams in range(args.team_min, args.team_max + 1):
        print(f"\n{num_teams} teams")
        started_at = time.perf_counter()
        plan = create_plan(
            list(range(num_teams)),
            args.fields,
            args.rounds,
            args.teams_per_game,
            max_seconds=args.max_seconds,
        )
        metrics = collect_metrics(
            plan,
            num_teams,
            args.fields,
            args.rounds,
            args.teams_per_game,
        )
        metrics["elapsed seconds"] = round(time.perf_counter() - started_at, 3)
        print_metrics("v2", metrics)
        if args.compare_old:
            print_metrics("old", try_old_scheduler(num_teams, args.fields, args.rounds))


if __name__ == "__main__":
    main()
