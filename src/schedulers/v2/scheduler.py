from schedulers.v2.config import DEFAULT_MAX_SECONDS, resolve_seed
from schedulers.v2.dummy_team import create_plan_with_dummy_team
from schedulers.v2.optimizer import ScheduleOptimizer
from schedulers.v2.utils import (
    create_team_liste,
    get_unrated_games,
    rating_game_counts,
)


def create_plan(
    team_list,
    number_fields=4,
    number_rounds=8,
    teams_per_game=4,
    seed=None,
    max_seconds=DEFAULT_MAX_SECONDS,
):
    if not team_list:
        return []
    if number_fields <= 0 or number_rounds <= 0 or teams_per_game <= 0:
        raise ValueError("number_fields, number_rounds and teams_per_game must be positive")

    teams = list(team_list)
    capacity_per_round = number_fields * teams_per_game
    total_slots = capacity_per_round * number_rounds
    if len(teams) > total_slots:
        raise ValueError("Not enough game slots for every team to play")

    resolved_seed = resolve_seed(
        seed,
        len(teams),
        number_fields,
        number_rounds,
        teams_per_game,
    )
    if len(teams) + 1 == capacity_per_round:
        return create_plan_with_dummy_team(
            teams,
            number_fields,
            number_rounds,
            teams_per_game,
            resolved_seed,
            max_seconds,
            ScheduleOptimizer,
        )

    optimizer = ScheduleOptimizer(
        teams=teams,
        number_fields=number_fields,
        number_rounds=number_rounds,
        teams_per_game=teams_per_game,
        seed=resolved_seed,
        max_seconds=max_seconds,
    )
    return optimizer.create_plan()


def generate_plan(num_teams=25, num_fields=4, num_rounds=8):
    team_list = create_team_liste(num_teams)
    plan = create_plan(team_list, num_fields, num_rounds)
    rate_plan = get_unrated_games(plan)
    max_games_count = rating_game_counts(plan, rate_plan)
    return plan, max_games_count
