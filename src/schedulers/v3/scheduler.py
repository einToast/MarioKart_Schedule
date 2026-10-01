from schedulers.v3.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SECONDS,
    resolve_seed,
)
from schedulers.v3.optimizer import ScheduleOptimizer
from schedulers.v3.utils import (
    create_team_list,
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
    max_iterations=DEFAULT_MAX_ITERATIONS,
):
    if not team_list:
        return []
    if number_fields <= 0 or number_rounds <= 0 or teams_per_game <= 0:
        raise ValueError(
            "number_fields, number_rounds and teams_per_game must be positive"
        )

    teams = list(team_list)
    if len(teams) < teams_per_game:
        raise ValueError("Not enough teams to fill a single game")
    total_slots = number_fields * teams_per_game * number_rounds
    if len(teams) > total_slots:
        raise ValueError("Not enough game slots for every team to play")

    optimizer = ScheduleOptimizer(
        teams=teams,
        number_fields=number_fields,
        number_rounds=number_rounds,
        teams_per_game=teams_per_game,
        seed=resolve_seed(
            seed,
            len(teams),
            number_fields,
            number_rounds,
            teams_per_game,
        ),
        max_seconds=max_seconds,
        max_iterations=max_iterations,
    )
    return optimizer.create_plan()


def generate_plan(num_teams=25, num_fields=4, num_rounds=8, num_teams_per_game=4):
    team_list = create_team_list(num_teams)
    plan = create_plan(team_list, num_fields, num_rounds, num_teams_per_game)
    rate_plan = get_unrated_games(plan)
    max_games_count = rating_game_counts(plan, rate_plan)
    return plan, max_games_count
