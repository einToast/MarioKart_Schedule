from schedulers.v5.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SECONDS,
    DEFAULT_POLISH_STEPS,
    resolve_seed,
)
from schedulers.v5.optimizer import ScheduleOptimizer
from schedulers.v5.utils import (
    create_team_list,
    get_unrated_games,
    rating_game_counts,
    resolve_field_sizes,
)


def create_plan(
    team_list,
    number_fields=4,
    number_rounds=8,
    teams_per_game=4,
    seed=None,
    max_seconds=DEFAULT_MAX_SECONDS,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    polish_steps=DEFAULT_POLISH_STEPS,
):
    """Create a plan ``[round][field][team]``.

    ``teams_per_game`` is one number for all fields or a list with the
    number of teams per game for every field, e.g. ``[4, 4, 2]``. A team
    plays at most once per round, so there must be at least as many teams as
    one round has seats.
    """
    if not team_list:
        return []
    if number_rounds <= 0:
        raise ValueError("number_rounds must be positive")
    field_sizes = resolve_field_sizes(number_fields, teams_per_game)

    teams = list(team_list)
    if len(teams) < sum(field_sizes):
        raise ValueError("Not enough teams to fill every field of a round")
    total_slots = sum(field_sizes) * number_rounds
    if len(teams) > total_slots:
        raise ValueError("Not enough game slots for every team to play")

    optimizer = ScheduleOptimizer(
        teams=teams,
        field_sizes=field_sizes,
        number_rounds=number_rounds,
        seed=resolve_seed(seed, len(teams), number_rounds, field_sizes),
        max_seconds=max_seconds,
        max_iterations=max_iterations,
        polish_steps=polish_steps,
    )
    return optimizer.create_plan()


def generate_plan(num_teams=25, num_fields=4, num_rounds=8, num_teams_per_game=4):
    team_list = create_team_list(num_teams)
    plan = create_plan(team_list, num_fields, num_rounds, num_teams_per_game)
    rate_plan = get_unrated_games(plan)
    max_games_count = rating_game_counts(plan, rate_plan)
    return plan, max_games_count
