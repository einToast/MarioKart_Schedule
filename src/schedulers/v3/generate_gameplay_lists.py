from schedulers.v3.config import DEFAULT_MAX_SECONDS
from schedulers.v3.scheduler import create_plan, generate_plan
from schedulers.v3.utils import (
    create_team_list,
    flatten,
    get_unrated_games,
)

__all__ = [
    "DEFAULT_MAX_SECONDS",
    "create_plan",
    "create_team_list",
    "flatten",
    "generate_plan",
    "get_unrated_games",
]


if __name__ == "__main__":
    generated_plan, generated_max_games_count = generate_plan()
    print(generated_plan)
    print(generated_max_games_count)
