from schedulers.v2.config import DEFAULT_MAX_SECONDS
from schedulers.v2.scheduler import create_plan, generate_plan
from schedulers.v2.utils import (
    count_in_list,
    create_team_list,
    flatten,
    get_unrated_games,
)


if __name__ == "__main__":
    generated_plan, generated_max_games_count = generate_plan()
    print(generated_plan)
    print(generated_max_games_count)
