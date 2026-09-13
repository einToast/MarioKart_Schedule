DEFAULT_MAX_SECONDS = 0.35


def resolve_seed(seed, num_teams, number_fields, number_rounds, teams_per_game):
    if seed is not None:
        return seed
    return (
        num_teams * 10_007
        + number_fields * 1_009
        + number_rounds * 101
        + teams_per_game
    )
