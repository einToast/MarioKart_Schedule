from collections import Counter, defaultdict


def create_team_list(length=10):
    return list(range(length))


def flatten(values):
    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(flatten(value))
        else:
            result.append(value)
    return result


def count_in_list(values, target):
    return sum(1 for value in flatten(values) if value == target)


def create_play_targets(teams, total_slots, np_rng):
    base_games, extra_games = divmod(total_slots, len(teams))
    shuffled = [teams[idx] for idx in np_rng.permutation(len(teams))]
    extra_teams = set(shuffled[:extra_games])
    return {team: base_games + (1 if team in extra_teams else 0) for team in teams}


def resolve_field_sizes(number_fields, teams_per_game):
    """Return the number of teams per game for every field as a list.

    ``teams_per_game`` is either one number used for all fields or a list
    with one number per field. Raises ``ValueError`` if a size is not a
    positive integer or the list does not have ``number_fields`` entries.
    """
    if number_fields <= 0:
        raise ValueError("number_fields must be positive")
    if isinstance(teams_per_game, (list, tuple)):
        sizes = list(teams_per_game)
        if len(sizes) != number_fields:
            raise ValueError("teams_per_game needs one entry per field")
    else:
        sizes = [teams_per_game] * number_fields
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("teams_per_game must be positive integers")
    return sizes


def field_share_bounds(games, field_sizes):
    """Range of games a team with ``games`` games should play on each field.

    A field gets a share of the games in proportion to its size. Returns
    ``(low, high)``, the shares rounded down and up, as two lists.
    """
    capacity = sum(field_sizes)
    low = [games * size // capacity for size in field_sizes]
    high = [-(-games * size // capacity) for size in field_sizes]
    return low, high


def pair_key(team_a, team_b, team_order):
    idx_a = team_order[team_a]
    idx_b = team_order[team_b]
    if idx_a < idx_b:
        return team_a, team_b
    return team_b, team_a


def get_unrated_games(game_plan):
    teams = set(flatten(game_plan))
    played_games = Counter(flatten(game_plan))
    min_games = min(played_games.values()) if played_games else 0
    rated_games = Counter()
    rate_game_plan = []

    for round_plan in game_plan:
        rated_round = []
        for field_plan in round_plan:
            rated_field = []
            for team in field_plan:
                is_rated = rated_games[team] < min_games
                rated_field.append(is_rated)
                if is_rated:
                    rated_games[team] += 1
            rated_round.append(rated_field)
        rate_game_plan.append(rated_round)

    missing_teams = teams - set(rated_games)
    for team in missing_teams:
        rated_games[team] = 0

    return rate_game_plan


def rating_game_counts(plan, rate_plan):
    counts = defaultdict(int)
    for round_idx, round_plan in enumerate(plan):
        for field_idx, field_plan in enumerate(round_plan):
            for team_idx, team in enumerate(field_plan):
                if rate_plan[round_idx][field_idx][team_idx]:
                    counts[team] += 1
    return set(counts.values())
