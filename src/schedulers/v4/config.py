DEFAULT_MAX_ITERATIONS = 150_000
DEFAULT_MAX_SECONDS = 5.0


def resolve_seed(seed, num_teams, number_rounds, field_sizes):
    if seed is not None:
        return seed
    sizes_value = 0
    for size in field_sizes:
        sizes_value = sizes_value * 31 + size
    return (
        num_teams * 10_007
        + len(field_sizes) * 1_009
        + number_rounds * 101
        + sizes_value
    )
