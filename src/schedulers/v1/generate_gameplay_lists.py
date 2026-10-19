import random

import pandas as pd
from schedulers.v1.helper import (
    check_game_plan,
    count_in_list,
    create_duells,
    create_filled_df,
    create_team_liste,
    flatten,
)


DEFAULT_SEEDS = {
    15: 17,
    16: 15,
    17: 4,
    18: 3955,
    19: 3823,
    20: 452,
    21: 18,
    22: 28,
    23: 60,
    24: 33,
    25: 62,
}


def _duell_count(df, t_1, t_2):
    row = df[
        ((df["team_1"] == t_1) & (df["team_2"] == t_2))
        | ((df["team_1"] == t_2) & (df["team_2"] == t_1))
    ]
    if row.empty:
        return None
    return row["num_games"].values[0]


def _first_unplayed_partner(teams, idx, df):
    t_1 = teams[idx]
    for t_2 in teams[idx + 1 :]:
        if _duell_count(df, t_1, t_2) == 0:
            return t_2
    return None


def _pair_teams(teams, df):
    """Pairs teams that have not played each other yet, then the rest in order."""
    pairs = []
    idx = 0
    while idx < len(teams):
        partner = _first_unplayed_partner(teams, idx, df)
        if partner is None:
            idx += 1
            continue
        t_1 = teams[idx]
        teams.remove(t_1)
        teams.remove(partner)
        pairs.append({t_1, partner})
        idx = 0
    while len(teams) >= 2:
        t_1 = teams[0]
        t_2 = teams[1]
        teams.remove(t_1)
        teams.remove(t_2)
        pairs.append({t_1, t_2})
    return pairs


def _closest_pair_idx(pairs, df):
    min_sum, min_idx = -1, -1
    for j in range(1, len(pairs) - 1):
        temp_sum = sum(
            _duell_count(df, t_1, t_2) for t_1 in pairs[0] for t_2 in pairs[j]
        )
        if temp_sum < min_sum or min_sum == -1:
            min_sum, min_idx = temp_sum, j
    return min_idx


def _get_best_combination(teams, df):
    """gets the best combinations of given teams"""
    combinations = []
    pairs = _pair_teams(teams, df)
    while len(pairs) > 1:
        p_1 = pairs[0]
        p_2 = pairs[_closest_pair_idx(pairs, df)]
        combinations.append(list(p_1.union(p_2)))
        pairs = [p for p in pairs if p not in [p_1, p_2]]
    return combinations


def _sync_df(r_plan, df):
    for teams_on_field in r_plan:
        teams_on_field.sort()
        for t_1_idx in range(len(teams_on_field)):
            for t_2 in teams_on_field[t_1_idx:]:
                bedingung = (df["team_1"] == teams_on_field[t_1_idx]) & (
                    df["team_2"] == t_2
                )
                df.loc[bedingung, "num_games"] += 1
    return df


def _switch_need(f_plan, t_f_s):
    return sum(t_f_s[t] if t_f_s[t] else -100 for t in f_plan)


def _move_to_switch(r_plan, zero_team, t_f_s):
    """Swaps ``zero_team`` onto switch 1 with a team that played there more than once."""
    switch_field = r_plan[0]
    swap_idx = next(
        (idx for idx, t in enumerate(switch_field) if t_f_s[t] > 1),
        None,
    )
    if swap_idx is None:
        return
    t_f_s[switch_field[swap_idx]] -= 1
    for f_plan in r_plan[1 : len(r_plan) - 1]:
        for t_idx, t in enumerate(f_plan):
            if t == zero_team:
                f_plan[t_idx] = switch_field[swap_idx]
                switch_field[swap_idx] = zero_team
                t_f_s[zero_team] += 1


def _sort_game_plan(g_plan, teams):
    new_g_plan = []
    t_f_s = dict.fromkeys(teams, 0)  # key: team, value: times on switch 1
    for r_plan in g_plan:
        new_r_plan = tuple(sorted(r_plan, key=lambda f: _switch_need(f, t_f_s)))
        for t in new_r_plan[0]:
            t_f_s[t] += 1
        new_g_plan.append(new_r_plan)
    zero_teams = [key for key, value in t_f_s.items() if value == 0]
    if not zero_teams:
        return new_g_plan, True
    if len(zero_teams) >= 2 or not count_in_list(new_g_plan[-1], zero_teams[0]):
        return new_g_plan, False
    _move_to_switch(new_g_plan[-1], zero_teams[0], t_f_s)
    return new_g_plan, True


def _rotate_teams(t_l, idx_last, slots):
    if idx_last < slots:
        extend_beginning = t_l[slots:] + t_l[0 : idx_last + 1]
        random.shuffle(extend_beginning)
        return extend_beginning + t_l[idx_last + 1 : slots], idx_last + len(t_l[slots:])
    new_end = t_l[0:slots]
    random.shuffle(new_end)
    return t_l[slots:] + new_end, idx_last - slots


def _build_game_plan(team_list, number_fields, number_rounds, max_duell_repeats):
    game_plan = []
    t_l = team_list[:]
    df_duells = create_duells(t_l)
    idx_last = (
        len(t_l) - 1
    )  # all teams after this index have more games played then the ones before
    slots = number_fields * 4

    for _ in range(number_rounds):
        round_plan = _get_best_combination(t_l[0:slots], df_duells)
        df_duells = _sync_df(round_plan, df_duells)
        if df_duells["num_games"].max() > max_duell_repeats:
            return None
        game_plan.append(round_plan)
        t_l, idx_last = _rotate_teams(t_l, idx_last, slots)
    return game_plan, df_duells


def _better_plan(curr_best, candidate, df_duells, team_list, max_duell_repeats, seed):
    game_plan, valid = candidate
    df_duells_new = create_filled_df(team_list, game_plan)
    max_new = df_duells_new["num_games"].max()
    if not (
        valid
        and df_duells["num_games"].max() >= max_new
        and max_new <= max_duell_repeats
    ):
        return curr_best
    repeats = len(df_duells_new[df_duells["num_games"] == max_new])
    if curr_best[1] > max_new or (curr_best[1] == max_new and curr_best[2] > repeats):
        return game_plan, max_new, repeats, seed
    return curr_best


def create_plan(
    team_list,
    number_fields=4,
    number_rounds=8,
    max_duell_repeats=100,
    min_searches=0,
    max_searches=20,
    seed_counter=0,
):
    if seed_counter == 0:
        seed_counter = DEFAULT_SEEDS.get(len(team_list), 0)

    i = 0
    curr_best = (None, 1234, 1234, 0)
    while True:
        random.seed(seed_counter + i)
        i += 1
        built = _build_game_plan(
            team_list, number_fields, number_rounds, max_duell_repeats
        )
        if built is not None:
            game_plan, df_duells = built
            candidate = _sort_game_plan(game_plan[:], team_list)
            curr_best = _better_plan(
                curr_best,
                candidate,
                df_duells,
                team_list,
                max_duell_repeats,
                seed_counter + i - 1,
            )
            if candidate[1] and i >= min_searches and curr_best[1] <= max_duell_repeats:
                # print("SEED:", curr_best[3])
                return curr_best[0]
        if i >= max_searches:
            return curr_best[0]


def get_unrated_games(game_plan):
    teams = list(set(flatten(game_plan)))
    df = pd.DataFrame({"team": teams}).assign(num_games=0)
    min_games = -1
    for t in teams:
        c = count_in_list(game_plan, t)
        if min_games > c or min_games == -1:
            min_games = c
    rate_game_plan = []
    for r in game_plan:
        rate_game_plan.append([])
        for f in r:
            rate_game_plan[-1].append([])
            for t in f:
                if df[df["team"] == t]["num_games"].values.tolist()[0] >= min_games:
                    rate_game_plan[-1][-1].append(False)
                else:
                    df.loc[(df["team"] == t), "num_games"] += 1
                    rate_game_plan[-1][-1].append(True)
    return rate_game_plan


def generate_plan(num_teams=25, num_fields=4, num_rounds=8, _num_teams_per_game=4):
    team_list = create_team_liste(num_teams)
    plan = create_plan(team_list, num_fields, num_rounds)
    # print(plan)
    rate_plan = get_unrated_games(plan)
    # print(rate_plan)
    max_games_count = check_game_plan(plan, rate_plan)
    return plan, max_games_count


if __name__ == '__main__':
    generated_plan = create_plan(create_team_liste(20), 4, 8)
    print(generated_plan)
    generated_rate_plan = get_unrated_games(generated_plan)
    print(generated_rate_plan)
    check_game_plan(generated_plan, generated_rate_plan)
