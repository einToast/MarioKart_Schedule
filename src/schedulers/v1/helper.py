import itertools
from collections import Counter

import pandas as pd
from schedulers.v1.team import Team


def create_team_liste(length=10):
    return list(range(length))


def count_in_list(temp_list, x):
    counter = 0
    for element in temp_list:
        if isinstance(element, list) or isinstance(element, tuple):
            counter += count_in_list(element, x)
        else:
            if element == x:
                counter += 1
    return counter


def flatten(liste):
    ergebnis = []
    for element in liste:
        if isinstance(element, list) or isinstance(element, tuple):
            ergebnis.extend(flatten(element))
        elif isinstance(element, (int, float, Team)):
            ergebnis.append(element)
    return ergebnis


def create_duells(t_l):
    combs = itertools.combinations(t_l, 2)
    df = pd.DataFrame(combs, columns=["team_1", "team_2"]).assign(num_games=0)
    return df


def create_filled_df(_t_l, plan_games):
    combs = itertools.combinations(_t_l, 2)
    df = pd.DataFrame(combs, columns=["team_1", "team_2"]).assign(num_games=0)
    for _round in plan_games:
        for _field in _round:
            for _t_1_idx in range(len(_field)):
                for _t_2 in _field[_t_1_idx:]:
                    _bed = (df["team_1"] == _field[_t_1_idx]) & (df["team_2"] == _t_2)

                    df.loc[_bed, "num_games"] += 1
    return df


def check_game_plan(plan, eval_plan=None):
    if not eval_plan:
        return {0}
    rating_games = Counter(
        team
        for round_plan, round_eval in zip(plan, eval_plan)
        for field, field_eval in zip(round_plan, round_eval)
        for team, rated in zip(field, field_eval)
        if rated
    )
    return set(rating_games.values())
