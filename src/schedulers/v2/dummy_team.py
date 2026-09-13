import itertools
import random
import time
from collections import Counter

from schedulers.v2.utils import pair_key


def create_plan_with_dummy_team(
    teams,
    number_fields,
    number_rounds,
    teams_per_game,
    seed,
    max_seconds,
    optimizer_cls,
):
    dummy_team = object()
    deadline = time.monotonic() + max_seconds
    repair_optimizer = optimizer_cls(
        teams=teams,
        number_fields=number_fields,
        number_rounds=number_rounds,
        teams_per_game=teams_per_game,
        seed=seed,
        max_seconds=max_seconds,
    )

    general_optimizer = optimizer_cls(
        teams=teams,
        number_fields=number_fields,
        number_rounds=number_rounds,
        teams_per_game=teams_per_game,
        seed=seed,
        max_seconds=max(0.2, max_seconds * 0.45),
    )
    best_plan = general_optimizer.create_plan()
    best_score = repair_optimizer.score_plan(best_plan)

    attempt = 0
    extended_teams = list(teams) + [dummy_team]
    while time.monotonic() < deadline:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds < 0.3:
            break
        base_optimizer = optimizer_cls(
            teams=extended_teams,
            number_fields=number_fields,
            number_rounds=number_rounds,
            teams_per_game=teams_per_game,
            seed=seed + 101_003 * attempt,
            max_seconds=min(0.45, remaining_seconds * 0.35),
        )
        try:
            base_plan = base_optimizer.create_plan()
            plan = replace_dummy_team(
                base_plan,
                teams,
                dummy_team,
                number_rounds,
                teams_per_game,
                search_limit=18_000,
            )
        except RuntimeError:
            attempt += 1
            continue

        score = repair_optimizer.score_plan(plan)
        if score < best_score:
            best_plan = plan
            best_score = score
        attempt += 1

    repair_rng = random.Random(seed + 2_000_003)
    return repair_optimizer.local_improve(
        best_plan,
        best_score,
        repair_rng,
        deadline,
    )


def replace_dummy_team(
    base_plan,
    teams,
    dummy_team,
    number_rounds,
    teams_per_game,
    search_limit=120_000,
):
    plan = [[list(field) for field in round_plan] for round_plan in base_plan]
    team_order = {team: idx for idx, team in enumerate(teams)}
    pair_counts = Counter(
        {
            pair_key(team_a, team_b, team_order): 0
            for team_a, team_b in itertools.combinations(teams, 2)
        }
    )
    dummy_slots = []

    for round_idx, round_plan in enumerate(plan):
        for field_idx, field_plan in enumerate(round_plan):
            if dummy_team in field_plan:
                field_plan.remove(dummy_team)
                dummy_slots.append((round_idx, field_idx))
            for team_a, team_b in itertools.combinations(field_plan, 2):
                pair_counts[pair_key(team_a, team_b, team_order)] += 1

    if len(dummy_slots) != number_rounds:
        raise RuntimeError("Expected the dummy team exactly once per round")

    best_plan = None
    best_score = None
    searched = 0

    def replacement_score(candidate_plan, candidate_pair_counts):
        max_duel = max(candidate_pair_counts.values()) if candidate_pair_counts else 0
        pairs_at_max = sum(
            1 for count in candidate_pair_counts.values() if count == max_duel
        )
        repeated_pairs = sum(
            1 for count in candidate_pair_counts.values() if count >= 2
        )
        switch_teams = set()
        field_counts = {team: Counter() for team in teams}
        for round_plan in candidate_plan:
            switch_teams.update(round_plan[0])
            for field_idx, field_plan in enumerate(round_plan):
                for team in field_plan:
                    field_counts[team][field_idx] += 1
        missing_switch = len(set(teams) - switch_teams)
        max_field_repeat = max(
            max(counts.values()) if counts else 0
            for counts in field_counts.values()
        )
        min_distinct_fields = min(len(counts) for counts in field_counts.values())
        return (
            missing_switch,
            max_duel,
            pairs_at_max,
            repeated_pairs,
            max_field_repeat,
            -min_distinct_fields,
        )

    def recurse(slot_idx, used_replacements, candidate_pair_counts):
        nonlocal best_plan, best_score, searched
        searched += 1
        if searched > search_limit:
            return
        if slot_idx == len(dummy_slots):
            score = replacement_score(plan, candidate_pair_counts)
            if best_score is None or score < best_score:
                best_score = score
                best_plan = [[list(field) for field in round_plan] for round_plan in plan]
            return

        round_idx, field_idx = dummy_slots[slot_idx]
        field_plan = plan[round_idx][field_idx]
        options = []
        for team in teams:
            if team in used_replacements or team in field_plan:
                continue
            pair_values = [
                candidate_pair_counts[pair_key(team, other, team_order)]
                for other in field_plan
            ]
            options.append(
                (
                    max(pair_values),
                    sum(1 for value in pair_values if value >= 2),
                    sum(pair_values),
                    team,
                )
            )
        options.sort()

        for _, creates_triples, _, team in options:
            if best_score is not None and best_score[1] <= 2 and creates_triples:
                continue
            field_plan.append(team)
            used_replacements.add(team)
            changed_pairs = []
            for other in field_plan:
                if other == team:
                    continue
                key = pair_key(team, other, team_order)
                candidate_pair_counts[key] += 1
                changed_pairs.append(key)

            recurse(slot_idx + 1, used_replacements, candidate_pair_counts)

            for key in changed_pairs:
                candidate_pair_counts[key] -= 1
            used_replacements.remove(team)
            field_plan.pop()

    recurse(0, set(), pair_counts.copy())

    if best_plan is None:
        raise RuntimeError("Could not replace dummy team")

    for round_plan in best_plan:
        for field_plan in round_plan:
            if len(field_plan) != teams_per_game:
                raise RuntimeError("Dummy replacement created an incomplete field")
    return best_plan
