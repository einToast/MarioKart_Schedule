import itertools
import random
import time
from collections import Counter

import numpy as np

from schedulers.v2.utils import create_play_targets, flatten, pair_key


class ScheduleOptimizer:
    def __init__(
        self,
        teams,
        number_fields,
        number_rounds,
        teams_per_game,
        seed,
        max_seconds,
    ):
        self.teams = teams
        self.number_fields = number_fields
        self.number_rounds = number_rounds
        self.teams_per_game = teams_per_game
        self.seed = seed
        self.max_seconds = max_seconds
        self.capacity_per_round = number_fields * teams_per_game
        self.total_slots = self.capacity_per_round * number_rounds
        self.team_order = {team: idx for idx, team in enumerate(teams)}
        self.all_pairs = [
            pair_key(team_a, team_b, self.team_order)
            for team_a, team_b in itertools.combinations(teams, 2)
        ]

    def create_plan(self):
        deadline = time.monotonic() + self.max_seconds
        construction_deadline = time.monotonic() + self.max_seconds * 0.65
        best_plan = None
        best_score = None
        attempts = self._attempt_budget()

        for attempt in range(attempts):
            if attempt > 0 and time.monotonic() >= construction_deadline:
                break

            rng = random.Random(self.seed + attempt * 7919)
            np_rng = np.random.default_rng(self.seed + attempt * 7919)
            plan = self._build_candidate(rng, np_rng)
            if plan is None:
                continue

            score = self.score_plan(plan)
            if best_score is None or score < best_score:
                best_plan = plan
                best_score = score
                if self._is_excellent(score):
                    break

        if best_plan is None:
            raise RuntimeError("Could not generate a valid schedule")

        repair_rng = random.Random(self.seed + 1_000_003)
        return self.local_improve(best_plan, best_score, repair_rng, deadline)

    def _attempt_budget(self):
        if len(self.teams) <= 18:
            return 450
        if len(self.teams) <= 22:
            return 350
        return 250

    def _is_excellent(self, score):
        same_round_duplicates, missing_switch, max_duel, pairs_at_max = score[:4]
        unavoidable_duplicates = max(0, self.capacity_per_round - len(self.teams))
        unavoidable_duplicates *= self.number_rounds
        return (
            same_round_duplicates == unavoidable_duplicates
            and missing_switch == 0
            and max_duel <= 2
            and pairs_at_max <= self._expected_pairs_at_two()
        )

    def _expected_pairs_at_two(self):
        pair_slots = self.number_rounds * self.number_fields
        pair_slots *= self.teams_per_game * (self.teams_per_game - 1) // 2
        if not self.all_pairs:
            return 0
        return max(0, pair_slots - len(self.all_pairs))

    def _build_candidate(self, rng, np_rng):
        targets = create_play_targets(self.teams, self.total_slots, np_rng)
        rounds = self._build_round_rosters(targets, rng)
        if rounds is None:
            return None

        pair_counts = Counter()
        field_counts = {team: Counter() for team in self.teams}
        switch_counts = Counter()
        plan = []

        for round_entries in rounds:
            fields = self._build_round_fields(
                round_entries,
                pair_counts,
                field_counts,
                switch_counts,
                rng,
            )
            if fields is None:
                return None
            plan.append(fields)
            self._apply_round(fields, pair_counts, field_counts, switch_counts)

        return plan

    def _build_round_rosters(self, targets, rng):
        remaining = dict(targets)
        rounds = []

        for round_idx in range(self.number_rounds):
            rounds_left_after = self.number_rounds - round_idx - 1
            entries = []

            if len(self.teams) < self.capacity_per_round:
                for team in self.teams:
                    if remaining[team] > 0:
                        entries.append(team)
                        remaining[team] -= 1

                while len(entries) < self.capacity_per_round:
                    choices = [
                        team
                        for team in self.teams
                        if remaining[team] > 0 and entries.count(team) < self.teams_per_game
                    ]
                    if not choices:
                        return None
                    choices.sort(
                        key=lambda team: (
                            -remaining[team],
                            rng.random(),
                        )
                    )
                    team = choices[0]
                    entries.append(team)
                    remaining[team] -= 1
            else:
                mandatory = [
                    team
                    for team in self.teams
                    if remaining[team] > rounds_left_after
                ]
                if len(mandatory) > self.capacity_per_round:
                    return None

                entries.extend(mandatory)
                for team in mandatory:
                    remaining[team] -= 1

                fill_candidates = [
                    team
                    for team in self.teams
                    if team not in mandatory and remaining[team] > 0
                ]
                fill_candidates.sort(key=lambda team: (-remaining[team], rng.random()))

                for team in fill_candidates:
                    if len(entries) >= self.capacity_per_round:
                        break
                    entries.append(team)
                    remaining[team] -= 1

                if len(entries) != self.capacity_per_round:
                    return None

            rng.shuffle(entries)
            rounds.append(entries)

        if any(remaining.values()):
            return None
        return rounds

    def _build_round_fields(
        self,
        round_entries,
        pair_counts,
        field_counts,
        switch_counts,
        rng,
    ):
        best_fields = None
        best_score = None
        attempts = 90

        for attempt in range(attempts):
            fields = [[] for _ in range(self.number_fields)]
            entries = list(round_entries)
            rng.shuffle(entries)

            if attempt % 3 != 1:
                self._prefill_switch_field(entries, fields[0], switch_counts, rng)

            ordered_entries = sorted(
                entries,
                key=lambda team: (
                    switch_counts[team] > 0,
                    field_counts[team][0],
                    rng.random(),
                ),
            )

            if not self._assign_entries_to_fields(
                ordered_entries,
                fields,
                pair_counts,
                field_counts,
                switch_counts,
                rng,
            ):
                continue

            score = self._score_round_delta(
                fields,
                pair_counts,
                field_counts,
                switch_counts,
            )
            if best_score is None or score < best_score:
                best_fields = fields
                best_score = score

        return best_fields

    def _prefill_switch_field(self, entries, switch_field, switch_counts, rng):
        candidates = sorted(
            set(entries),
            key=lambda team: (
                switch_counts[team] > 0,
                switch_counts[team],
                rng.random(),
            ),
        )
        for team in candidates:
            if len(switch_field) >= self.teams_per_game:
                break
            if team not in entries:
                continue
            switch_field.append(team)
            entries.remove(team)

    def _assign_entries_to_fields(
        self,
        entries,
        fields,
        pair_counts,
        field_counts,
        switch_counts,
        rng,
    ):
        for team in entries:
            options = []
            for field_idx, field in enumerate(fields):
                if len(field) >= self.teams_per_game or team in field:
                    continue
                options.append(
                    (
                        self._entry_field_cost(
                            team,
                            field_idx,
                            field,
                            pair_counts,
                            field_counts,
                            switch_counts,
                        )
                        + rng.random() * 0.01,
                        field_idx,
                    )
                )

            if not options:
                return False

            options.sort()
            chosen_field_idx = options[0][1]
            fields[chosen_field_idx].append(team)

        return all(len(field) == self.teams_per_game for field in fields)

    def _entry_field_cost(
        self,
        team,
        field_idx,
        field,
        pair_counts,
        field_counts,
        switch_counts,
    ):
        cost = 0.0
        for other in field:
            played = pair_counts[pair_key(team, other, self.team_order)]
            if played >= 2:
                cost += 1_000_000
            elif played == 1:
                cost += 1_000
            else:
                cost += 1

        field_repeats = field_counts[team][field_idx]
        cost += field_repeats * field_repeats * 30
        if field_idx == 0 and switch_counts[team] == 0:
            cost -= 10_000
        if field_idx == 0:
            cost += switch_counts[team] * 150
        return cost

    def _score_round_delta(
        self,
        fields,
        pair_counts,
        field_counts,
        switch_counts,
    ):
        third_pair_creations = 0
        second_pair_creations = 0
        pair_pressure = 0
        new_switch_teams = 0
        repeated_field_hits = 0
        max_field_after = 0

        for field_idx, field in enumerate(fields):
            for team in field:
                if field_idx == 0 and switch_counts[team] == 0:
                    new_switch_teams += 1
                repeated_field_hits += field_counts[team][field_idx]
                max_field_after = max(max_field_after, field_counts[team][field_idx] + 1)

            for team_a, team_b in itertools.combinations(field, 2):
                played = pair_counts[pair_key(team_a, team_b, self.team_order)]
                pair_pressure += played
                if played >= 2:
                    third_pair_creations += 1
                elif played == 1:
                    second_pair_creations += 1

        return (
            third_pair_creations,
            second_pair_creations,
            pair_pressure,
            -new_switch_teams,
            max_field_after,
            repeated_field_hits,
        )

    def _apply_round(self, fields, pair_counts, field_counts, switch_counts):
        for field_idx, field in enumerate(fields):
            for team in field:
                field_counts[team][field_idx] += 1
                if field_idx == 0:
                    switch_counts[team] += 1
            for team_a, team_b in itertools.combinations(field, 2):
                pair_counts[pair_key(team_a, team_b, self.team_order)] += 1

    def score_plan(self, plan):
        play_counts = Counter(flatten(plan))
        pair_counts = self._pair_counts(plan)
        field_counts = self._field_counts(plan)
        switch_teams = set()
        same_round_duplicates = 0

        for round_plan in plan:
            round_teams = flatten(round_plan)
            same_round_duplicates += len(round_teams) - len(set(round_teams))
            switch_teams.update(round_plan[0])

        max_duel = max(pair_counts.values()) if pair_counts else 0
        pairs_at_max = sum(1 for count in pair_counts.values() if count == max_duel)
        repeat_pairs = sum(1 for count in pair_counts.values() if count >= 2)
        repeat_excess = sum(max(0, count - 1) for count in pair_counts.values())
        missing_switch = len(set(self.teams) - switch_teams)
        play_spread = max(play_counts.values()) - min(play_counts.values())
        max_field_repeat = max(
            max(counts.values()) if counts else 0
            for counts in field_counts.values()
        )
        min_distinct_fields = min(len(counts) for counts in field_counts.values())
        field_repeat_excess = sum(
            sum(max(0, count - 1) for count in counts.values())
            for counts in field_counts.values()
        )
        switch_spread = self._switch_spread(plan)

        return (
            same_round_duplicates,
            missing_switch,
            max_duel,
            pairs_at_max,
            repeat_pairs,
            repeat_excess,
            play_spread,
            max_field_repeat,
            -min_distinct_fields,
            field_repeat_excess,
            switch_spread,
        )

    def local_improve(self, plan, score, rng, deadline):
        current_score = score
        best_score = score
        best_plan = self._copy_plan(plan)
        positions = [
            (round_idx, field_idx, team_idx)
            for round_idx in range(self.number_rounds)
            for field_idx in range(self.number_fields)
            for team_idx in range(self.teams_per_game)
        ]
        allowed_round_duplicates = max(0, self.capacity_per_round - len(self.teams))
        steps = 0
        max_steps = 80_000 if len(self.teams) <= 18 else 45_000

        while steps < max_steps and time.monotonic() < deadline:
            steps += 1
            pos_a, pos_b = rng.sample(positions, 2)
            if self._slot(plan, pos_a) == self._slot(plan, pos_b):
                continue
            if not self._swap_is_valid(plan, pos_a, pos_b, allowed_round_duplicates):
                continue

            self._swap_slots(plan, pos_a, pos_b)
            new_score = self.score_plan(plan)
            if new_score <= current_score or rng.random() < self._anneal_probability(
                current_score,
                new_score,
                steps,
                max_steps,
            ):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_plan = self._copy_plan(plan)
                    if self._is_excellent(best_score):
                        break
            else:
                self._swap_slots(plan, pos_a, pos_b)

        return best_plan

    def _anneal_probability(self, old_score, new_score, steps, max_steps):
        if new_score[:3] > old_score[:3]:
            return 0.0
        temperature = max(0.01, 1.0 - steps / max_steps)
        old_value = self._weighted_score(old_score)
        new_value = self._weighted_score(new_score)
        if new_value <= old_value:
            return 0.15 * temperature
        return min(0.05, temperature / (new_value - old_value + 1))

    def _weighted_score(self, score):
        weights = (
            10_000_000,
            1_000_000,
            100_000,
            1_000,
            500,
            200,
            100,
            30,
            30,
            10,
            1,
        )
        return sum(value * weight for value, weight in zip(score, weights))

    def _slot(self, plan, position):
        round_idx, field_idx, team_idx = position
        return plan[round_idx][field_idx][team_idx]

    def _swap_slots(self, plan, pos_a, pos_b):
        round_a, field_a, team_a = pos_a
        round_b, field_b, team_b = pos_b
        plan[round_a][field_a][team_a], plan[round_b][field_b][team_b] = (
            plan[round_b][field_b][team_b],
            plan[round_a][field_a][team_a],
        )

    def _swap_is_valid(self, plan, pos_a, pos_b, allowed_round_duplicates):
        self._swap_slots(plan, pos_a, pos_b)
        affected_rounds = {pos_a[0], pos_b[0]}
        affected_fields = {(pos_a[0], pos_a[1]), (pos_b[0], pos_b[1])}
        valid = True

        for round_idx, field_idx in affected_fields:
            field = plan[round_idx][field_idx]
            if len(field) != len(set(field)):
                valid = False
                break

        if valid:
            for round_idx in affected_rounds:
                round_teams = flatten(plan[round_idx])
                round_duplicates = len(round_teams) - len(set(round_teams))
                if round_duplicates > allowed_round_duplicates:
                    valid = False
                    break

        self._swap_slots(plan, pos_a, pos_b)
        return valid

    def _copy_plan(self, plan):
        return [[list(field) for field in round_plan] for round_plan in plan]

    def _pair_counts(self, plan):
        counts = Counter({pair: 0 for pair in self.all_pairs})
        for round_plan in plan:
            for field_plan in round_plan:
                for team_a, team_b in itertools.combinations(field_plan, 2):
                    counts[pair_key(team_a, team_b, self.team_order)] += 1
        return counts

    def _field_counts(self, plan):
        counts = {team: Counter() for team in self.teams}
        for round_plan in plan:
            for field_idx, field_plan in enumerate(round_plan):
                for team in field_plan:
                    counts[team][field_idx] += 1
        return counts

    def _switch_spread(self, plan):
        counts = Counter()
        for round_plan in plan:
            for team in round_plan[0]:
                counts[team] += 1
        values = [counts[team] for team in self.teams]
        return max(values) - min(values)
