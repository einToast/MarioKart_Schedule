import itertools
import math
import random
import time
from collections import Counter

import numpy as np
from schedulers.v2.plan_state import PlanState
from schedulers.v2.utils import create_play_targets, flatten, pair_key

CONSTRUCTION_TIME_SHARE = 0.3
ROUND_FIELD_ATTEMPTS = 30
TARGETED_MOVE_SHARE = 0.5


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
        self._ideal_score = self._compute_ideal_score()

    def create_plan(self):
        deadline = time.monotonic() + self.max_seconds
        construction_deadline = (
            time.monotonic() + self.max_seconds * CONSTRUCTION_TIME_SHARE
        )
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
        return self.local_improve(best_plan, repair_rng, deadline)

    def _attempt_budget(self):
        if len(self.teams) <= 18:
            return 450
        if len(self.teams) <= 22:
            return 350
        return 250

    def _is_excellent(self, score):
        return all(value <= bound for value, bound in zip(score, self._ideal_score))

    def _compute_ideal_score(self):
        """Per-criterion lower bounds of ``score_plan``; ``inf`` means unconstrained."""
        team_count = len(self.teams)
        fields = self.number_fields
        pair_slots = self.number_rounds * fields
        pair_slots *= self.teams_per_game * (self.teams_per_game - 1) // 2
        base_games, extra_games = divmod(self.total_slots, team_count)
        max_games = base_games + (1 if extra_games else 0)
        switch_slots = self.number_rounds * self.teams_per_game

        duplicates = max(0, self.capacity_per_round - team_count) * self.number_rounds
        field_repeat_excess = (team_count - extra_games) * max(0, base_games - fields)
        field_repeat_excess += extra_games * max(0, base_games + 1 - fields)

        # Each team only has team_count - 1 distinct opponents, so a team playing
        # games * (teams_per_game - 1) opponent slots must repeat the surplus.
        opponents_per_game = self.teams_per_game - 1
        team_surplus = (team_count - extra_games) * max(
            0, base_games * opponents_per_game - (team_count - 1)
        )
        team_surplus += extra_games * max(
            0, (base_games + 1) * opponents_per_game - (team_count - 1)
        )
        repeat_excess = max(pair_slots - len(self.all_pairs), -(-team_surplus // 2))
        max_duel = -(-pair_slots // len(self.all_pairs)) if self.all_pairs else 0
        return (
            duplicates,
            0,
            max(1, max_duel),
            math.inf,
            math.inf,
            max(0, repeat_excess),
            1 if extra_games else 0,
            -(-max_games // fields),
            -min(base_games, fields),
            field_repeat_excess,
            1 if switch_slots % team_count else 0,
        )

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

            if len(self.teams) < self.capacity_per_round:
                entries = self._build_small_round_entries(remaining, rng)
            else:
                entries = self._build_large_round_entries(
                    remaining, rounds_left_after, rng
                )
            if entries is None:
                return None

            rng.shuffle(entries)
            rounds.append(entries)

        if any(remaining.values()):
            return None
        return rounds

    def _build_small_round_entries(self, remaining, rng):
        entries = []
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

        return entries

    def _build_large_round_entries(self, remaining, rounds_left_after, rng):
        mandatory = [team for team in self.teams if remaining[team] > rounds_left_after]
        if len(mandatory) > self.capacity_per_round:
            return None

        entries = list(mandatory)
        for team in mandatory:
            remaining[team] -= 1

        fill_candidates = [
            team for team in self.teams if team not in mandatory and remaining[team] > 0
        ]
        fill_candidates.sort(key=lambda team: (-remaining[team], rng.random()))

        for team in fill_candidates:
            if len(entries) >= self.capacity_per_round:
                break
            entries.append(team)
            remaining[team] -= 1

        if len(entries) != self.capacity_per_round:
            return None

        return entries

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

        for attempt in range(ROUND_FIELD_ATTEMPTS):
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
                max_field_after = max(
                    max_field_after, field_counts[team][field_idx] + 1
                )

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
            max(counts.values()) if counts else 0 for counts in field_counts.values()
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

    def local_improve(self, plan, rng, deadline):
        state = PlanState(self.team_order, plan)
        current_score = best_score = state.score()
        best_plan = state.snapshot()
        positions = [
            (round_idx, field_idx, team_idx)
            for round_idx in range(self.number_rounds)
            for field_idx in range(self.number_fields)
            for team_idx in range(self.teams_per_game)
        ]
        position_count = len(positions)
        allowed_round_duplicates = max(0, self.capacity_per_round - len(self.teams))
        started_at = time.monotonic()
        duration = max(deadline - started_at, 1e-9)
        excellent = self._is_excellent(best_score)

        while not excellent:
            now = time.monotonic()
            if now >= deadline:
                break

            pos_a = None
            if rng.random() < TARGETED_MOVE_SHARE:
                pos_a = state.repeated_pair_position(rng)
            if pos_a is None:
                pos_a = positions[int(rng.random() * position_count)]
            pos_b = positions[int(rng.random() * position_count)]
            if not state.swap_is_valid(pos_a, pos_b, allowed_round_duplicates):
                continue

            state.swap(pos_a, pos_b)
            new_score = state.score()
            progress = (now - started_at) / duration
            if new_score <= current_score or rng.random() < self._anneal_probability(
                current_score, new_score, progress
            ):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_plan = state.snapshot()
                    excellent = self._is_excellent(best_score)
            else:
                state.swap(pos_a, pos_b)

        return [
            [[self.teams[idx] for idx in field] for field in round_plan]
            for round_plan in best_plan
        ]

    def _anneal_probability(self, old_score, new_score, progress):
        if new_score[:3] > old_score[:3]:
            return 0.0
        temperature = max(0.01, 1.0 - progress)
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

    def _pair_counts(self, plan):
        counts = Counter(dict.fromkeys(self.all_pairs, 0))
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
