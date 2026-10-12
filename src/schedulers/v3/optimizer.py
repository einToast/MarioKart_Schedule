import itertools
import math
import random
import time
from collections import Counter

import numpy as np
from schedulers.v3.config import DEFAULT_MAX_ITERATIONS
from schedulers.v3.plan_state import PlanState
from schedulers.v3.utils import create_play_targets, flatten, pair_key

RESTARTS = 2
CONSTRUCTION_CANDIDATES = 3
CONSTRUCTION_ATTEMPTS = 200
ROUND_FIELD_ATTEMPTS = 30
ANNEAL_SHARE = 0.85
START_TEMPERATURE = 6.0
END_TEMPERATURE = 0.15
FIELD_SWAP_SHARE = 0.1
DOUBLE_SWAP_SHARE = 0.3
TARGETED_MOVE_SHARE = 0.5
DEADLINE_CHECK_INTERVAL = 256


class ScheduleOptimizer:
    def __init__(
        self,
        teams,
        number_fields,
        number_rounds,
        teams_per_game,
        seed,
        max_seconds,
        max_iterations=DEFAULT_MAX_ITERATIONS,
    ):
        self.teams = teams
        self.number_fields = number_fields
        self.number_rounds = number_rounds
        self.teams_per_game = teams_per_game
        self.seed = seed
        self.max_seconds = max_seconds
        self.max_iterations = max_iterations
        self.capacity_per_round = number_fields * teams_per_game
        self.total_slots = self.capacity_per_round * number_rounds
        self.team_order = {team: idx for idx, team in enumerate(teams)}
        self.all_pairs = [
            pair_key(team_a, team_b, self.team_order)
            for team_a, team_b in itertools.combinations(teams, 2)
        ]
        self._ideal_score = self._compute_ideal_score()
        self._free_duels = max(1, self._pair_slots() // max(1, len(self.all_pairs)))

    def create_plan(self):
        """Run ``RESTARTS`` independent searches and return the best plan.

        Raises ``RuntimeError`` if no valid starting plan could be built.
        """
        started_at = time.monotonic()
        best_plan = None
        best_score = None
        attempt = 0

        for restart in range(RESTARTS):
            deadline = started_at + self.max_seconds * (restart + 1) / RESTARTS
            plan, attempt = self._construct(attempt, deadline)
            if plan is None:
                continue

            rng = random.Random(self.seed + 1_000_003 * (restart + 1))
            plan = self.local_improve(
                plan, rng, deadline, self.max_iterations // RESTARTS
            )
            score = self.score_plan(plan)
            if best_score is None or score < best_score:
                best_plan = plan
                best_score = score
                if self._is_excellent(score):
                    break

        if best_plan is None:
            raise RuntimeError("Could not generate a valid schedule")
        return best_plan

    def _construct(self, attempt, deadline):
        """Build up to ``CONSTRUCTION_CANDIDATES`` greedy plans and keep the best.

        ``attempt`` is the first attempt number to use; each attempt has its own
        random seed. Returns ``(plan, next_attempt)`` so that the next restart
        continues with unused seeds. ``plan`` is ``None`` if every attempt failed.
        """
        best_plan = None
        best_score = None
        candidates = 0
        last_attempt = attempt + CONSTRUCTION_ATTEMPTS

        while attempt < last_attempt and candidates < CONSTRUCTION_CANDIDATES:
            if best_plan is not None and time.monotonic() >= deadline:
                break

            rng = random.Random(self.seed + attempt * 7919)
            np_rng = np.random.default_rng(self.seed + attempt * 7919)
            attempt += 1
            plan = self._build_candidate(rng, np_rng)
            if plan is None:
                continue

            candidates += 1
            score = self.score_plan(plan)
            if best_score is None or score < best_score:
                best_plan = plan
                best_score = score

        return best_plan, attempt

    def _is_excellent(self, score):
        return all(value <= bound for value, bound in zip(score, self._ideal_score))

    def _compute_ideal_score(self):
        """Per-criterion lower bounds of ``score_plan``; ``inf`` means unconstrained."""
        team_count = len(self.teams)
        fields = self.number_fields
        pair_slots = self._pair_slots()
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
        repeat_excess = max(0, repeat_excess)
        max_duel = 2 if repeat_excess else 1
        if self.all_pairs:
            max_duel = max(
                max_duel,
                -(-pair_slots // len(self.all_pairs)),
                -(-max_games * opponents_per_game // (team_count - 1)),
            )
        return (
            duplicates,
            max(0, team_count - switch_slots),
            0,
            max_duel,
            math.inf,
            math.inf,
            repeat_excess,
            1 if extra_games else 0,
            -(-max_games // fields),
            -min(base_games, fields),
            field_repeat_excess,
            1 if switch_slots % team_count else 0,
        )

    def _pair_slots(self):
        duels_per_game = self.teams_per_game * (self.teams_per_game - 1) // 2
        return self.number_rounds * self.number_fields * duels_per_game

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
            self._field_imbalance(play_counts, field_counts),
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

    def local_improve(self, plan, rng, deadline, iterations=None):
        """Improve ``plan`` and return the best plan found.

        The first ``ANNEAL_SHARE`` of the iterations anneal on
        ``PlanState.energy``; the rest hill-climb on the score tuple, starting
        from the best plan so far. The search stops after ``iterations`` moves,
        at ``deadline``, or as soon as the score reaches its lower bounds.
        """
        if iterations is None:
            iterations = self.max_iterations
        state = PlanState(self.team_order, plan, self._free_duels)
        best_score = state.score()
        best_plan = state.snapshot()
        if self._is_excellent(best_score):
            return self._to_teams(best_plan)

        allowed_round_duplicates = max(0, self.capacity_per_round - len(self.teams))
        started_at = time.monotonic()
        duration = max(deadline - started_at, 1e-9)
        cooling = math.log(END_TEMPERATURE / START_TEMPERATURE)
        temperature = START_TEMPERATURE
        annealing = True
        best_energy = state.energy
        current_score = best_score
        random_value = rng.random

        for iteration in range(iterations):
            if iteration % DEADLINE_CHECK_INTERVAL == 0:
                elapsed = (time.monotonic() - started_at) / duration
                if elapsed >= 1.0:
                    break
                progress = max(iteration / iterations, elapsed) / ANNEAL_SHARE
                if progress < 1.0:
                    temperature = START_TEMPERATURE * math.exp(cooling * progress)
                elif annealing:
                    annealing = False
                    state = PlanState(
                        self.team_order, self._to_teams(best_plan), self._free_duels
                    )
                    current_score = best_score

            move = self._pick_move(state, rng, allowed_round_duplicates)
            if move is None:
                continue

            old_energy = state.energy
            self._apply_move(state, move)
            if annealing:
                delta = state.energy - old_energy
                if delta > 0 and random_value() >= math.exp(-delta / temperature):
                    self._apply_move(state, move)
                    continue
                if state.energy > best_energy:
                    continue
                best_energy = state.energy
                new_score = state.score()
            else:
                new_score = state.score()
                if new_score > current_score:
                    self._apply_move(state, move)
                    continue
                current_score = new_score

            if new_score < best_score:
                best_score = new_score
                best_plan = state.snapshot()
                if self._is_excellent(best_score):
                    break

        return self._to_teams(best_plan)

    def _pick_move(self, state, rng, allowed_round_duplicates):
        """Pick a random move for ``_apply_move``.

        Returns ``(round, field_a, field_b)`` for a swap of two whole fields,
        ``(pos_a, pos_b)`` for a swap of two team slots,
        ``(pos_a, pos_b, pos_c, pos_d)`` for a double swap, or ``None`` if the
        chosen move is not possible.
        """
        pos_a = None
        if self.number_fields > 1:
            choice = rng.random()
            if choice < FIELD_SWAP_SHARE:
                round_idx = int(rng.random() * self.number_rounds)
                field_a = int(rng.random() * self.number_fields)
                field_b = int(rng.random() * (self.number_fields - 1))
                if field_b >= field_a:
                    field_b += 1
                return round_idx, field_a, field_b
            double_swap = choice < FIELD_SWAP_SHARE + DOUBLE_SWAP_SHARE
        else:
            double_swap = False

        if rng.random() < TARGETED_MOVE_SHARE:
            pos_a = state.repeated_pair_position(rng)
        if pos_a is None:
            pos_a = self._random_position(rng)
        if double_swap:
            return self._pick_double_swap(state, rng, pos_a)
        pos_b = self._random_position(rng)
        if not state.swap_is_valid(pos_a, pos_b, allowed_round_duplicates):
            return None
        return pos_a, pos_b

    def _pick_double_swap(self, state, rng, pos_a):
        """Two slot swaps that leave every team's field counts unchanged.

        The team at ``pos_a`` trades places with a team on another field of the
        same round, and the two trade back in a second round where they sit on
        each other's fields. Only duels change. Returns ``None`` if there is
        no such second round.
        """
        round_idx, field_a, _ = pos_a
        field_b = int(rng.random() * (self.number_fields - 1))
        if field_b >= field_a:
            field_b += 1
        pos_b = (round_idx, field_b, int(rng.random() * self.teams_per_game))
        if not state.swap_is_valid(pos_a, pos_b, 0):
            return None

        team_a = state.plan[round_idx][field_a][pos_a[2]]
        team_b = state.plan[round_idx][field_b][pos_b[2]]
        first_round = int(rng.random() * self.number_rounds)
        for offset in range(self.number_rounds):
            other_round = (first_round + offset) % self.number_rounds
            if other_round == round_idx:
                continue
            other_plan = state.plan[other_round]
            if team_a in other_plan[field_b] and team_b in other_plan[field_a]:
                pos_c = (other_round, field_b, other_plan[field_b].index(team_a))
                pos_d = (other_round, field_a, other_plan[field_a].index(team_b))
                if state.swap_is_valid(pos_c, pos_d, 0):
                    return pos_a, pos_b, pos_c, pos_d
        return None

    def _random_position(self, rng):
        return (
            int(rng.random() * self.number_rounds),
            int(rng.random() * self.number_fields),
            int(rng.random() * self.teams_per_game),
        )

    @staticmethod
    def _apply_move(state, move):
        """Apply a move from ``_pick_move``; applying it again undoes it."""
        if len(move) == 3:
            state.swap_fields(*move)
        elif len(move) == 4:
            state.swap(move[0], move[1])
            state.swap(move[2], move[3])
        else:
            state.swap(*move)

    def _to_teams(self, index_plan):
        """Convert a plan of team indices back to the original team objects."""
        return [
            [[self.teams[idx] for idx in field] for field in round_plan]
            for round_plan in index_plan
        ]

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

    def _field_imbalance(self, play_counts, field_counts):
        """Games that keep a team from using all fields evenly.

        A team should play on every field ``games // fields`` times or once
        more; every game above or below that range counts one.
        """
        imbalance = 0
        for team in self.teams:
            low = play_counts[team] // self.number_fields
            high = -(-play_counts[team] // self.number_fields)
            for field_idx in range(self.number_fields):
                count = field_counts[team][field_idx]
                imbalance += max(0, count - high) + max(0, low - count)
        return imbalance

    def _switch_spread(self, plan):
        counts = Counter()
        for round_plan in plan:
            for team in round_plan[0]:
                counts[team] += 1
        values = [counts[team] for team in self.teams]
        return max(values) - min(values)
