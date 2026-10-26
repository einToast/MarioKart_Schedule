import math
import random
import time

from schedulers.v2.config import DEFAULT_MAX_ITERATIONS
from schedulers.v2.anneal_state import (
    MISSING_FIELD_WEIGHT,
    MISSING_SWITCH_WEIGHT,
    SITTING,
    AnnealState,
)

START_TEMPERATURE = 3.0
END_TEMPERATURE = 0.05
QUENCH_ITERATIONS = 60_000
QUENCH_TEMPERATURE = 0.3
QUENCH_TIE_SCALE = 32.0
QUENCH_END = 0.05
ANNEAL_END = 0.9
POLISH_STEPS = 500
TIDY_SWEEPS = 10
FIELD_SWAP_SHARE = 0.1
TRADE_SHARE = 0.3
CROSS_ROUND_SHARE = 0.4
TARGETED_SHARE = 0.3
SCHEDULE_INTERVAL = 1024
POLISH_CANDIDATES = 12
POLISH_SITTING_TEAMS = 6
TABU_TENURE = 6
EPSILON = 1e-9


class Annealer:
    """Simulated annealing on ``AnnealState.energy``, then a short tabu search.

    Used when there are at least as many teams as one round has seats. The
    search counts iterations, so a seed gives the same plan on every machine.
    If ``max_seconds`` runs out first, the temperature follows the elapsed
    time instead, so a cut search still cools down.
    """

    def __init__(
        self,
        teams,
        number_fields,
        number_rounds,
        teams_per_game,
        seed,
        max_seconds,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        polish_steps=POLISH_STEPS,
    ):
        self.teams = teams
        self.field_sizes = [teams_per_game] * number_fields
        self.number_fields = number_fields
        self.number_rounds = number_rounds
        self.seed = seed
        self.max_seconds = max_seconds
        self.max_iterations = max_iterations
        self.polish_steps = polish_steps
        self.capacity_per_round = sum(self.field_sizes)
        self.has_sitting = len(teams) > self.capacity_per_round and number_rounds > 1
        self._slots = [
            (field_idx, idx)
            for field_idx, size in enumerate(self.field_sizes)
            for idx in range(size)
        ]
        self._other_slots = [
            [slot for slot in self._slots if slot[0] != field_idx]
            for field_idx in range(self.number_fields)
        ]
        self._equal_size_fields = [
            (field_a, field_b)
            for field_a in range(self.number_fields)
            for field_b in range(self.number_fields)
            if field_a != field_b
            and self.field_sizes[field_a] == self.field_sizes[field_b]
        ]

    def create_plan(self):
        started_at = time.monotonic()
        deadline = started_at + self.max_seconds
        rng = random.Random(self.seed)
        plan, games = self._initial_plan(rng)
        if not self._has_moves():
            return self._to_teams(plan)

        # Easy plans reach the lower bound in a short, cold run already.
        state = AnnealState(
            plan, len(self.teams), self.field_sizes, games, QUENCH_TIE_SCALE
        )
        quench_plan, _ = self._anneal(
            state,
            rng,
            min(QUENCH_ITERATIONS, self.max_iterations),
            started_at + self.max_seconds * QUENCH_END,
            self._energy_lower_bound(games, state.free, QUENCH_TIE_SCALE),
            QUENCH_TEMPERATURE,
        )
        state = AnnealState(quench_plan, len(self.teams), self.field_sizes, games)
        lower_bound = self._energy_lower_bound(games, state.free)
        best_plan, best_energy = state.snapshot(), state.energy
        if best_energy > lower_bound + EPSILON:
            plan, energy = self._anneal(
                state,
                rng,
                self.max_iterations,
                started_at + self.max_seconds * ANNEAL_END,
                lower_bound,
                START_TEMPERATURE,
            )
            if energy < best_energy - EPSILON:
                best_plan, best_energy = plan, energy
        if self.polish_steps and best_energy > lower_bound + EPSILON:
            state = AnnealState(best_plan, len(self.teams), self.field_sizes, games)
            best_plan, best_energy = self._polish(state, rng, deadline, lower_bound)
        self._tidy_fields(best_plan, deadline)
        return self._to_teams(best_plan)

    def _has_moves(self):
        """Whether any move can change the plan."""
        return self.number_fields > 1 or self.has_sitting

    def _initial_plan(self, rng):
        """Lay the teams out round after round in one random order.

        Every round takes the next ``capacity_per_round`` teams of the
        repeated order, so no team plays twice in a round and play counts
        differ by at most one.
        """
        team_count = len(self.teams)
        order = list(range(team_count))
        rng.shuffle(order)
        total_slots = self.capacity_per_round * self.number_rounds
        sequence = [order[idx % team_count] for idx in range(total_slots)]
        games = [0] * team_count
        for team in sequence:
            games[team] += 1

        plan = []
        for round_idx in range(self.number_rounds):
            start = round_idx * self.capacity_per_round
            entries = sequence[start : start + self.capacity_per_round]
            rng.shuffle(entries)
            round_plan = []
            for size in self.field_sizes:
                round_plan.append(entries[:size])
                entries = entries[size:]
            plan.append(round_plan)
        return plan, games

    def _energy_lower_bound(self, games, free, tie_scale=1.0):
        """Energy no plan can go below; reaching it ends the search early."""
        team_count = len(self.teams)
        pairs = team_count * (team_count - 1) // 2
        pair_slots = self.number_rounds * sum(
            size * (size - 1) // 2 for size in self.field_sizes
        )
        # Every duel beyond ``free`` per pair costs at least 1.
        extra_duels = max(0, pair_slots - pairs * free)
        # A team meets at least ``smallest game - 1`` opponents per game but
        # only has ``team_count - 1`` of them.
        opponents_per_game = min(self.field_sizes) - 1
        surplus = sum(
            max(0, team_games * opponents_per_game - (team_count - 1) * free)
            for team_games in games
        )
        duel_bound = max(extra_duels, -(-surplus // 2))
        switch_slots = self.number_rounds * self.field_sizes[0]
        missing_switch = max(0, team_count - switch_slots)
        # A team with fewer games than fields misses some of them, and a field
        # with fewer seats than teams is missed by some teams.
        missing_fields = max(
            sum(max(0, self.number_fields - team_games) for team_games in games),
            sum(
                max(0, team_count - self.number_rounds * size)
                for size in self.field_sizes
            ),
        )
        return (
            duel_bound
            + missing_switch * MISSING_SWITCH_WEIGHT
            + missing_fields * MISSING_FIELD_WEIGHT * tie_scale
        )

    def _anneal(
        self, state, rng, iterations, end_time, lower_bound, start_temperature
    ):
        """Anneal ``state`` from ``start_temperature`` down to ``END_TEMPERATURE``.

        The search ends after ``iterations`` or at ``end_time``, whichever
        comes first, and cools down with the larger share of both. Returns the
        best plan and its energy; ``state`` keeps the last plan.
        """
        rand = rng.random
        exp = math.exp
        rounds = self.number_rounds
        fields = self.number_fields
        slots = self._slots
        slot_count = len(slots)
        plan_slots = rounds * slot_count
        other_slots = self._other_slots
        equal_fields = self._equal_size_fields
        field_swap_share = FIELD_SWAP_SHARE if equal_fields else 0.0
        trade_share = field_swap_share + (TRADE_SHARE if rounds > 1 else 0.0)
        cross_share = CROSS_ROUND_SHARE if self.has_sitting else 0.0
        cooling = math.log(END_TEMPERATURE / start_temperature)
        started_at = time.monotonic()
        duration = max(end_time - started_at, 1e-9)

        energy = state.energy
        best_energy = energy
        best_plan = state.snapshot()
        temperature = start_temperature

        for iteration in range(iterations):
            if iteration % SCHEDULE_INTERVAL == 0:
                progress = max(
                    iteration / iterations,
                    (time.monotonic() - started_at) / duration,
                )
                if best_energy <= lower_bound + EPSILON or progress >= 1.0:
                    break
                temperature = start_temperature * exp(cooling * progress)

            choice = rand()
            if choice < field_swap_share:
                round_idx = int(rand() * rounds)
                field_a, field_b = equal_fields[int(rand() * len(equal_fields))]
                delta = state.field_swap_delta(round_idx, field_a, field_b)
                if delta > 0 and rand() >= exp(-delta / temperature):
                    continue
                state.field_swap(round_idx, field_a, field_b)
            elif choice < trade_share:
                move = self._pick_trade(state, rand)
                if move is None:
                    continue
                delta = state.trade_delta(*move)
                if delta > 0 and rand() >= exp(-delta / temperature):
                    continue
                state.trade(*move)
            else:
                slot = state.conflict_slot(rand) if rand() < TARGETED_SHARE else None
                if slot is None:
                    pick = int(rand() * plan_slots)
                    round_a = pick // slot_count
                    field_a, idx_a = slots[pick % slot_count]
                else:
                    round_a, field_a, idx_a = slot
                move = None
                if cross_share and rand() < cross_share:
                    move = self._pick_cross(state, rand, round_a, field_a, idx_a)
                if move is not None:
                    delta = state.cross_delta(*move)
                    if delta > 0 and rand() >= exp(-delta / temperature):
                        continue
                    state.cross(*move)
                elif fields > 1:
                    partners = other_slots[field_a]
                    field_b, idx_b = partners[int(rand() * len(partners))]
                    delta = state.swap_delta(round_a, field_a, idx_a, field_b, idx_b)
                    if delta > 0 and rand() >= exp(-delta / temperature):
                        continue
                    state.swap(round_a, field_a, idx_a, field_b, idx_b)
                else:
                    continue

            energy += delta
            if energy < best_energy - EPSILON:
                best_energy = energy
                best_plan = state.snapshot()

        state.energy = energy
        return best_plan, best_energy

    def _pick_cross(self, state, rand, round_a, field_a, idx_a):
        """Move for ``AnnealState.cross``, or ``None`` if the team plays both rounds."""
        team_a = state.plan[round_a][field_a][idx_a]
        round_b = int(rand() * (self.number_rounds - 1))
        if round_b >= round_a:
            round_b += 1
        if state.field_of[round_b][team_a] != SITTING:
            return None
        waiting = state.sitting[round_a]
        team_b = waiting[int(rand() * len(waiting))]
        field_b = state.field_of[round_b][team_b]
        if field_b == SITTING:
            return None
        idx_b = state.plan[round_b][field_b].index(team_b)
        return round_a, field_a, idx_a, round_b, field_b, idx_b

    def _pick_trade(self, state, rand):
        """Move for ``AnnealState.trade``, or ``None`` if there is no partner."""
        rounds = self.number_rounds
        round_a = int(rand() * rounds)
        round_b = int(rand() * (rounds - 1))
        if round_b >= round_a:
            round_b += 1
        team_a = int(rand() * len(self.teams))
        field_x = state.field_of[round_a][team_a]
        field_y = state.field_of[round_b][team_a]
        if field_x == field_y:
            return None
        if field_y == SITTING:
            candidates = state.sitting[round_a]
        else:
            candidates = state.plan[round_a][field_y]
        count = len(candidates)
        if not count:
            return None
        start = int(rand() * count)
        field_of_b = state.field_of[round_b]
        for offset in range(count):
            team_b = candidates[(start + offset) % count]
            if field_of_b[team_b] == field_x:
                return round_a, round_b, team_a, team_b
        return None

    def _polish(self, state, rng, deadline, lower_bound):
        """Tabu search over moves of teams in repeated duels.

        Every step applies the best move of a few such teams, even if it makes
        the plan worse, and keeps the moved teams in their rounds for a few
        steps unless a move beats the best plan so far.
        """
        energy = state.energy
        best_energy = energy
        best_plan = state.snapshot()
        tabu = {}

        for step in range(self.polish_steps):
            if best_energy <= lower_bound + EPSILON:
                break
            if step % 32 == 0 and time.monotonic() > deadline:
                break
            slots = state.conflict_slots()
            if not slots:
                break
            if len(slots) > POLISH_CANDIDATES:
                slots = rng.sample(slots, POLISH_CANDIDATES)

            best_move = self._best_polish_move(
                state, rng, slots, tabu, step, best_energy - energy
            )
            if best_move is None:
                continue
            delta, move = best_move
            moved = self._apply_polish_move(state, move)
            energy += delta
            until = step + TABU_TENURE + int(rng.random() * 3)
            for team_round in moved:
                tabu[team_round] = until
            if energy < best_energy - EPSILON:
                best_energy = energy
                best_plan = state.snapshot()

        return best_plan, best_energy

    def _best_polish_move(self, state, rng, slots, tabu, step, aspiration):
        """Best ``(delta, move)`` for ``slots``; ties are broken at random."""
        plan = state.plan
        best_delta = math.inf
        best_move = None
        ties = 0
        for round_a, field_a, idx_a in slots:
            team_a = plan[round_a][field_a][idx_a]
            for move, teams_rounds in self._polish_moves(
                state, rng, round_a, field_a, idx_a, team_a
            ):
                if move[0] == "swap":
                    delta = state.swap_delta(*move[1:])
                else:
                    delta = state.cross_delta(*move[1:])
                if delta >= aspiration - EPSILON and any(
                    tabu.get(team_round, -1) > step for team_round in teams_rounds
                ):
                    continue
                if delta < best_delta - EPSILON:
                    best_delta = delta
                    best_move = move
                    ties = 1
                elif delta < best_delta + EPSILON:
                    ties += 1
                    if rng.random() * ties < 1:
                        best_move = move
        if best_move is None:
            return None
        return best_delta, best_move

    def _polish_moves(self, state, rng, round_a, field_a, idx_a, team_a):
        """Swaps inside the round and with teams that sit the round out.

        Yields every move with the ``(team, round)`` places it takes teams out
        of, which must not be tabu.
        """
        plan = state.plan
        for field_b, idx_b in self._slots:
            if field_b == field_a:
                continue
            team_b = plan[round_a][field_b][idx_b]
            move = ("swap", round_a, field_a, idx_a, field_b, idx_b)
            yield move, ((team_a, round_a), (team_b, round_a))
        if not self.has_sitting:
            return
        waiting = state.sitting[round_a]
        if len(waiting) > POLISH_SITTING_TEAMS:
            waiting = rng.sample(waiting, POLISH_SITTING_TEAMS)
        field_of = state.field_of
        for team_b in waiting:
            for round_b in range(self.number_rounds):
                field_b = field_of[round_b][team_b]
                if field_b == SITTING or field_of[round_b][team_a] != SITTING:
                    continue
                idx_b = plan[round_b][field_b].index(team_b)
                move = ("cross", round_a, field_a, idx_a, round_b, field_b, idx_b)
                yield move, ((team_a, round_a), (team_b, round_b))

    @staticmethod
    def _apply_polish_move(state, move):
        """Apply a move of ``_polish_moves``; return the ``(team, round)`` it fills."""
        if move[0] == "swap":
            _, round_idx, field_a, idx_a, field_b, idx_b = move
            team_a = state.plan[round_idx][field_a][idx_a]
            team_b = state.plan[round_idx][field_b][idx_b]
            state.swap(round_idx, field_a, idx_a, field_b, idx_b)
            return (team_a, round_idx), (team_b, round_idx)
        _, round_a, field_a, idx_a, round_b, field_b, idx_b = move
        team_a = state.plan[round_a][field_a][idx_a]
        team_b = state.plan[round_b][field_b][idx_b]
        state.cross(round_a, field_a, idx_a, round_b, field_b, idx_b)
        return (team_a, round_b), (team_b, round_a)

    def _tidy_fields(self, plan, deadline):
        """Swap whole games between fields while the field use gets better.

        Swapping two games of a round keeps every duel, so this only works on
        what comes after the duels in v2's order: the Main Switch, the most
        games a team plays on one field, the fewest fields a team plays on,
        and how evenly the teams share the Main Switch. ``plan`` is changed in
        place.
        """
        counts = [[0] * self.number_fields for _ in self.teams]
        for round_plan in plan:
            for field_idx, game in enumerate(round_plan):
                for team in game:
                    counts[team][field_idx] += 1
        best_key = self._field_use_key(counts)
        for _ in range(TIDY_SWEEPS):
            improved = False
            for round_plan in plan:
                for field_a in range(self.number_fields):
                    for field_b in range(field_a + 1, self.number_fields):
                        if time.monotonic() > deadline:
                            return
                        self._move_games(counts, round_plan, field_a, field_b)
                        key = self._field_use_key(counts)
                        if key < best_key:
                            best_key = key
                            improved = True
                        else:
                            self._move_games(counts, round_plan, field_b, field_a)
            if not improved:
                return

    @staticmethod
    def _move_games(counts, round_plan, field_a, field_b):
        """Swap the games on two fields of a round and update ``counts``."""
        for team in round_plan[field_a]:
            counts[team][field_a] -= 1
            counts[team][field_b] += 1
        for team in round_plan[field_b]:
            counts[team][field_b] -= 1
            counts[team][field_a] += 1
        round_plan[field_a], round_plan[field_b] = (
            round_plan[field_b],
            round_plan[field_a],
        )

    @staticmethod
    def _field_use_key(counts):
        """Field use in v2's order, then how many teams sit at each extreme.

        The two counts only break ties, so a lower key never makes the plan
        worse in v2's order.
        """
        most = max(max(team_counts) for team_counts in counts)
        at_most = sum(team_counts.count(most) for team_counts in counts)
        distinct = [
            sum(1 for count in team_counts if count) for team_counts in counts
        ]
        fewest = min(distinct)
        switch = [team_counts[0] for team_counts in counts]
        return (
            switch.count(0),
            most,
            -fewest,
            sum(sum(team_counts) for team_counts in counts) - sum(distinct),
            max(switch) - min(switch),
            at_most,
            distinct.count(fewest),
        )

    def _to_teams(self, index_plan):
        """Convert a plan of team indices back to the original team objects."""
        return [
            [[self.teams[idx] for idx in field] for field in round_plan]
            for round_plan in index_plan
        ]
