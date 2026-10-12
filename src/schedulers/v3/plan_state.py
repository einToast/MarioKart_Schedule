PAIR_WEIGHT = 8
FIELD_WEIGHT = 2
FIELD_IMBALANCE_WEIGHT = 24
MISSING_SWITCH_WEIGHT = 40


class PlanState:
    """Plan with incrementally maintained counters behind the optimizer score.

    ``score()`` returns the same tuple as ``ScheduleOptimizer.score_plan``, but
    a move only updates the counters of the fields it touches instead of
    rescanning the plan.

    ``energy`` is a single number the annealing search minimises. It grows
    quadratically with how often a team plays on the same field and with how
    often a duel repeats beyond ``free_duels``, and adds a penalty for every
    team that never plays on field 0 and for every game counted by
    ``field_imbalance``.

    ``field_imbalance`` counts the games that keep a team from using all
    fields evenly: a team with ``games`` games should play on every field
    either ``games // fields`` times or once more, and every game above or
    below that range counts one.
    """

    def __init__(self, team_order, plan, free_duels=1):
        self.team_count = len(team_order)
        self.num_rounds = len(plan)
        self.num_fields = len(plan[0])
        team_count = self.team_count

        self.plan = [
            [[team_order[team] for team in field] for field in round_plan]
            for round_plan in plan
        ]
        self.pair_counts = [0] * (team_count * team_count)
        self.pair_costs = [
            PAIR_WEIGHT * max(0, count + 1 - free_duels) ** 2
            for count in range(self.num_rounds * self.num_fields + 1)
        ]
        self.pair_histogram = [0] * (self.num_rounds * self.num_fields + 2)
        self.pair_histogram[0] = team_count * (team_count - 1) // 2
        self.max_duel = 0
        self.repeat_pairs = 0
        self.repeat_excess = 0
        self.field_counts = [[0] * self.num_fields for _ in range(team_count)]
        self.max_field_repeat = [0] * team_count
        self.distinct_fields = [0] * team_count
        self.field_repeat_excess = 0
        self.round_counts = [[0] * team_count for _ in range(self.num_rounds)]
        self.round_duplicates = [0] * self.num_rounds
        self.duplicate_total = 0
        self.missing_switch = team_count
        self.energy = team_count * MISSING_SWITCH_WEIGHT

        games = [0] * team_count
        for round_plan in self.plan:
            for field in round_plan:
                for team in field:
                    games[team] += 1
        self.field_low = [count // self.num_fields for count in games]
        self.field_high = [-(-count // self.num_fields) for count in games]
        self.field_imbalance = sum(self.field_low) * self.num_fields
        self.energy += self.field_imbalance * FIELD_IMBALANCE_WEIGHT

        for round_idx, round_plan in enumerate(self.plan):
            for field_idx, field in enumerate(round_plan):
                for idx, team in enumerate(field):
                    for other in field[:idx]:
                        self._add_pair(team, other)
                    self._add_field(team, field_idx)
                    self._add_round(round_idx, team)

        played = [sum(counts) for counts in self.field_counts]
        played = [count for count in played if count]
        self.play_spread = max(played) - min(played) if played else 0

    def score(self):
        switch_counts = [counts[0] for counts in self.field_counts]
        return (
            self.duplicate_total,
            self.missing_switch,
            self.field_imbalance,
            self.max_duel,
            self.pair_histogram[self.max_duel],
            self.repeat_pairs,
            self.repeat_excess,
            self.play_spread,
            max(self.max_field_repeat),
            -min(self.distinct_fields),
            self.field_repeat_excess,
            max(switch_counts) - min(switch_counts),
        )

    def swap_is_valid(self, pos_a, pos_b, allowed_round_duplicates):
        round_a, field_a, idx_a = pos_a
        round_b, field_b, idx_b = pos_b
        if round_a == round_b and field_a == field_b:
            return False

        team_a = self.plan[round_a][field_a][idx_a]
        team_b = self.plan[round_b][field_b][idx_b]
        if team_a == team_b:
            return False
        if team_b in self.plan[round_a][field_a]:
            return False
        if team_a in self.plan[round_b][field_b]:
            return False
        if round_a == round_b:
            return True

        return self._round_stays_valid(
            round_a, team_a, team_b, allowed_round_duplicates
        ) and self._round_stays_valid(
            round_b, team_b, team_a, allowed_round_duplicates
        )

    def swap(self, pos_a, pos_b):
        """Swap two slots; calling it again with the same arguments undoes it."""
        round_a, field_a, idx_a = pos_a
        round_b, field_b, idx_b = pos_b
        team_a = self.plan[round_a][field_a][idx_a]
        team_b = self.plan[round_b][field_b][idx_b]

        self._leave(round_a, field_a, idx_a)
        self._leave(round_b, field_b, idx_b)
        self.plan[round_a][field_a][idx_a] = team_b
        self.plan[round_b][field_b][idx_b] = team_a
        self._enter(round_a, field_a, idx_a)
        self._enter(round_b, field_b, idx_b)

    def swap_fields(self, round_idx, field_a, field_b):
        """Swap two whole fields of a round; calling it again undoes it.

        Only the field counters change, every duel stays the same.
        """
        round_plan = self.plan[round_idx]
        for team in round_plan[field_a]:
            self._remove_field(team, field_a)
            self._add_field(team, field_b)
        for team in round_plan[field_b]:
            self._remove_field(team, field_b)
            self._add_field(team, field_a)
        round_plan[field_a], round_plan[field_b] = (
            round_plan[field_b],
            round_plan[field_a],
        )

    def repeated_pair_position(self, rng, tries=6):
        """Slot of a team in a field that hosts a most-repeated duel, if found."""
        max_duel = self.max_duel
        if max_duel < 2:
            return None
        for _ in range(tries):
            round_idx = int(rng.random() * self.num_rounds)
            field_idx = int(rng.random() * self.num_fields)
            field = self.plan[round_idx][field_idx]
            for idx, team_a in enumerate(field):
                for other_idx in range(idx + 1, len(field)):
                    key = self._pair_key(team_a, field[other_idx])
                    if self.pair_counts[key] >= max_duel:
                        chosen = idx if rng.random() < 0.5 else other_idx
                        return round_idx, field_idx, chosen
        return None

    def snapshot(self):
        return [[list(field) for field in round_plan] for round_plan in self.plan]

    def _round_stays_valid(self, round_idx, leaving, entering, allowed_duplicates):
        counts = self.round_counts[round_idx]
        duplicates = self.round_duplicates[round_idx]
        if counts[leaving] > 1:
            duplicates -= 1
        if counts[entering] > 0:
            duplicates += 1
        return duplicates <= allowed_duplicates

    def _leave(self, round_idx, field_idx, idx):
        field = self.plan[round_idx][field_idx]
        team = field[idx]
        for other_idx, other in enumerate(field):
            if other_idx != idx:
                self._remove_pair(team, other)
        self._remove_field(team, field_idx)
        self._remove_round(round_idx, team)

    def _enter(self, round_idx, field_idx, idx):
        field = self.plan[round_idx][field_idx]
        team = field[idx]
        for other_idx, other in enumerate(field):
            if other_idx != idx:
                self._add_pair(team, other)
        self._add_field(team, field_idx)
        self._add_round(round_idx, team)

    def _pair_key(self, team_a, team_b):
        if team_a < team_b:
            return team_a * self.team_count + team_b
        return team_b * self.team_count + team_a

    def _add_pair(self, team_a, team_b):
        key = self._pair_key(team_a, team_b)
        count = self.pair_counts[key]
        self.pair_counts[key] = count + 1
        self.energy += self.pair_costs[count]
        self.pair_histogram[count] -= 1
        self.pair_histogram[count + 1] += 1
        if count + 1 > self.max_duel:
            self.max_duel = count + 1
        if count >= 1:
            self.repeat_excess += 1
            if count == 1:
                self.repeat_pairs += 1

    def _remove_pair(self, team_a, team_b):
        key = self._pair_key(team_a, team_b)
        count = self.pair_counts[key]
        self.pair_counts[key] = count - 1
        self.energy -= self.pair_costs[count - 1]
        self.pair_histogram[count] -= 1
        self.pair_histogram[count - 1] += 1
        while self.max_duel and not self.pair_histogram[self.max_duel]:
            self.max_duel -= 1
        if count >= 2:
            self.repeat_excess -= 1
            if count == 2:
                self.repeat_pairs -= 1

    def _add_field(self, team, field_idx):
        counts = self.field_counts[team]
        count = counts[field_idx]
        counts[field_idx] = count + 1
        self.energy += count * FIELD_WEIGHT
        if count < self.field_low[team]:
            self.field_imbalance -= 1
            self.energy -= FIELD_IMBALANCE_WEIGHT
        elif count >= self.field_high[team]:
            self.field_imbalance += 1
            self.energy += FIELD_IMBALANCE_WEIGHT
        if count == 0:
            self.distinct_fields[team] += 1
            if field_idx == 0:
                self.missing_switch -= 1
                self.energy -= MISSING_SWITCH_WEIGHT
        else:
            self.field_repeat_excess += 1
        if count + 1 > self.max_field_repeat[team]:
            self.max_field_repeat[team] = count + 1

    def _remove_field(self, team, field_idx):
        counts = self.field_counts[team]
        count = counts[field_idx]
        counts[field_idx] = count - 1
        self.energy -= (count - 1) * FIELD_WEIGHT
        if count <= self.field_low[team]:
            self.field_imbalance += 1
            self.energy += FIELD_IMBALANCE_WEIGHT
        elif count > self.field_high[team]:
            self.field_imbalance -= 1
            self.energy -= FIELD_IMBALANCE_WEIGHT
        if count == 1:
            self.distinct_fields[team] -= 1
            if field_idx == 0:
                self.missing_switch += 1
                self.energy += MISSING_SWITCH_WEIGHT
        else:
            self.field_repeat_excess -= 1
        if count == self.max_field_repeat[team]:
            self.max_field_repeat[team] = max(counts)

    def _add_round(self, round_idx, team):
        counts = self.round_counts[round_idx]
        if counts[team]:
            self.round_duplicates[round_idx] += 1
            self.duplicate_total += 1
        counts[team] += 1

    def _remove_round(self, round_idx, team):
        counts = self.round_counts[round_idx]
        counts[team] -= 1
        if counts[team]:
            self.round_duplicates[round_idx] -= 1
            self.duplicate_total -= 1
