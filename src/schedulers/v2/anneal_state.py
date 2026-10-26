SITTING = -1

DUEL_GROWTH = 8.0
MISSING_SWITCH_WEIGHT = 30.0
MISSING_FIELD_WEIGHT = 0.25
FIELD_REPEAT_WEIGHT = 0.125


def free_duels(team_count, number_rounds, field_sizes):
    """How often every pair can meet before the plan runs out of new pairs."""
    pair_slots = number_rounds * sum(size * (size - 1) // 2 for size in field_sizes)
    pairs = team_count * (team_count - 1) // 2
    return max(1, pair_slots // max(1, pairs))


def duel_costs(free, max_count):
    costs = []
    for count in range(max_count + 1):
        over = count - free
        costs.append(sum(DUEL_GROWTH**k for k in range(over)) if over > 0 else 0.0)
    return costs


def field_costs(games, num_fields, number_rounds, tie_scale=1.0):
    """Cost of every field count, per team and field.

    Every game beyond a team's even share of a field (its games divided by
    the fields, rounded up) costs ``FIELD_REPEAT_WEIGHT``, never playing on a
    field costs ``MISSING_FIELD_WEIGHT``, and on field 0
    ``MISSING_SWITCH_WEIGHT`` on top. All of it weighs less than one repeated
    duel except the Main Switch. ``tie_scale`` multiplies the first two.
    """
    tables = []
    for team_games in games:
        share = -(-team_games // num_fields)
        team_table = []
        for field_idx in range(num_fields):
            row = []
            for count in range(number_rounds + 2):
                cost = tie_scale * FIELD_REPEAT_WEIGHT * max(0, count - share)
                if count == 0:
                    cost += tie_scale * MISSING_FIELD_WEIGHT
                    if field_idx == 0:
                        cost += MISSING_SWITCH_WEIGHT
                row.append(cost)
            team_table.append(row)
        tables.append(team_table)
    return tables


class AnnealState:
    """Plan of team indices with the counters behind the annealing energy.

    ``energy`` is the number the search minimises: ``duel_costs`` for every
    pair of teams plus ``field_costs`` for every team and field. Every move
    has a ``*_delta`` method that returns the change of the energy without
    touching the plan, and a method that applies the move and keeps the
    counters up to date, so a rejected move costs only its delta.

    ``field_of[round][team]`` is the field a team plays on in a round, or
    ``SITTING``; ``sitting[round]`` lists the teams that sit the round out.
    """

    def __init__(self, plan, team_count, field_sizes, games, tie_scale=1.0):
        self.team_count = team_count
        self.field_sizes = list(field_sizes)
        self.num_fields = len(field_sizes)
        self.num_rounds = len(plan)
        self.free = free_duels(team_count, self.num_rounds, field_sizes)

        costs = duel_costs(self.free, self.num_rounds + 2)
        self.duel_cost = costs
        self.duel_up = [costs[c + 1] - costs[c] for c in range(len(costs) - 1)]
        self.duel_down = [0.0] + [costs[c - 1] - costs[c] for c in range(1, len(costs))]
        self.field_cost = field_costs(
            games, self.num_fields, self.num_rounds, tie_scale
        )

        self.plan = [[list(field) for field in round_plan] for round_plan in plan]
        self.field_of = [[SITTING] * team_count for _ in range(self.num_rounds)]
        self.pair_counts = [0] * (team_count * team_count)
        self.field_counts = [0] * (team_count * self.num_fields)
        for round_idx, round_plan in enumerate(self.plan):
            for field_idx, field in enumerate(round_plan):
                for idx, team in enumerate(field):
                    self.field_of[round_idx][team] = field_idx
                    self.field_counts[team * self.num_fields + field_idx] += 1
                    for other in field[:idx]:
                        self.pair_counts[team * team_count + other] += 1
                        self.pair_counts[other * team_count + team] += 1
        self.sitting = [
            [team for team in range(team_count) if round_field[team] == SITTING]
            for round_field in self.field_of
        ]
        self.energy = self.compute_energy()

    def compute_energy(self):
        team_count = self.team_count
        fields = self.num_fields
        energy = 0.0
        for team in range(team_count):
            row = team * team_count
            for other in range(team + 1, team_count):
                energy += self.duel_cost[self.pair_counts[row + other]]
            table = self.field_cost[team]
            for field_idx in range(fields):
                energy += table[field_idx][self.field_counts[team * fields + field_idx]]
        return energy

    def snapshot(self):
        return [[list(field) for field in round_plan] for round_plan in self.plan]

    def swap_delta(self, round_idx, field_a, idx_a, field_b, idx_b):
        """Two teams of one round on different fields trade places."""
        team_count = self.team_count
        pairs = self.pair_counts
        up = self.duel_up
        down = self.duel_down
        round_plan = self.plan[round_idx]
        group_a = round_plan[field_a]
        group_b = round_plan[field_b]
        team_a = group_a[idx_a]
        team_b = group_b[idx_b]
        row_a = team_a * team_count
        row_b = team_b * team_count
        delta = 0.0
        for other in group_a:
            if other != team_a:
                delta += down[pairs[row_a + other]] + up[pairs[row_b + other]]
        for other in group_b:
            if other != team_b:
                delta += down[pairs[row_b + other]] + up[pairs[row_a + other]]
        fields = self.num_fields
        counts = self.field_counts
        table_a = self.field_cost[team_a]
        table_b = self.field_cost[team_b]
        a_on_a = counts[team_a * fields + field_a]
        a_on_b = counts[team_a * fields + field_b]
        b_on_a = counts[team_b * fields + field_a]
        b_on_b = counts[team_b * fields + field_b]
        return (
            delta
            + table_a[field_a][a_on_a - 1]
            - table_a[field_a][a_on_a]
            + table_a[field_b][a_on_b + 1]
            - table_a[field_b][a_on_b]
            + table_b[field_b][b_on_b - 1]
            - table_b[field_b][b_on_b]
            + table_b[field_a][b_on_a + 1]
            - table_b[field_a][b_on_a]
        )

    def swap(self, round_idx, field_a, idx_a, field_b, idx_b):
        round_plan = self.plan[round_idx]
        team_a = round_plan[field_a][idx_a]
        team_b = round_plan[field_b][idx_b]
        self._replace(round_plan[field_a], idx_a, team_b)
        self._replace(round_plan[field_b], idx_b, team_a)
        self._move_field(team_a, field_a, field_b)
        self._move_field(team_b, field_b, field_a)
        field_of = self.field_of[round_idx]
        field_of[team_a] = field_b
        field_of[team_b] = field_a

    def cross_delta(self, round_a, field_a, idx_a, round_b, field_b, idx_b):
        """Team a moves from ``round_a`` into team b's slot in ``round_b``.

        Team b takes a's slot in turn; a must sit out ``round_b`` and b
        ``round_a``, so both keep their number of games.
        """
        team_count = self.team_count
        pairs = self.pair_counts
        up = self.duel_up
        down = self.duel_down
        group_a = self.plan[round_a][field_a]
        group_b = self.plan[round_b][field_b]
        team_a = group_a[idx_a]
        team_b = group_b[idx_b]
        row_a = team_a * team_count
        row_b = team_b * team_count
        delta = 0.0
        # A team in both groups keeps its duels with a and b.
        for other in group_a:
            if other != team_a and other not in group_b:
                delta += down[pairs[row_a + other]] + up[pairs[row_b + other]]
        for other in group_b:
            if other != team_b and other not in group_a:
                delta += down[pairs[row_b + other]] + up[pairs[row_a + other]]
        if field_a != field_b:
            delta += self._field_trade_delta(team_a, field_a, team_b, field_b)
        return delta

    def cross(self, round_a, field_a, idx_a, round_b, field_b, idx_b):
        group_a = self.plan[round_a][field_a]
        group_b = self.plan[round_b][field_b]
        team_a = group_a[idx_a]
        team_b = group_b[idx_b]
        self._replace(group_a, idx_a, team_b)
        self._replace(group_b, idx_b, team_a)
        if field_a != field_b:
            self._move_field(team_a, field_a, field_b)
            self._move_field(team_b, field_b, field_a)
        self.field_of[round_a][team_a] = SITTING
        self.field_of[round_a][team_b] = field_a
        self.field_of[round_b][team_b] = SITTING
        self.field_of[round_b][team_a] = field_b
        sitting_a = self.sitting[round_a]
        sitting_a[sitting_a.index(team_b)] = team_a
        sitting_b = self.sitting[round_b]
        sitting_b[sitting_b.index(team_a)] = team_b

    def trade_delta(self, round_a, round_b, team_a, team_b):
        """Teams a and b trade places in two rounds.

        In ``round_a`` team a is where team b is in ``round_b`` and the other
        way round (a field or ``SITTING``), so after trading in both rounds
        every team keeps its field counts and games; only duels change.
        """
        team_count = self.team_count
        pairs = self.pair_counts
        costs = self.duel_cost
        field_x = self.field_of[round_a][team_a]
        field_y = self.field_of[round_b][team_a]
        plan_a = self.plan[round_a]
        plan_b = self.plan[round_b]
        # a leaves group_ax and joins group_ay; in round b it leaves group_by
        # and joins group_bx. b does the opposite.
        group_ax = plan_a[field_x] if field_x != SITTING else ()
        group_ay = plan_a[field_y] if field_y != SITTING else ()
        group_bx = plan_b[field_x] if field_x != SITTING else ()
        group_by = plan_b[field_y] if field_y != SITTING else ()
        row_a = team_a * team_count
        row_b = team_b * team_count
        delta = 0.0
        for other in group_ax:
            if other == team_a or other in group_bx:
                continue
            change = -2 if other in group_by else -1
            count_a = pairs[row_a + other]
            count_b = pairs[row_b + other]
            delta += costs[count_a + change] - costs[count_a]
            delta += costs[count_b - change] - costs[count_b]
        for other in group_ay:
            if other == team_b or other in group_by:
                continue
            change = 2 if other in group_bx else 1
            count_a = pairs[row_a + other]
            count_b = pairs[row_b + other]
            delta += costs[count_a + change] - costs[count_a]
            delta += costs[count_b - change] - costs[count_b]
        for other in group_bx:
            if other == team_b or other in group_ax or other in group_ay:
                continue
            count_a = pairs[row_a + other]
            count_b = pairs[row_b + other]
            delta += costs[count_a + 1] - costs[count_a]
            delta += costs[count_b - 1] - costs[count_b]
        for other in group_by:
            if other == team_a or other in group_ax or other in group_ay:
                continue
            count_a = pairs[row_a + other]
            count_b = pairs[row_b + other]
            delta += costs[count_a - 1] - costs[count_a]
            delta += costs[count_b + 1] - costs[count_b]
        return delta

    def trade(self, round_a, round_b, team_a, team_b):
        self._exchange(round_a, team_a, team_b)
        self._exchange(round_b, team_a, team_b)

    def field_swap_delta(self, round_idx, field_a, field_b):
        """Two games of the same size in one round trade their fields."""
        fields = self.num_fields
        counts = self.field_counts
        delta = 0.0
        for team in self.plan[round_idx][field_a]:
            table = self.field_cost[team]
            count_a = counts[team * fields + field_a]
            count_b = counts[team * fields + field_b]
            delta += table[field_a][count_a - 1] - table[field_a][count_a]
            delta += table[field_b][count_b + 1] - table[field_b][count_b]
        for team in self.plan[round_idx][field_b]:
            table = self.field_cost[team]
            count_a = counts[team * fields + field_a]
            count_b = counts[team * fields + field_b]
            delta += table[field_b][count_b - 1] - table[field_b][count_b]
            delta += table[field_a][count_a + 1] - table[field_a][count_a]
        return delta

    def field_swap(self, round_idx, field_a, field_b):
        round_plan = self.plan[round_idx]
        field_of = self.field_of[round_idx]
        for team in round_plan[field_a]:
            self._move_field(team, field_a, field_b)
            field_of[team] = field_b
        for team in round_plan[field_b]:
            self._move_field(team, field_b, field_a)
            field_of[team] = field_a
        round_plan[field_a], round_plan[field_b] = (
            round_plan[field_b],
            round_plan[field_a],
        )

    def conflict_slot(self, rand, tries=4):
        """Slot of a team in a duel played more often than free, if found."""
        team_count = self.team_count
        pairs = self.pair_counts
        free = self.free
        for _ in range(tries):
            round_idx = int(rand() * self.num_rounds)
            field_idx = int(rand() * self.num_fields)
            group = self.plan[round_idx][field_idx]
            size = len(group)
            for idx in range(size):
                row = group[idx] * team_count
                for other_idx in range(idx + 1, size):
                    if pairs[row + group[other_idx]] > free:
                        chosen = idx if rand() < 0.5 else other_idx
                        return round_idx, field_idx, chosen
        return None

    def conflict_slots(self):
        """All slots of teams in a duel played more often than free."""
        team_count = self.team_count
        pairs = self.pair_counts
        free = self.free
        slots = []
        for round_idx, round_plan in enumerate(self.plan):
            for field_idx, group in enumerate(round_plan):
                for idx, team in enumerate(group):
                    row = team * team_count
                    for other in group:
                        if other != team and pairs[row + other] > free:
                            slots.append((round_idx, field_idx, idx))
                            break
        return slots

    def _field_trade_delta(self, team_a, field_a, team_b, field_b):
        """Team a moves from field a to field b, team b the other way."""
        fields = self.num_fields
        counts = self.field_counts
        table_a = self.field_cost[team_a]
        table_b = self.field_cost[team_b]
        a_on_a = counts[team_a * fields + field_a]
        a_on_b = counts[team_a * fields + field_b]
        b_on_a = counts[team_b * fields + field_a]
        b_on_b = counts[team_b * fields + field_b]
        return (
            table_a[field_a][a_on_a - 1]
            - table_a[field_a][a_on_a]
            + table_a[field_b][a_on_b + 1]
            - table_a[field_b][a_on_b]
            + table_b[field_b][b_on_b - 1]
            - table_b[field_b][b_on_b]
            + table_b[field_a][b_on_a + 1]
            - table_b[field_a][b_on_a]
        )

    def _replace(self, group, idx, entering):
        """Put ``entering`` into ``group`` at ``idx`` and update the duels."""
        team_count = self.team_count
        pairs = self.pair_counts
        leaving = group[idx]
        for other in group:
            if other != leaving:
                pairs[leaving * team_count + other] -= 1
                pairs[other * team_count + leaving] -= 1
                pairs[entering * team_count + other] += 1
                pairs[other * team_count + entering] += 1
        group[idx] = entering

    def _move_field(self, team, from_field, to_field):
        counts = self.field_counts
        counts[team * self.num_fields + from_field] -= 1
        counts[team * self.num_fields + to_field] += 1

    def _exchange(self, round_idx, team_a, team_b):
        """Teams a and b trade places in one round; one of them may sit."""
        round_plan = self.plan[round_idx]
        field_of = self.field_of[round_idx]
        field_a = field_of[team_a]
        field_b = field_of[team_b]
        if field_a != SITTING and field_b != SITTING:
            self.swap(
                round_idx,
                field_a,
                round_plan[field_a].index(team_a),
                field_b,
                round_plan[field_b].index(team_b),
            )
            return
        if field_a == SITTING:
            team_a, team_b = team_b, team_a
            field_a = field_b
        # team_a plays on field_a, team_b sits.
        group = round_plan[field_a]
        self._replace(group, group.index(team_a), team_b)
        counts = self.field_counts
        counts[team_a * self.num_fields + field_a] -= 1
        counts[team_b * self.num_fields + field_a] += 1
        field_of[team_a] = SITTING
        field_of[team_b] = field_a
        sitting = self.sitting[round_idx]
        sitting[sitting.index(team_b)] = team_a
