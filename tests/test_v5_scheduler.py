import random
import sys
import time
import unittest
from collections import Counter

sys.path.insert(0, "src")

from schedulers.v5.benchmark import collect_metrics
from schedulers.v5.config import resolve_seed
from schedulers.v5.generate_gameplay_lists import (
    create_plan as wrapper_create_plan,
    create_team_list,
    generate_plan,
    get_unrated_games,
)
from schedulers.v5.optimizer import ScheduleOptimizer
from schedulers.v5.plan_state import SITTING, PlanState
from schedulers.v5.scheduler import create_plan
from schedulers.v5.utils import field_share_bounds, flatten


TEST_ITERATIONS = 100_000
TEST_POLISH_STEPS = 100


def make_optimizer(num_teams, field_sizes, iterations=TEST_ITERATIONS):
    return ScheduleOptimizer(
        teams=create_team_list(num_teams),
        field_sizes=field_sizes,
        number_rounds=8,
        seed=12345,
        max_seconds=60.0,
        max_iterations=iterations,
        polish_steps=TEST_POLISH_STEPS,
    )


def make_plan(num_teams, field_sizes, seed=0, number_rounds=8):
    teams = create_team_list(num_teams)
    return create_plan(
        teams,
        len(field_sizes),
        number_rounds,
        field_sizes,
        seed=seed,
        max_iterations=TEST_ITERATIONS,
        polish_steps=TEST_POLISH_STEPS,
    )


def make_state(num_teams, field_sizes, seed=0):
    optimizer = make_optimizer(num_teams, field_sizes)
    plan, games = optimizer._initial_plan(random.Random(seed))
    return PlanState(plan, num_teams, field_sizes, games)


def recomputed(state):
    games = Counter(flatten(state.plan))
    return PlanState(
        state.plan,
        state.team_count,
        state.field_sizes,
        [games[team] for team in range(state.team_count)],
    )


class SchedulerV5Tests(unittest.TestCase):
    def assert_valid_metrics(
        self, metrics, field_sizes, num_rounds, max_duel_repeat=None
    ):
        self.assertTrue(metrics["shape_ok"])
        self.assertEqual([], metrics["missing switch-1 teams"])
        self.assertEqual(1, len(metrics["rating-count set"]))
        self.assertEqual(0, metrics["same-round duplicates"])
        self.assertEqual(0, metrics["games outside a field's share"])

        play_counts = metrics["play-count set"]
        self.assertLessEqual(max(play_counts) - min(play_counts), 1)

        if max_duel_repeat is not None:
            self.assertLessEqual(metrics["max duel repeat"], max_duel_repeat)

    def assert_state_matches_plan(self, state):
        fresh = recomputed(state)
        self.assertEqual(fresh.pair_counts, state.pair_counts)
        self.assertEqual(fresh.field_counts, state.field_counts)
        self.assertEqual(fresh.field_of, state.field_of)
        self.assertEqual(
            sorted(map(sorted, fresh.sitting)), sorted(map(sorted, state.sitting))
        )
        self.assertAlmostEqual(fresh.energy, state.compute_energy())

    def test_main_tournament_shape_and_quality(self):
        plan = make_plan(20, [4, 4, 4, 4])

        metrics = collect_metrics(plan, 20, 4, 8)
        self.assert_valid_metrics(metrics, [4, 4, 4, 4], 8, max_duel_repeat=2)

    def test_every_field_gets_its_own_number_of_teams(self):
        plan = make_plan(20, [4, 4, 2])

        for round_plan in plan:
            self.assertEqual([4, 4, 2], [len(field) for field in round_plan])
        metrics = collect_metrics(plan, 20, 3, 8, [4, 4, 2])
        self.assert_valid_metrics(metrics, [4, 4, 2], 8, max_duel_repeat=1)

    def test_teams_use_the_fields_in_proportion_to_their_size(self):
        for num_teams, field_sizes in ((12, [4, 4, 2]), (16, [4, 3, 2, 2])):
            plan = make_plan(num_teams, field_sizes)

            games = Counter(flatten(plan))
            field_counts = {team: [0] * len(field_sizes) for team in games}
            for round_plan in plan:
                for field_idx, field_plan in enumerate(round_plan):
                    for team in field_plan:
                        field_counts[team][field_idx] += 1
            for team, counts in field_counts.items():
                low, high = field_share_bounds(games[team], field_sizes)
                for field_idx, count in enumerate(counts):
                    self.assertGreaterEqual(count, low[field_idx], (team, counts))
                    self.assertLessEqual(count, high[field_idx], (team, counts))

    def test_large_tournament_has_no_repeated_duels(self):
        plan = make_plan(28, [4, 4, 4, 4])

        metrics = collect_metrics(plan, 28, 4, 8)
        self.assert_valid_metrics(metrics, [4, 4, 4, 4], 8, max_duel_repeat=1)

    def test_every_team_plays_every_round_when_seats_match_teams(self):
        plan = make_plan(8, [4, 4])

        for round_plan in plan:
            self.assertEqual(list(range(8)), sorted(flatten(round_plan)))
        metrics = collect_metrics(plan, 8, 2, 8)
        self.assert_valid_metrics(metrics, [4, 4], 8, max_duel_repeat=4)

    def test_rating_plan_marks_equal_rated_game_counts(self):
        plan = make_plan(18, [4, 4, 4, 4])

        rated = Counter()
        for round_plan, rate_round in zip(plan, get_unrated_games(plan)):
            for field_plan, rate_field in zip(round_plan, rate_round):
                for team, is_rated in zip(field_plan, rate_field):
                    if is_rated:
                        rated[team] += 1
        self.assertEqual(18, len(rated))
        self.assertEqual(1, len(set(rated.values())))

    def test_generate_plan_keeps_legacy_return_shape(self):
        plan, max_games_count = generate_plan(26, 4, 8, 4)

        self.assertEqual(8, len(plan))
        self.assertEqual({4}, {len(round_plan) for round_plan in plan})
        self.assertEqual({4}, max_games_count)

    def test_generate_plan_accepts_a_size_per_field(self):
        plan, max_games_count = generate_plan(20, 3, 8, [4, 4, 2])

        self.assertEqual(8, len(plan))
        self.assertEqual([4, 4, 2], [len(field) for field in plan[0]])
        self.assertEqual({4}, max_games_count)

    def test_wrapper_create_plan_matches_scheduler_api(self):
        teams = create_team_list(24)
        self.assertEqual(
            create_plan(teams, 4, 8, 4, seed=3, max_iterations=20_000),
            wrapper_create_plan(teams, 4, 8, 4, seed=3, max_iterations=20_000),
        )

    def test_same_seed_gives_same_plan(self):
        self.assertEqual(make_plan(17, [4, 4, 4, 4]), make_plan(17, [4, 4, 4, 4]))
        self.assertNotEqual(
            make_plan(17, [4, 4, 4, 4], seed=1), make_plan(17, [4, 4, 4, 4], seed=2)
        )

    def test_resolve_seed_depends_on_the_field_sizes(self):
        self.assertEqual(7, resolve_seed(7, 20, 8, [4, 4, 4]))
        self.assertEqual(
            resolve_seed(None, 20, 8, [4, 4, 2]), resolve_seed(None, 20, 8, [4, 4, 2])
        )
        self.assertNotEqual(
            resolve_seed(None, 20, 8, [4, 4, 2]), resolve_seed(None, 20, 8, [4, 2, 4])
        )

    def test_invalid_inputs_raise_value_error(self):
        teams = create_team_list(20)
        invalid = (
            {"number_fields": 0},
            {"number_rounds": 0},
            {"teams_per_game": 0},
            {"teams_per_game": [4, 4]},
            {"teams_per_game": [4, 4, 4, 0]},
            {"teams_per_game": [4, 4, 4, 2.5]},
            {"teams_per_game": [4, 4, 4, True]},
        )
        for kwargs in invalid:
            with self.assertRaises(ValueError, msg=kwargs):
                create_plan(teams, **kwargs)

    def test_fewer_teams_than_seats_per_round_raise_value_error(self):
        with self.assertRaises(ValueError):
            create_plan(create_team_list(15))
        with self.assertRaises(ValueError):
            create_plan(create_team_list(9), 3, 8, [4, 4, 2])

    def test_more_teams_than_game_slots_raise_value_error(self):
        with self.assertRaises(ValueError):
            create_plan(create_team_list(33), 4, 2, 4)

    def test_empty_team_list_returns_empty_plan(self):
        self.assertEqual([], create_plan([]))

    def test_keeps_the_original_team_objects(self):
        teams = [f"team-{idx}" for idx in range(16)]
        plan = create_plan(teams, max_iterations=5_000, polish_steps=0)

        self.assertEqual(sorted(teams), sorted(set(flatten(plan))))

    def test_time_budget_cuts_the_search_short(self):
        started_at = time.monotonic()
        plan = create_plan(
            create_team_list(20), max_seconds=0.05, max_iterations=10_000_000
        )

        self.assertLess(time.monotonic() - started_at, 2.0)
        self.assertTrue(collect_metrics(plan, 20, 4, 8)["shape_ok"])

    def test_initial_plan_spreads_the_games(self):
        optimizer = make_optimizer(19, [4, 4, 2])
        plan, games = optimizer._initial_plan(random.Random(1))

        self.assertEqual(Counter(flatten(plan)), Counter(dict(enumerate(games))))
        self.assertLessEqual(max(games) - min(games), 1)
        for round_plan in plan:
            round_teams = flatten(round_plan)
            self.assertEqual(len(round_teams), len(set(round_teams)))
            self.assertEqual([4, 4, 2], [len(field) for field in round_plan])

    def test_every_move_delta_matches_the_new_energy(self):
        for num_teams, field_sizes in ((19, [4, 4, 4]), (14, [4, 3, 2, 2])):
            optimizer = make_optimizer(num_teams, field_sizes)
            state = make_state(num_teams, field_sizes)
            rng = random.Random(3)
            applied = Counter()
            for _ in range(3_000):
                kind, move = self._random_move(optimizer, state, rng)
                if move is None:
                    continue
                delta = getattr(state, kind + "_delta")(*move)
                before = state.compute_energy()
                getattr(state, kind)(*move)
                self.assertAlmostEqual(before + delta, state.compute_energy())
                applied[kind] += 1
            self.assert_state_matches_plan(state)
            self.assertEqual({"swap", "cross", "trade", "field_swap"}, set(applied))

    def test_trade_keeps_field_counts_and_games(self):
        optimizer = make_optimizer(18, [4, 4, 4])
        state = make_state(18, [4, 4, 4])
        rng = random.Random(5)
        applied = 0
        for _ in range(2_000):
            move = optimizer._pick_trade(state, rng.random)
            if move is None:
                continue
            field_counts = list(state.field_counts)
            games = Counter(flatten(state.plan))
            state.trade(*move)
            self.assertEqual(field_counts, state.field_counts)
            self.assertEqual(games, Counter(flatten(state.plan)))
            applied += 1
        self.assertGreater(applied, 100)

    def test_field_swap_keeps_every_duel(self):
        state = make_state(16, [4, 4, 4, 4])
        pair_counts = list(state.pair_counts)

        state.field_swap(2, 0, 3)

        self.assertEqual(pair_counts, state.pair_counts)
        self.assert_state_matches_plan(state)

    def test_cross_keeps_every_team_at_its_games(self):
        optimizer = make_optimizer(22, [4, 4, 4, 4])
        state = make_state(22, [4, 4, 4, 4])
        rng = random.Random(9)
        games = Counter(flatten(state.plan))
        applied = 0
        for _ in range(500):
            move = optimizer._pick_cross(
                state, rng.random, rng.randrange(8), rng.randrange(4), rng.randrange(4)
            )
            if move is None:
                continue
            state.cross(*move)
            applied += 1
        self.assertGreater(applied, 50)
        self.assertEqual(games, Counter(flatten(state.plan)))
        for round_plan in state.plan:
            round_teams = flatten(round_plan)
            self.assertEqual(len(round_teams), len(set(round_teams)))
        self.assert_state_matches_plan(state)

    def test_energy_lower_bound(self):
        # 32 teams fill every slot four times: nothing needs to repeat.
        optimizer = make_optimizer(32, [4, 4, 4, 4])
        self.assertEqual(0, optimizer._energy_lower_bound([4] * 32, 1))

        # 20 teams: 8 of them play 7 games (21 opponent slots vs 19
        # opponents), which forces at least 8 repeated duels.
        optimizer = make_optimizer(20, [4, 4, 4, 4])
        games = [7] * 8 + [6] * 12
        self.assertEqual(8, optimizer._energy_lower_bound(games, 1))

        # 20 teams but only 16 seats on the small field in 8 rounds.
        optimizer = make_optimizer(20, [4, 4, 2])
        self.assertEqual(1.0, optimizer._energy_lower_bound([4] * 20, 1))

    def test_plans_never_go_below_the_lower_bound(self):
        for num_teams, field_sizes in ((16, [4, 4, 4, 4]), (13, [4, 4, 2])):
            optimizer = make_optimizer(num_teams, field_sizes, iterations=30_000)
            plan = optimizer.create_plan()
            state = recomputed(PlanState(plan, num_teams, field_sizes, [1] * num_teams))
            games = Counter(flatten(plan))
            bound = optimizer._energy_lower_bound(
                [games[team] for team in range(num_teams)], state.free
            )
            self.assertGreaterEqual(state.energy, bound)

    def test_sitting_teams_are_tracked(self):
        state = make_state(22, [4, 4, 4, 4])

        for round_idx, waiting in enumerate(state.sitting):
            self.assertEqual(6, len(waiting))
            for team in waiting:
                self.assertEqual(SITTING, state.field_of[round_idx][team])

    @staticmethod
    def _random_move(optimizer, state, rng):
        choice = rng.random()
        if choice < 0.25:
            if not optimizer._equal_size_fields:
                return "field_swap", None
            field_a, field_b = rng.choice(optimizer._equal_size_fields)
            return "field_swap", (rng.randrange(state.num_rounds), field_a, field_b)
        if choice < 0.5:
            return "trade", optimizer._pick_trade(state, rng.random)
        round_a = rng.randrange(state.num_rounds)
        field_a, idx_a = rng.choice(optimizer._slots)
        if choice < 0.75:
            return "cross", optimizer._pick_cross(
                state, rng.random, round_a, field_a, idx_a
            )
        field_b, idx_b = rng.choice(optimizer._other_slots[field_a])
        return "swap", (round_a, field_a, idx_a, field_b, idx_b)


if __name__ == "__main__":
    unittest.main()
