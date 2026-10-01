import random
import sys
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, "src")

from schedulers.v3.benchmark import collect_metrics
from schedulers.v3.config import resolve_seed
from schedulers.v3.generate_gameplay_lists import (
    DEFAULT_MAX_SECONDS,
    create_plan as wrapper_create_plan,
    create_team_list,
    generate_plan,
    get_unrated_games,
)
from schedulers.v3 import optimizer as optimizer_module
from schedulers.v3.optimizer import ScheduleOptimizer
from schedulers.v3.plan_state import PlanState
from schedulers.v3.scheduler import create_plan
from schedulers.v3.utils import flatten


# Tests stop on iterations instead of time so a loaded machine cannot change
# the outcome.
TEST_ITERATIONS = 30_000


def make_optimizer(num_teams, num_fields):
    return ScheduleOptimizer(
        teams=create_team_list(num_teams),
        number_fields=num_fields,
        number_rounds=8,
        teams_per_game=4,
        seed=12345,
        max_seconds=5.0,
        max_iterations=TEST_ITERATIONS,
    )


def build_candidate(optimizer):
    seed = 0
    while True:
        plan = optimizer._build_candidate(
            random.Random(seed), np.random.default_rng(seed)
        )
        if plan is not None:
            return plan
        seed += 1


class SchedulerV3Tests(unittest.TestCase):
    def assert_valid_metrics(
        self,
        metrics,
        num_teams,
        num_fields,
        num_rounds,
        max_duel_repeat=None,
    ):
        self.assertTrue(metrics["shape_ok"])
        self.assertEqual([], metrics["missing switch-1 teams"])
        self.assertEqual(1, len(metrics["rating-count set"]))

        play_counts = metrics["play-count set"]
        self.assertLessEqual(max(play_counts) - min(play_counts), 1)

        capacity_per_round = num_fields * 4
        expected_duplicates = max(0, capacity_per_round - num_teams) * num_rounds
        self.assertEqual(expected_duplicates, metrics["same-round duplicates"])

        if max_duel_repeat is not None:
            self.assertLessEqual(metrics["max duel repeat"], max_duel_repeat)

    def test_main_tournament_shape_and_quality(self):
        plan = create_plan(
            create_team_list(20),
            number_fields=4,
            number_rounds=8,
            teams_per_game=4,
            seed=12345,
            max_iterations=TEST_ITERATIONS,
        )

        metrics = collect_metrics(plan, 20, 4, 8)
        self.assert_valid_metrics(metrics, 20, 4, 8, max_duel_repeat=2)

    def test_small_tournament_with_one_field(self):
        plan = create_plan(
            create_team_list(14),
            number_fields=1,
            number_rounds=8,
            teams_per_game=4,
            seed=12345,
            max_iterations=TEST_ITERATIONS,
        )

        metrics = collect_metrics(plan, 14, 1, 8)
        self.assert_valid_metrics(metrics, 14, 1, 8, max_duel_repeat=1)

    def test_small_tournament_duplicate_slots_are_explicit(self):
        plan = create_plan(
            create_team_list(10),
            number_fields=3,
            number_rounds=8,
            teams_per_game=4,
            seed=12345,
            max_iterations=TEST_ITERATIONS,
        )

        metrics = collect_metrics(plan, 10, 3, 8)
        self.assert_valid_metrics(metrics, 10, 3, 8)
        self.assertEqual(16, metrics["same-round duplicates"])

    def test_rating_plan_marks_equal_rated_game_counts(self):
        plan = create_plan(
            create_team_list(25),
            number_fields=4,
            number_rounds=8,
            teams_per_game=4,
            seed=12345,
            max_iterations=TEST_ITERATIONS,
        )

        rate_plan = get_unrated_games(plan)
        metrics = collect_metrics(plan, 25, 4, 8)
        self.assertEqual([5], metrics["rating-count set"])
        self.assertEqual(len(plan), len(rate_plan))
        self.assertEqual(len(plan[0]), len(rate_plan[0]))
        self.assertEqual(len(plan[0][0]), len(rate_plan[0][0]))

    def test_generate_plan_keeps_legacy_return_shape(self):
        plan, max_games_count = generate_plan(num_teams=18, num_fields=4, num_rounds=8)

        metrics = collect_metrics(plan, 18, 4, 8)
        self.assert_valid_metrics(metrics, 18, 4, 8, max_duel_repeat=2)
        self.assertEqual({7}, max_games_count)

    def test_wrapper_create_plan_matches_scheduler_api(self):
        wrapper_plan = wrapper_create_plan(
            create_team_list(16),
            number_fields=4,
            number_rounds=8,
            teams_per_game=4,
            seed=77,
            max_iterations=TEST_ITERATIONS,
        )
        direct_plan = create_plan(
            create_team_list(16),
            number_fields=4,
            number_rounds=8,
            teams_per_game=4,
            seed=77,
            max_iterations=TEST_ITERATIONS,
        )

        self.assertEqual(direct_plan, wrapper_plan)

    def test_resolve_seed_is_deterministic_without_curated_table(self):
        self.assertEqual(204988, resolve_seed(None, 20, 4, 8, 4))
        self.assertEqual(42, resolve_seed(42, 20, 4, 8, 4))

    def test_invalid_inputs_raise_value_error(self):
        teams_5 = create_team_list(5)

        with self.assertRaises(ValueError):
            create_plan(teams_5, number_fields=0)
        with self.assertRaises(ValueError):
            create_plan(teams_5, number_rounds=0)
        with self.assertRaises(ValueError):
            create_plan(teams_5, teams_per_game=0)
        teams_33 = create_team_list(33)
        with self.assertRaises(ValueError):
            create_plan(teams_33, number_fields=4, number_rounds=2)

    def test_empty_team_list_returns_empty_plan(self):
        self.assertEqual([], create_plan([]))

    def test_large_field_has_no_repeated_duels(self):
        for seed in (12345, 777):
            plan = create_plan(
                create_team_list(28),
                number_fields=4,
                number_rounds=8,
                teams_per_game=4,
                seed=seed,
            )

            metrics = collect_metrics(plan, 28, 4, 8)
            self.assert_valid_metrics(metrics, 28, 4, 8, max_duel_repeat=1)

    def test_incremental_score_matches_full_score(self):
        for num_teams, fields in ((25, 4), (14, 4), (10, 3)):
            optimizer = make_optimizer(num_teams, fields)
            plan = build_candidate(optimizer)
            state = PlanState(optimizer.team_order, plan)
            self.assertEqual(optimizer.score_plan(plan), state.score())

            rng = random.Random(1)
            positions = [
                (round_idx, field_idx, team_idx)
                for round_idx in range(8)
                for field_idx in range(fields)
                for team_idx in range(4)
            ]
            allowed_duplicates = max(0, fields * 4 - num_teams)
            moves = 0
            for _ in range(3000):
                if rng.random() < 0.2:
                    field_a, field_b = rng.sample(range(fields), 2)
                    move = (state.swap_fields, (rng.randrange(8), field_a, field_b))
                else:
                    pos_a, pos_b = rng.choice(positions), rng.choice(positions)
                    if not state.swap_is_valid(pos_a, pos_b, allowed_duplicates):
                        continue
                    move = (state.swap, (pos_a, pos_b))

                before = state.score(), state.energy
                move[0](*move[1])
                moved = [
                    [[optimizer.teams[idx] for idx in field] for field in round_plan]
                    for round_plan in state.plan
                ]
                self.assertEqual(optimizer.score_plan(moved), state.score())
                self.assertEqual(
                    PlanState(optimizer.team_order, moved).energy, state.energy
                )

                if rng.random() < 0.4:
                    move[0](*move[1])
                    self.assertEqual(before, (state.score(), state.energy))
                moves += 1
            self.assertGreater(moves, 100)

    def test_swap_fields_keeps_every_duel(self):
        optimizer = make_optimizer(20, 4)
        state = PlanState(optimizer.team_order, build_candidate(optimizer))
        pair_counts = list(state.pair_counts)
        fields_before = [set(field) for field in state.plan[2]]

        state.swap_fields(2, 0, 3)

        self.assertEqual(pair_counts, state.pair_counts)
        self.assertEqual(fields_before[0], set(state.plan[2][3]))
        self.assertEqual(fields_before[3], set(state.plan[2][0]))

    def test_same_seed_gives_same_plan(self):
        plans = [
            create_plan(create_team_list(22), seed=7, max_iterations=TEST_ITERATIONS)
            for _ in range(3)
        ]

        self.assertEqual(plans[0], plans[1])
        self.assertEqual(plans[0], plans[2])

    def test_time_budget_cuts_the_search_short(self):
        started_at = time.monotonic()
        plan = create_plan(
            create_team_list(25),
            seed=7,
            max_seconds=0.2,
            max_iterations=50_000_000,
        )

        self.assertLess(time.monotonic() - started_at, 2.0)
        self.assert_valid_metrics(collect_metrics(plan, 25, 4, 8), 25, 4, 8)

    def test_restarts_keep_the_best_plan(self):
        teams = create_team_list(20)
        with mock.patch.object(optimizer_module, "RESTARTS", 1):
            single = ScheduleOptimizer(teams, 4, 8, 4, 12345, 5.0, 10_000)
            single_score = single.score_plan(single.create_plan())
        double = ScheduleOptimizer(teams, 4, 8, 4, 12345, 5.0, 20_000)

        self.assertLessEqual(double.score_plan(double.create_plan()), single_score)

    def test_local_improve_keeps_plan_valid_and_never_worsens_score(self):
        optimizer = make_optimizer(22, 4)
        plan = build_candidate(optimizer)
        score = optimizer.score_plan(plan)

        improved = optimizer.local_improve(
            plan, random.Random(3), time.monotonic() + 5.0
        )

        self.assertLessEqual(optimizer.score_plan(improved), score)
        metrics = collect_metrics(improved, 22, 4, 8)
        self.assertTrue(metrics["shape_ok"])
        self.assertEqual(0, metrics["same-round duplicates"])
        self.assertEqual(sorted(flatten(plan)), sorted(flatten(improved)))

    def test_every_team_uses_the_fields_evenly(self):
        cases = ((8, 2), (13, 2), (12, 3), (20, 3), (16, 4), (19, 4), (25, 4))
        for num_teams, fields in cases:
            plan = create_plan(create_team_list(num_teams), number_fields=fields)

            field_counts = {team: [0] * fields for team in range(num_teams)}
            for round_plan in plan:
                for field_idx, field_plan in enumerate(round_plan):
                    for team in field_plan:
                        field_counts[team][field_idx] += 1
            for team, counts in field_counts.items():
                self.assertLessEqual(
                    max(counts) - min(counts), 1, (num_teams, fields, team, counts)
                )

    def test_two_full_fields_spread_the_duels(self):
        # 8 teams on 2 fields all play every round and must sit 4 times on each
        # field, so only double swaps can change duels without breaking that.
        plan = create_plan(create_team_list(8), number_fields=2)

        metrics = collect_metrics(plan, 8, 2, 8)
        self.assert_valid_metrics(metrics, 8, 2, 8, max_duel_repeat=4)

    def test_double_swap_keeps_field_counts(self):
        optimizer = make_optimizer(16, 4)
        state = PlanState(optimizer.team_order, build_candidate(optimizer))
        rng = random.Random(5)
        applied = 0
        for _ in range(2000):
            move = optimizer._pick_double_swap(
                state, rng, optimizer._random_position(rng)
            )
            if move is None:
                continue

            field_counts = [list(counts) for counts in state.field_counts]
            pair_counts = list(state.pair_counts)
            optimizer._apply_move(state, move)
            self.assertEqual(field_counts, state.field_counts)
            self.assertNotEqual(pair_counts, state.pair_counts)
            self.assertEqual(0, state.duplicate_total)
            applied += 1
        self.assertGreater(applied, 20)

    def test_field_imbalance_counts_games_outside_the_even_range(self):
        optimizer = make_optimizer(4, 2)
        even = [[[0, 1, 2, 3], [0, 1, 2, 3]] for _ in range(8)]
        self.assertEqual(0, optimizer.score_plan(even)[2])

        # Teams 0 and 4 play all 8 games on one field instead of 4 on each:
        # 4 games too many there and 4 too few on the other, for both teams.
        optimizer = make_optimizer(5, 2)
        uneven = [[[0, 1, 2, 3], [4, 1, 2, 3]] for _ in range(8)]
        self.assertEqual(16, optimizer.score_plan(uneven)[2])
        self.assertEqual(16, PlanState(optimizer.team_order, uneven).score()[2])

    def test_ideal_score_bounds(self):
        # 32 teams fill every slot four times: nothing needs to repeat.
        ideal = make_optimizer(32, 4)._ideal_score
        self.assertEqual((0, 0, 0, 1), ideal[:4])
        self.assertEqual((0, 0, 0), (ideal[6], ideal[7], ideal[11]))
        self.assertEqual((1, -4, 0), (ideal[8], ideal[9], ideal[10]))

        # 20 teams: 8 of them play 7 games (21 opponent slots vs 19 opponents),
        # which forces at least 8 repeated pairs, more than pair counting gives.
        self.assertEqual(8, make_optimizer(20, 4)._ideal_score[6])

        # 21 teams: 192 duels fit into 210 pairs, but the two teams with 7 games
        # still have to meet somebody twice.
        self.assertEqual(2, make_optimizer(21, 4)._ideal_score[3])

        # 40 teams on 8 fields: Main Switch only has 32 slots.
        self.assertEqual(8, make_optimizer(40, 8)._ideal_score[1])

    def test_fewer_teams_than_a_game_needs_raise_value_error(self):
        with self.assertRaises(ValueError):
            create_plan(create_team_list(3), teams_per_game=4)

    def test_one_team_short_of_a_full_round(self):
        plan = create_plan(create_team_list(15))

        metrics = collect_metrics(plan, 15, 4, 8)
        self.assert_valid_metrics(metrics, 15, 4, 8, max_duel_repeat=3)
        self.assertEqual(8, metrics["same-round duplicates"])


if __name__ == "__main__":
    unittest.main()
