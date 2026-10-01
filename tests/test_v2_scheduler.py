import random
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, "src")

from schedulers.v2.benchmark import collect_metrics
from schedulers.v2.config import resolve_seed
from schedulers.v2.generate_gameplay_lists import (
    DEFAULT_MAX_SECONDS,
    create_plan as wrapper_create_plan,
    create_team_list,
    generate_plan,
    get_unrated_games,
)
from schedulers.v2.optimizer import ScheduleOptimizer
from schedulers.v2.plan_state import PlanState
from schedulers.v2.scheduler import create_plan
from schedulers.v2.utils import flatten


def make_optimizer(num_teams, num_fields):
    return ScheduleOptimizer(
        teams=create_team_list(num_teams),
        number_fields=num_fields,
        number_rounds=8,
        teams_per_game=4,
        seed=12345,
        max_seconds=0.15,
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


class SchedulerV2Tests(unittest.TestCase):
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
            max_seconds=0.15,
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
            max_seconds=0.15,
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
            max_seconds=0.15,
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
            max_seconds=0.15,
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
            max_seconds=0.15,
        )
        direct_plan = create_plan(
            create_team_list(16),
            number_fields=4,
            number_rounds=8,
            teams_per_game=4,
            seed=77,
            max_seconds=0.15,
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
                max_seconds=1.0,
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
            swaps = 0
            for _ in range(3000):
                pos_a, pos_b = rng.choice(positions), rng.choice(positions)
                if not state.swap_is_valid(pos_a, pos_b, allowed_duplicates):
                    continue

                before = state.score()
                state.swap(pos_a, pos_b)
                swapped = [
                    [[optimizer.teams[idx] for idx in field] for field in round_plan]
                    for round_plan in state.plan
                ]
                self.assertEqual(optimizer.score_plan(swapped), state.score())

                if rng.random() < 0.4:
                    state.swap(pos_a, pos_b)
                    self.assertEqual(before, state.score())
                swaps += 1
            self.assertGreater(swaps, 100)

    def test_local_improve_keeps_plan_valid_and_never_worsens_score(self):
        optimizer = make_optimizer(22, 4)
        plan = build_candidate(optimizer)
        score = optimizer.score_plan(plan)

        improved = optimizer.local_improve(
            plan, random.Random(3), time.monotonic() + 0.2
        )

        self.assertLessEqual(optimizer.score_plan(improved), score)
        metrics = collect_metrics(improved, 22, 4, 8)
        self.assertTrue(metrics["shape_ok"])
        self.assertEqual(0, metrics["same-round duplicates"])
        self.assertEqual(sorted(flatten(plan)), sorted(flatten(improved)))

    def test_ideal_score_bounds(self):
        # 32 teams fill every slot four times: nothing needs to repeat.
        ideal = make_optimizer(32, 4)._ideal_score
        self.assertEqual((0, 0, 1), ideal[:3])
        self.assertEqual((0, 0, 0), (ideal[5], ideal[6], ideal[10]))
        self.assertEqual((1, -4, 0), (ideal[7], ideal[8], ideal[9]))

        # 20 teams: 8 of them play 7 games (21 opponent slots vs 19 opponents),
        # which forces at least 8 repeated pairs, more than pair counting gives.
        self.assertEqual(8, make_optimizer(20, 4)._ideal_score[5])

    def test_tiny_fixed_shape_is_not_supported(self):
        teams = create_team_list(3)
        
        with self.assertRaises(RuntimeError):
            create_plan(
                teams,
                number_fields=4,
                number_rounds=8,
                teams_per_game=4,
                max_seconds=0.05,
            )


if __name__ == "__main__":
    unittest.main()
