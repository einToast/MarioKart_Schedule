import sys
import unittest

sys.path.insert(0, "src")

from schedulers.v2.benchmark import collect_metrics
from schedulers.v2.config import resolve_seed
from schedulers.v2.generate_gameplay_lists import (
    DEFAULT_MAX_SECONDS,
    create_plan as wrapper_create_plan,
    create_team_liste,
    generate_plan,
    get_unrated_games,
)
from schedulers.v2.scheduler import create_plan


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
            create_team_liste(20),
            number_fields=4,
            number_rounds=8,
            seed=12345,
            max_seconds=0.15,
        )

        metrics = collect_metrics(plan, 20, 4, 8)
        self.assert_valid_metrics(metrics, 20, 4, 8, max_duel_repeat=2)

    def test_small_tournament_with_one_field(self):
        plan = create_plan(
            create_team_liste(14),
            number_fields=1,
            number_rounds=8,
            seed=12345,
            max_seconds=0.15,
        )

        metrics = collect_metrics(plan, 14, 1, 8)
        self.assert_valid_metrics(metrics, 14, 1, 8, max_duel_repeat=1)

    def test_small_tournament_duplicate_slots_are_explicit(self):
        plan = create_plan(
            create_team_liste(10),
            number_fields=3,
            number_rounds=8,
            seed=12345,
            max_seconds=0.15,
        )

        metrics = collect_metrics(plan, 10, 3, 8)
        self.assert_valid_metrics(metrics, 10, 3, 8)
        self.assertEqual(16, metrics["same-round duplicates"])

    def test_rating_plan_marks_equal_rated_game_counts(self):
        plan = create_plan(
            create_team_liste(25),
            number_fields=4,
            number_rounds=8,
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
            create_team_liste(16),
            number_fields=4,
            number_rounds=8,
            seed=77,
            max_seconds=0.15,
        )
        direct_plan = create_plan(
            create_team_liste(16),
            number_fields=4,
            number_rounds=8,
            seed=77,
            max_seconds=0.15,
        )

        self.assertEqual(direct_plan, wrapper_plan)

    def test_resolve_seed_is_deterministic_without_curated_table(self):
        self.assertEqual(204988, resolve_seed(None, 20, 4, 8, 4))
        self.assertEqual(42, resolve_seed(42, 20, 4, 8, 4))

    def test_invalid_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            create_plan(create_team_liste(5), number_fields=0)
        with self.assertRaises(ValueError):
            create_plan(create_team_liste(5), number_rounds=0)
        with self.assertRaises(ValueError):
            create_plan(create_team_liste(5), teams_per_game=0)
        with self.assertRaises(ValueError):
            create_plan(create_team_liste(33), number_fields=4, number_rounds=2)

    def test_empty_team_list_returns_empty_plan(self):
        self.assertEqual([], create_plan([]))

    def test_tiny_fixed_shape_is_not_supported(self):
        with self.assertRaises(RuntimeError):
            create_plan(
                create_team_liste(3),
                number_fields=4,
                number_rounds=8,
                max_seconds=0.05,
            )


if __name__ == "__main__":
    unittest.main()
