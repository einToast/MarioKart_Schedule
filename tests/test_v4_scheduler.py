import random
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, "src")

from schedulers.v4.benchmark import collect_metrics
from schedulers.v4.config import resolve_seed
from schedulers.v4.generate_gameplay_lists import (
    create_plan as wrapper_create_plan,
    create_team_list,
    generate_plan,
    get_unrated_games,
)
from schedulers.v4.optimizer import (
    DOUBLE_SWAP,
    FIELD_SWAP,
    SLOT_SWAP,
    ScheduleOptimizer,
)
from schedulers.v4.plan_state import PlanState
from schedulers.v4.scheduler import create_plan
from schedulers.v4.utils import field_share_bounds, flatten, resolve_field_sizes


# Tests stop on iterations instead of time so a loaded machine cannot change
# the outcome.
TEST_ITERATIONS = 30_000


def make_optimizer(num_teams, field_sizes):
    return ScheduleOptimizer(
        teams=create_team_list(num_teams),
        field_sizes=field_sizes,
        number_rounds=8,
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


class SchedulerV4Tests(unittest.TestCase):
    def assert_valid_metrics(
        self,
        metrics,
        num_teams,
        field_sizes,
        num_rounds,
        max_duel_repeat=None,
    ):
        self.assertTrue(metrics["shape_ok"])
        self.assertEqual([], metrics["missing switch-1 teams"])
        self.assertEqual(1, len(metrics["rating-count set"]))

        play_counts = metrics["play-count set"]
        self.assertLessEqual(max(play_counts) - min(play_counts), 1)

        expected_duplicates = max(0, sum(field_sizes) - num_teams) * num_rounds
        self.assertEqual(expected_duplicates, metrics["same-round duplicates"])

        if max_duel_repeat is not None:
            self.assertLessEqual(metrics["max duel repeat"], max_duel_repeat)

    def test_same_size_on_every_field_shape_and_quality(self):
        plan = create_plan(
            create_team_list(20),
            number_fields=4,
            number_rounds=8,
            teams_per_game=4,
            seed=12345,
            max_iterations=TEST_ITERATIONS,
        )

        metrics = collect_metrics(plan, 20, 4, 8)
        self.assert_valid_metrics(metrics, 20, [4, 4, 4, 4], 8, max_duel_repeat=2)

    def test_number_and_list_of_equal_sizes_give_the_same_plan(self):
        plans = [
            create_plan(
                create_team_list(18),
                number_fields=3,
                teams_per_game=teams_per_game,
                max_iterations=TEST_ITERATIONS,
            )
            for teams_per_game in (4, [4, 4, 4], (4, 4, 4))
        ]

        self.assertEqual(plans[0], plans[1])
        self.assertEqual(plans[0], plans[2])

    def test_every_field_gets_its_own_number_of_teams(self):
        cases = (
            (20, [4, 4, 2]),
            (16, [4, 3, 2, 2]),
            (12, [2, 4]),
            (25, [4, 4, 4, 2]),
            (30, [8, 4, 2, 2]),
            (9, [6, 2, 1]),
        )
        for num_teams, field_sizes in cases:
            plan = create_plan(
                create_team_list(num_teams),
                number_fields=len(field_sizes),
                teams_per_game=field_sizes,
                max_iterations=TEST_ITERATIONS,
            )

            self.assertEqual(8, len(plan))
            for round_plan in plan:
                self.assertEqual(field_sizes, [len(field) for field in round_plan])
                for field_plan in round_plan:
                    self.assertEqual(len(field_plan), len(set(field_plan)))
            metrics = collect_metrics(plan, num_teams, len(field_sizes), 8, field_sizes)
            self.assert_valid_metrics(metrics, num_teams, field_sizes, 8)

    def test_mixed_sizes_with_enough_teams_have_no_repeated_duels(self):
        for num_teams, field_sizes in ((20, [4, 4, 2]), (16, [4, 3, 2, 2])):
            plan = create_plan(
                create_team_list(num_teams),
                number_fields=len(field_sizes),
                teams_per_game=field_sizes,
            )

            metrics = collect_metrics(plan, num_teams, len(field_sizes), 8, field_sizes)
            self.assert_valid_metrics(
                metrics, num_teams, field_sizes, 8, max_duel_repeat=1
            )

    def test_teams_use_the_fields_in_proportion_to_their_size(self):
        cases = ((20, [4, 4, 2]), (6, [4, 2]), (25, [4, 4, 4, 2]), (10, [4, 4, 2]))
        for num_teams, field_sizes in cases:
            plan = create_plan(
                create_team_list(num_teams),
                number_fields=len(field_sizes),
                teams_per_game=field_sizes,
            )

            games = [0] * num_teams
            field_counts = [[0] * len(field_sizes) for _ in range(num_teams)]
            for round_plan in plan:
                for field_idx, field_plan in enumerate(round_plan):
                    for team in field_plan:
                        games[team] += 1
                        field_counts[team][field_idx] += 1
            for team, counts in enumerate(field_counts):
                low, high = field_share_bounds(games[team], field_sizes)
                for field_idx, count in enumerate(counts):
                    self.assertTrue(
                        low[field_idx] <= count <= high[field_idx],
                        (num_teams, field_sizes, team, counts),
                    )

    def test_field_share_bounds(self):
        # 8 games on fields with 4 and 2 seats: 5.33 and 2.67 games.
        self.assertEqual(([5, 2], [6, 3]), field_share_bounds(8, [4, 2]))
        self.assertEqual(([2, 2, 2], [2, 2, 2]), field_share_bounds(6, [4, 4, 4]))
        self.assertEqual(([1, 1], [2, 2]), field_share_bounds(3, [4, 4]))

    def test_resolve_field_sizes(self):
        self.assertEqual([4, 4, 4], resolve_field_sizes(3, 4))
        self.assertEqual([4, 2], resolve_field_sizes(2, [4, 2]))
        for number_fields, teams_per_game in (
            (0, 4),
            (2, 0),
            (2, [4]),
            (2, [4, 2, 2]),
            (2, [4, 0]),
            (2, [4, -1]),
            (2, [4, 2.5]),
            (2, [4, "2"]),
            (2, [4, True]),
            (2, None),
        ):
            with self.assertRaises(ValueError):
                resolve_field_sizes(number_fields, teams_per_game)

    def test_small_tournament_duplicate_slots_are_explicit(self):
        plan = create_plan(
            create_team_list(10),
            number_fields=3,
            teams_per_game=[4, 4, 4],
            seed=12345,
            max_iterations=TEST_ITERATIONS,
        )

        metrics = collect_metrics(plan, 10, 3, 8)
        self.assert_valid_metrics(metrics, 10, [4, 4, 4], 8)
        self.assertEqual(16, metrics["same-round duplicates"])

    def test_rating_plan_marks_equal_rated_game_counts(self):
        field_sizes = [4, 4, 2]
        plan = create_plan(
            create_team_list(17),
            number_fields=3,
            teams_per_game=field_sizes,
            seed=12345,
            max_iterations=TEST_ITERATIONS,
        )

        rate_plan = get_unrated_games(plan)
        metrics = collect_metrics(plan, 17, 3, 8, field_sizes)
        self.assertEqual([4], metrics["rating-count set"])
        self.assertEqual(field_sizes, [len(field) for field in rate_plan[0]])

    def test_generate_plan_keeps_legacy_return_shape(self):
        plan, max_games_count = generate_plan(num_teams=18, num_fields=4, num_rounds=8)

        metrics = collect_metrics(plan, 18, 4, 8)
        self.assert_valid_metrics(metrics, 18, [4, 4, 4, 4], 8, max_duel_repeat=2)
        self.assertEqual({7}, max_games_count)

    def test_generate_plan_accepts_a_size_per_field(self):
        plan, max_games_count = generate_plan(
            num_teams=20, num_fields=3, num_rounds=8, num_teams_per_game=[4, 4, 2]
        )

        self.assertEqual([4, 4, 2], [len(field) for field in plan[0]])
        self.assertEqual({4}, max_games_count)

    def test_wrapper_create_plan_matches_scheduler_api(self):
        arguments = {
            "number_fields": 3,
            "teams_per_game": [4, 2, 2],
            "seed": 77,
            "max_iterations": TEST_ITERATIONS,
        }

        self.assertEqual(
            create_plan(create_team_list(16), **arguments),
            wrapper_create_plan(create_team_list(16), **arguments),
        )

    def test_resolve_seed_depends_on_the_field_sizes(self):
        self.assertEqual(42, resolve_seed(42, 20, 8, [4, 4]))
        self.assertEqual(203_092, resolve_seed(None, 20, 8, [4, 2]))
        self.assertNotEqual(
            resolve_seed(None, 20, 8, [4, 2]), resolve_seed(None, 20, 8, [2, 4])
        )

    def test_invalid_inputs_raise_value_error(self):
        teams_5 = create_team_list(5)

        with self.assertRaises(ValueError):
            create_plan(teams_5, number_fields=0)
        with self.assertRaises(ValueError):
            create_plan(teams_5, number_rounds=0)
        with self.assertRaises(ValueError):
            create_plan(teams_5, teams_per_game=0)
        with self.assertRaises(ValueError):
            create_plan(teams_5, number_fields=2, teams_per_game=[4, 0])
        with self.assertRaises(ValueError):
            create_plan(teams_5, number_fields=3, teams_per_game=[4, 2])
        teams_33 = create_team_list(33)
        with self.assertRaises(ValueError):
            create_plan(teams_33, number_fields=4, number_rounds=2)
        teams_13 = create_team_list(13)
        with self.assertRaises(ValueError):
            create_plan(
                teams_13,
                number_fields=2,
                number_rounds=2,
                teams_per_game=[4, 2],
            )

    def test_fewer_teams_than_the_largest_game_raise_value_error(self):
        teams_3 = create_team_list(3)
        with self.assertRaises(ValueError):
            create_plan(teams_3, number_fields=2, teams_per_game=[2, 4])

    def test_empty_team_list_returns_empty_plan(self):
        self.assertEqual([], create_plan([]))

    def test_incremental_score_matches_full_score(self):
        cases = ((25, [4, 4, 4, 4]), (14, [4, 3, 2]), (10, [4, 4, 4]), (7, [6, 2, 2]))
        for num_teams, field_sizes in cases:
            optimizer = make_optimizer(num_teams, field_sizes)
            plan = build_candidate(optimizer)
            state = PlanState(optimizer.team_order, plan)
            self.assertEqual(optimizer.score_plan(plan), state.score())

            rng = random.Random(1)
            positions = [
                (round_idx, field_idx, team_idx)
                for round_idx in range(8)
                for field_idx, size in enumerate(field_sizes)
                for team_idx in range(size)
            ]
            allowed_duplicates = max(0, sum(field_sizes) - num_teams)
            moves = 0
            for _ in range(3000):
                if rng.random() < 0.2 and optimizer._equal_size_fields:
                    field_a, field_b = rng.choice(optimizer._equal_size_fields)
                    move = (state.swap_fields, (rng.randrange(8), field_a, field_b))
                else:
                    pos_a, pos_b = rng.choice(positions), rng.choice(positions)
                    if not state.swap_is_valid(pos_a, pos_b, allowed_duplicates):
                        continue
                    move = (state.swap, (pos_a, pos_b))

                before = state.score(), state.energy
                move[0](*move[1])
                moved = optimizer._to_teams(state.plan)
                self.assertEqual(optimizer.score_plan(moved), state.score())
                self.assertEqual(
                    PlanState(optimizer.team_order, moved).energy, state.energy
                )

                if rng.random() < 0.4:
                    move[0](*move[1])
                    after = state.score(), state.energy
                    self.assertEqual(before, after)
                moves += 1
            self.assertGreater(moves, 100)

    def test_moves_keep_every_field_at_its_size(self):
        field_sizes = [4, 2, 4, 3]
        optimizer = make_optimizer(18, field_sizes)
        state = PlanState(optimizer.team_order, build_candidate(optimizer))
        rng = random.Random(9)
        kinds = set()
        for _ in range(5000):
            move = optimizer._pick_move(state, rng, 0)
            if move is None:
                continue
            kind, args = move
            if kind == FIELD_SWAP:
                self.assertEqual(field_sizes[args[1]], field_sizes[args[2]])
            kinds.add(kind)
            optimizer._apply_move(state, move)

        self.assertEqual({FIELD_SWAP, SLOT_SWAP, DOUBLE_SWAP}, kinds)
        self.assertEqual(0, state.duplicate_total)
        for round_plan in state.plan:
            self.assertEqual(field_sizes, [len(field) for field in round_plan])

    def test_fields_of_different_sizes_are_never_swapped(self):
        optimizer = make_optimizer(12, [4, 2])
        state = PlanState(optimizer.team_order, build_candidate(optimizer))
        rng = random.Random(9)

        self.assertEqual([], optimizer._equal_size_fields)
        for _ in range(2000):
            move = optimizer._pick_move(state, rng, 0)
            self.assertTrue(move is None or move[0] != FIELD_SWAP)

    def test_double_swap_keeps_field_counts(self):
        optimizer = make_optimizer(16, [4, 4, 2, 2])
        state = PlanState(optimizer.team_order, build_candidate(optimizer))
        rng = random.Random(5)
        applied = 0
        for _ in range(4000):
            move = optimizer._pick_double_swap(
                state, rng, optimizer._random_position(rng)
            )
            if move is None:
                continue

            field_counts = [list(counts) for counts in state.field_counts]
            optimizer._apply_move(state, move)
            self.assertEqual(field_counts, state.field_counts)
            self.assertEqual(0, state.duplicate_total)
            applied += 1
        self.assertGreater(applied, 20)

    def test_same_seed_gives_same_plan(self):
        plans = [
            create_plan(
                create_team_list(22),
                number_fields=4,
                teams_per_game=[4, 4, 3, 2],
                seed=7,
                max_iterations=TEST_ITERATIONS,
            )
            for _ in range(3)
        ]

        self.assertEqual(plans[0], plans[1])
        self.assertEqual(plans[0], plans[2])

    def test_time_budget_cuts_the_search_short(self):
        field_sizes = [4, 4, 4, 2]
        started_at = time.monotonic()
        plan = create_plan(
            create_team_list(21),
            teams_per_game=field_sizes,
            seed=7,
            max_seconds=0.2,
            max_iterations=50_000_000,
        )

        self.assertLess(time.monotonic() - started_at, 2.0)
        self.assert_valid_metrics(
            collect_metrics(plan, 21, 4, 8, field_sizes), 21, field_sizes, 8
        )

    def test_local_improve_keeps_plan_valid_and_never_worsens_score(self):
        field_sizes = [4, 4, 2, 2]
        optimizer = make_optimizer(22, field_sizes)
        plan = build_candidate(optimizer)
        score = optimizer.score_plan(plan)

        improved = optimizer.local_improve(
            plan, random.Random(3), time.monotonic() + 5.0
        )

        self.assertLessEqual(optimizer.score_plan(improved), score)
        metrics = collect_metrics(improved, 22, 4, 8, field_sizes)
        self.assertTrue(metrics["shape_ok"])
        self.assertEqual(0, metrics["same-round duplicates"])
        self.assertEqual(sorted(flatten(plan)), sorted(flatten(improved)))

    def test_field_imbalance_counts_games_outside_the_field_share(self):
        # Field 0 has 4 of the 6 seats, so 8 games mean 5 or 6 on field 0 and
        # 2 or 3 on field 1.
        optimizer = make_optimizer(6, [4, 2])
        fair = [[[0, 1, 2, 3], [4, 5]]] * 3
        fair += [[[2, 3, 4, 5], [0, 1]]] * 3
        fair += [[[4, 5, 0, 1], [2, 3]]] * 2
        self.assertEqual(0, optimizer.score_plan(fair)[2])
        self.assertEqual(0, PlanState(optimizer.team_order, fair).score()[2])

        # Teams 0-3 never leave field 0 (2 games too many there, 2 too few on
        # field 1) and teams 4-5 never leave field 1 (5 too many, 5 too few).
        uneven = [[[0, 1, 2, 3], [4, 5]] for _ in range(8)]
        self.assertEqual(36, optimizer.score_plan(uneven)[2])
        self.assertEqual(36, PlanState(optimizer.team_order, uneven).score()[2])

    def test_ideal_score_bounds(self):
        # 32 teams fill every slot four times: nothing needs to repeat.
        ideal = make_optimizer(32, [4, 4, 4, 4])._ideal_score
        self.assertEqual((0, 0, 0, 1), ideal[:4])
        self.assertEqual((0, 0, 0), (ideal[6], ideal[7], ideal[11]))
        self.assertEqual((1, -4, 0), (ideal[8], ideal[9], ideal[10]))

        # 20 teams: 8 of them play 7 games (21 opponent slots vs 19 opponents),
        # which forces at least 8 repeated pairs, more than pair counting gives.
        self.assertEqual(8, make_optimizer(20, [4, 4, 4, 4])._ideal_score[6])

        # Main Switch has 2 seats in 8 rounds, too few for 20 teams.
        self.assertEqual(4, make_optimizer(20, [2, 4, 4])._ideal_score[1])

        # 6 teams on a field with 4 seats: 32 slots, so somebody sits there 6
        # times and 26 + 10 slots go to a team that has been there before.
        ideal = make_optimizer(6, [4, 2])._ideal_score
        self.assertEqual((6, 36), (ideal[8], ideal[10]))

        # 8 rounds of 6 + 1 duels fit into the 15 pairs four times.
        self.assertEqual(4, ideal[3])

    def test_plans_reach_the_ideal_score_bounds_or_stay_above(self):
        for num_teams, field_sizes in ((6, [4, 2]), (9, [6, 2, 1]), (16, [4, 3, 2, 2])):
            optimizer = make_optimizer(num_teams, field_sizes)
            score = optimizer.score_plan(optimizer.create_plan())
            for value, bound in zip(score, optimizer._ideal_score):
                if bound != float("inf"):
                    self.assertGreaterEqual(value, bound, (num_teams, field_sizes))


if __name__ == "__main__":
    unittest.main()
