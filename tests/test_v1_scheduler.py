import contextlib
import io
import sys
import unittest
from collections import Counter

sys.path.insert(0, "src")

from schedulers.v1.generate_gameplay_lists import (
    generate_plan,
    get_unrated_games,
)
from schedulers.v1.helper import (
    count_in_list,
    create_filled_df,
    flatten,
)


class SchedulerV1Tests(unittest.TestCase):
    def test_generate_plan_returns_complete_balanced_schedule(self):
        with contextlib.redirect_stdout(io.StringIO()):
            plan, max_games_count = generate_plan(
                num_teams=16,
                num_fields=4,
                num_rounds=8,
            )

        self.assertEqual(8, len(plan))
        self.assertEqual({8}, max_games_count)

        expected_teams = set(range(16))
        for round_plan in plan:
            self.assertEqual(4, len(round_plan))
            self.assertTrue(all(len(field_plan) == 4 for field_plan in round_plan))
            self.assertEqual(expected_teams, set(flatten(round_plan)))

        main_field_teams = {team for round_plan in plan for team in round_plan[0]}
        self.assertEqual(expected_teams, main_field_teams)

    def test_unrated_games_gives_each_team_one_rating_game(self):
        plan = [
            [[0, 1, 2, 3]],
            [[0, 1, 2, 4]],
        ]

        rate_plan = get_unrated_games(plan)
        rated_counts = Counter()
        for round_idx, round_plan in enumerate(plan):
            for field_idx, field_plan in enumerate(round_plan):
                for team_idx, team in enumerate(field_plan):
                    if rate_plan[round_idx][field_idx][team_idx]:
                        rated_counts[team] += 1

        self.assertEqual({0, 1, 2, 3, 4}, set(rated_counts))
        self.assertEqual({1}, set(rated_counts.values()))

    def test_helpers_flatten_count_and_track_duels(self):
        plan = [
            [[0, 1, 2, 3]],
            [[0, 1, 2, 4]],
        ]

        self.assertEqual([0, 1, 2, 3, 0, 1, 2, 4], flatten(plan))
        self.assertEqual(2, count_in_list(plan, 0))
        self.assertEqual(1, count_in_list(plan, 4))

        duels = create_filled_df(range(5), plan)
        duel_0_1 = duels[(duels["team_1"] == 0) & (duels["team_2"] == 1)]
        duel_3_4 = duels[(duels["team_1"] == 3) & (duels["team_2"] == 4)]
        self.assertEqual(2, duel_0_1["num_games"].iloc[0])
        self.assertEqual(0, duel_3_4["num_games"].iloc[0])


if __name__ == "__main__":
    unittest.main()
