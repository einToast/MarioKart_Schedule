import sys
import unittest

sys.path.insert(0, "src")

from webserver import app


class ScheduleApiVersionTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_schedule_defaults_to_v3(self):
        response = self.client.post("/schedule", json={"num_teams": 20})

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(3, payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        self.assertEqual(6, payload["max_games_count"])

    def test_schedule_allows_v1_to_v3(self):
        for version in (1, 2, 3):
            with self.subTest(version=version):
                response = self.client.post(
                    "/schedule",
                    json={"num_teams": 16, "version": version},
                )

                self.assertEqual(200, response.status_code)
                payload = response.get_json()
                self.assertEqual(version, payload["version"])
                self.assertEqual(8, len(payload["plan"]))
                self.assertEqual(4, len(payload["plan"][0]))
                self.assertEqual(8, payload["max_games_count"])

    def test_schedule_v2_uses_num_rounds(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 2, "num_rounds": 4},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(2, payload["version"])
        self.assertEqual(4, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        self.assertEqual(4, payload["max_games_count"])

    def test_schedule_v2_uses_num_fields(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 2, "num_fields": 2},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(2, payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(2, len(payload["plan"][0]))
        
    def test_schedule_v2_uses_num_teams_per_game(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 2, "num_teams_per_game": 2},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(2, payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        self.assertEqual(4, payload["max_games_count"])

    def test_schedule_v2_uses_num_rounds_and_num_fields(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 2, "num_rounds": 4, "num_fields": 2},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(2, payload["version"])
        self.assertEqual(4, len(payload["plan"]))
        self.assertEqual(2, len(payload["plan"][0]))
        
        
    def test_schedule_v2_uses_num_rounds_and_num_teams_per_game(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 2, "num_rounds": 4, "num_teams_per_game": 2},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(2, payload["version"])
        self.assertEqual(4, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        
    def test_schedule_v2_uses_num_fields_and_num_teams_per_game(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 2, "num_fields": 2, "num_teams_per_game": 2},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(2, payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(2, len(payload["plan"][0]))
        self.assertEqual(2, payload["max_games_count"])
    
    def test_schedule_v2_uses_num_rounds_and_num_fields_and_num_teams_per_game(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 2, "num_rounds": 4, "num_fields": 2, "num_teams_per_game": 2},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(2, payload["version"])
        self.assertEqual(4, len(payload["plan"]))
        self.assertEqual(2, len(payload["plan"][0]))
        self.assertEqual(1, payload["max_games_count"])


    def test_schedule_rejects_invalid_version(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 6},
        )

        self.assertEqual(400, response.status_code)
        self.assertIn("version", response.get_json()["error"])

    def test_schedule_rejects_non_numeric_version(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": "latest"},
        )

        self.assertEqual(400, response.status_code)
        self.assertIn("version", response.get_json()["error"])

    def test_schedule_rejects_too_few_teams(self):
        for version in (2, 3, 4, 5):
            response = self.client.post(
                "/schedule", json={"num_teams": 3, "version": version}
            )

            self.assertEqual(400, response.status_code)

    def test_schedule_v3_uses_shape_parameters(self):
        response = self.client.post(
            "/schedule",
            json={
                "num_teams": 16,
                "version": 3,
                "num_rounds": 4,
                "num_fields": 2,
                "num_teams_per_game": 2,
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(4, len(payload["plan"]))
        self.assertEqual(2, len(payload["plan"][0]))
        self.assertEqual(2, len(payload["plan"][0][0]))
        self.assertEqual(1, payload["max_games_count"])

    def test_schedule_allows_v4(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": 4},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(4, payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        self.assertEqual(4, payload["num_teams_per_game"])
        self.assertEqual(8, payload["max_games_count"])

    def test_schedule_v4_uses_team_sizes_per_field(self):
        response = self.client.post(
            "/schedule",
            json={
                "num_teams": 20,
                "version": 4,
                "num_rounds": 6,
                "num_fields": 3,
                "num_teams_per_game": [4, 4, 2],
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(6, len(payload["plan"]))
        for round_plan in payload["plan"]:
            self.assertEqual([4, 4, 2], [len(field) for field in round_plan])
        self.assertEqual([4, 4, 2], payload["num_teams_per_game"])
        self.assertEqual(3, payload["max_games_count"])

    def test_schedule_v4_rejects_invalid_team_sizes_per_field(self):
        for sizes in ([4, 4], [4, 4, 0], [4, 4, "x"], [4, 4, None], [], "4,4,2"):
            response = self.client.post(
                "/schedule",
                json={
                    "num_teams": 20,
                    "version": 4,
                    "num_fields": 3,
                    "num_teams_per_game": sizes,
                },
            )

            self.assertEqual(400, response.status_code, sizes)

    def test_schedule_rejects_team_sizes_per_field_before_v4(self):
        for version in (1, 2, 3):
            response = self.client.post(
                "/schedule",
                json={
                    "num_teams": 16,
                    "version": version,
                    "num_teams_per_game": [4, 4, 4, 4],
                },
            )

            self.assertEqual(400, response.status_code, version)

    def test_schedule_allows_v5(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 18, "version": 5},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(5, payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        self.assertEqual(4, payload["num_teams_per_game"])
        self.assertEqual(7, payload["max_games_count"])

    def test_schedule_v5_uses_team_sizes_per_field(self):
        response = self.client.post(
            "/schedule",
            json={
                "num_teams": 20,
                "version": 5,
                "num_rounds": 6,
                "num_fields": 3,
                "num_teams_per_game": [4, 4, 2],
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(6, len(payload["plan"]))
        for round_plan in payload["plan"]:
            self.assertEqual([4, 4, 2], [len(field) for field in round_plan])
        self.assertEqual([4, 4, 2], payload["num_teams_per_game"])
        self.assertEqual(3, payload["max_games_count"])

    def test_schedule_v5_rejects_fewer_teams_than_seats_per_round(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 12, "version": 5},
        )

        self.assertEqual(400, response.status_code)

    def test_schedule_requires_num_teams(self):
        response = self.client.post("/schedule", json={"version": 2})

        self.assertEqual(400, response.status_code)
        self.assertIn("num_teams", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
