import sys
import unittest

sys.path.insert(0, "src")

from webserver import app


class ScheduleApiVersionTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_schedule_defaults_to_v2(self):
        response = self.client.post("/schedule", json={"num_teams": 20})

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual("v2", payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        self.assertEqual(6, payload["max_games_count"])

    def test_schedule_allows_v1(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": "v1"},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual("v1", payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        self.assertEqual(8, payload["max_games_count"])

    def test_schedule_allows_v2(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": "v2"},
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual("v2", payload["version"])
        self.assertEqual(8, len(payload["plan"]))
        self.assertEqual(4, len(payload["plan"][0]))
        self.assertEqual(8, payload["max_games_count"])

    def test_schedule_rejects_invalid_version(self):
        response = self.client.post(
            "/schedule",
            json={"num_teams": 16, "version": "v3"},
        )

        self.assertEqual(400, response.status_code)
        self.assertIn("version", response.get_json()["error"])

    def test_schedule_requires_num_teams(self):
        response = self.client.post("/schedule", json={"version": "v2"})

        self.assertEqual(400, response.status_code)
        self.assertIn("num_teams", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
