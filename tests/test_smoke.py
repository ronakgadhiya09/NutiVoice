"""Smoke tests for persistence, aggregations, and insights (no external APIs)."""

import os
import tempfile
import unittest
from datetime import date

import data_store
import insights


class TestDataStore(unittest.TestCase):
    def setUp(self):
        self._fd, self._path = tempfile.mkstemp(suffix=".db")
        os.close(self._fd)
        self._old = os.environ.get("NUTRIVOICE_DB_PATH")
        os.environ["NUTRIVOICE_DB_PATH"] = self._path
        data_store.init_schema()

    def tearDown(self):
        if self._old is None:
            os.environ.pop("NUTRIVOICE_DB_PATH", None)
        else:
            os.environ["NUTRIVOICE_DB_PATH"] = self._old
        try:
            os.remove(self._path)
        except OSError:
            pass

    def test_add_and_fetch_range(self):
        items = [
            {
                "Name": "Test food",
                "Calories": 100,
                "Protein": 10,
                "Carbs": 12,
                "Fats": 3,
                "nutrition_source": "db",
                "confidence_status": "matched",
            }
        ]
        data_store.add_meal_with_items(
            meal_type="lunch",
            source="text",
            raw_input="test",
            transcript=None,
            items=items,
        )
        d = date.today()
        rows = data_store.fetch_items_in_range(d, d)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Name"], "Test food")
        self.assertEqual(rows[0]["Calories"], 100)


class TestInsights(unittest.TestCase):
    def test_aggregate_by_day(self):
        rows = [
            {
                "created_at": "2025-01-01T08:00:00",
                "Calories": 500,
                "Protein": 30,
                "Carbs": 50,
                "Fats": 20,
            },
            {
                "created_at": "2025-01-01T12:00:00",
                "Calories": 300,
                "Protein": 10,
                "Carbs": 20,
                "Fats": 5,
            },
            {
                "created_at": "2025-01-02T08:00:00",
                "Calories": 2000,
                "Protein": 150,
                "Carbs": 200,
                "Fats": 70,
            },
        ]
        by = insights.aggregate_by_day(rows)
        self.assertEqual(by["2025-01-01"]["Calories"], 800)
        self.assertEqual(by["2025-01-01"]["Protein"], 40)
        self.assertEqual(by["2025-01-02"]["Calories"], 2000)

    def test_rule_based_insights_low_protein(self):
        goals = {"Calories": 2000, "Protein": 150, "Carbs": 250, "Fats": 70}
        rows = []
        for day in range(4, 11):
            rows.append(
                {
                    "created_at": f"2025-01-{day:02d}T08:00:00",
                    "Calories": 1800,
                    "Protein": 50,
                    "Carbs": 200,
                    "Fats": 60,
                }
            )
        out = insights.rule_based_insights(rows, goals)
        self.assertTrue(any("Protein" in s for s in out))


class TestUtilsPublic(unittest.TestCase):
    def test_estimate_batch_empty(self):
        import utils

        self.assertEqual(utils.estimate_nutrition_batch([], "fake-key"), [])


if __name__ == "__main__":
    unittest.main()
