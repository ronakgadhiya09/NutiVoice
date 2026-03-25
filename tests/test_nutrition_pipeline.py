import unittest
from unittest.mock import patch

import nutrition_service


class TestNutritionPipeline(unittest.TestCase):
    def test_normalization_strips_brand_noise(self):
        out = nutrition_service.normalize_food_name("Minute Meals Double Cheese Margherita Pizza (1 whole)")
        self.assertEqual(out, "cheese margherita pizza")

    @patch("nutrition_service.lookup_open_food_facts", return_value=None)
    @patch("utils.extract_food_entities")
    def test_db_exact_match_for_eggs(self, mock_extract, _mock_off):
        mock_extract.return_value = [
            {"name": "eggs", "quantity": 2, "unit": "eggs", "meal_type": "breakfast"}
        ]
        items, meal_type = nutrition_service.resolve_meal_text("2 eggs", api_key="fake")
        self.assertEqual(meal_type, "breakfast")
        self.assertEqual(len(items), 1)
        egg = items[0]
        self.assertEqual(egg["nutrition_source"], "db")
        self.assertEqual(egg["confidence_status"], "matched")
        self.assertGreaterEqual(egg["Protein"], 10)

    @patch("nutrition_service.lookup_open_food_facts", return_value=None)
    @patch("utils.extract_food_entities")
    def test_db_fuzzy_match_for_paneer_with_brand_noise(self, mock_extract, _mock_off):
        mock_extract.return_value = [
            {
                "name": "minute meals paneer butter masala",
                "quantity": 1,
                "unit": "serving",
                "meal_type": "lunch",
            }
        ]
        items, _ = nutrition_service.resolve_meal_text("minute meals paneer butter masala", api_key="fake")
        self.assertEqual(len(items), 1)
        paneer = items[0]
        self.assertIn("paneer", paneer["Name"])
        self.assertIn(paneer["confidence"], ("high", "medium"))
        self.assertIn(paneer["confidence_status"], ("matched", "needs_review"))

    @patch("nutrition_service.lookup_open_food_facts", return_value=None)
    @patch("utils.estimate_nutrition_batch")
    @patch("utils.extract_food_entities")
    def test_llm_mismatch_gets_corrected(self, mock_extract, mock_estimate, _mock_off):
        mock_extract.return_value = [
            {"name": "mystery curry", "quantity": 1, "unit": "serving", "meal_type": "dinner"}
        ]
        mock_estimate.return_value = [
            {
                "Name": "mystery curry",
                "quantity": 1,
                "unit": "serving",
                "Calories": 120,
                "Protein": 20,
                "Carbs": 20,
                "Fats": 20,
            }
        ]

        items, _ = nutrition_service.resolve_meal_text("mystery curry", api_key="fake")
        self.assertEqual(len(items), 1)
        out = items[0]
        # 20*4 + 20*4 + 20*9 = 340
        self.assertEqual(out["Calories"], 340)
        self.assertEqual(out["nutrition_source"], "llm_corrected")
        self.assertIn("corrected_macros", out["flags"])

    def test_coke_anomaly_flagged(self):
        out = nutrition_service.validate_and_correct(
            {
                "Name": "coke",
                "quantity": 1,
                "unit": "serving",
                "Calories": 100,
                "Protein": 8,
                "Carbs": 20,
                "Fats": 4,
                "nutrition_source": "llm_fallback",
                "confidence": "medium-low",
                "flags": ["llm_fallback"],
            }
        )
        self.assertIn("beverage_macro_anomaly", out["flags"])
        self.assertIn(out["confidence"], ("low", "medium-low"))


if __name__ == "__main__":
    unittest.main()
