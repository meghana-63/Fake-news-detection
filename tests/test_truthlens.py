"""Automated test suite for TruthLens AI.

Tests required scenarios:
1. Model prediction & probability calibration
2. Missing API key handling
3. API returning articles (with mocked and live verification)
4. API returning zero articles (INSUFFICIENT EVIDENCE check)
5. API returning an error (VERIFICATION UNAVAILABLE check)
6. Empty user input validation
7. Flask routes & health endpoint
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Set path to parent directory
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.prediction import predict_news, load_model_and_vectorizer
from utils.news_api import verify_live_news
from utils.text_processing import clean_text, extract_keywords, calculate_similarity
from database.db import init_db, add_scan_record, get_scan_history, clear_scan_history
from app import app

MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
TEST_DB_PATH = os.path.join(BASE_DIR, "database", "test_history.db")


class TruthLensComprehensiveTests(unittest.TestCase):
    """Test suite covering core ML, live API, evidence synthesis, and Flask endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        init_db(TEST_DB_PATH)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 1. Model Prediction Tests
    # -------------------------------------------------------------------------
    def test_model_prediction_valid_input(self):
        """Test that existing model files load and predict with confidence score."""
        sample_headline = "White House announces new economic and infrastructure initiatives."
        result = predict_news(sample_headline, MODEL_PATH, VECTORIZER_PATH)

        self.assertTrue(result["success"], "Prediction should succeed for readable news text")
        self.assertIn(result["label"], ["REAL NEWS", "FAKE NEWS"], "Prediction must be REAL NEWS or FAKE NEWS")
        self.assertIsNotNone(result["confidence"], "Confidence score must be provided")
        self.assertGreaterEqual(result["confidence"], 50.0, "Confidence should be at least 50%")
        self.assertLessEqual(result["confidence"], 100.0, "Confidence should not exceed 100%")
        self.assertIn("This is a machine-learning prediction", result["limitation"], "Disclaimer must be present")

    def test_model_handles_missing_files_gracefully(self):
        """Test graceful fallback when model files do not exist."""
        result = predict_news("Test news", "non_existent_model.pkl", "non_existent_vec.pkl")
        self.assertFalse(result["success"])
        self.assertEqual(result["label"], "MODEL UNAVAILABLE")
        self.assertIn("Ensure 'fake_news_model.pkl'", result["message"])

    # -------------------------------------------------------------------------
    # 2. Missing API Key Test
    # -------------------------------------------------------------------------
    def test_missing_api_key(self):
        """Test verification response when API token is missing or empty."""
        result_empty = verify_live_news("Government passes new education bill", api_token="")
        self.assertEqual(result_empty["status"], "VERIFICATION UNAVAILABLE")
        self.assertIn("token is missing", result_empty["message"])
        self.assertEqual(result_empty["articles"], [])

        result_none = verify_live_news("Government passes new education bill", api_token=None)
        self.assertEqual(result_none["status"], "VERIFICATION UNAVAILABLE")
        self.assertEqual(result_none["articles"], [])

    # -------------------------------------------------------------------------
    # 3. API Returning Articles Test
    # -------------------------------------------------------------------------
    @patch("utils.news_api._make_api_request")
    def test_api_returning_articles(self, mock_get):
        """Test proper parsing and ranking when TheNewsAPI returns valid articles."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"found": 2, "returned": 2},
            "data": [
                {
                    "title": "NASA discovers new exoplanet atmosphere compounds",
                    "description": "Astronomers using space telescopes discovered signatures of methane and carbon dioxide.",
                    "source": {"name": "Science Journal"},
                    "url": "https://example.com/nasa-discovery",
                    "published_at": "2026-10-15T12:00:00.000000Z"
                },
                {
                    "title": "Space Agency Updates Exoplanet Research",
                    "description": "Detailed analysis of atmospheric data from recent deep space surveys.",
                    "source": "Space News Daily",
                    "url": "https://example.com/space-updates",
                    "published_at": "2026-10-16T14:30:00.000000Z"
                }
            ]
        }
        mock_get.return_value = mock_response

        headline = "NASA discovers exoplanet atmosphere with carbon compounds"
        result = verify_live_news(headline, api_token="dummy_token_12345")

        self.assertEqual(result["status"], "RELATED SOURCES FOUND – NOT VERIFIED")
        self.assertEqual(len(result["articles"]), 2)
        
        # Verify required article fields
        first_art = result["articles"][0]
        self.assertIn("title", first_art)
        self.assertIn("source", first_art)
        self.assertIn("published", first_art)
        self.assertIn("description", first_art)
        self.assertIn("similarity", first_art)
        self.assertIn("url", first_art)
        self.assertTrue(first_art["url"].startswith("http"))
        self.assertGreater(first_art["similarity"], 0.0)

    # -------------------------------------------------------------------------
    # 4. API Returning Zero Articles Test
    # -------------------------------------------------------------------------
    @patch("utils.news_api._make_api_request")
    def test_api_returning_zero_articles(self, mock_get):
        """Test INSUFFICIENT EVIDENCE handling when zero articles match."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"meta": {"found": 0, "returned": 0}, "data": []}
        mock_get.return_value = mock_response

        headline = "Completely fictional event happening in unregistered location 9999"
        result = verify_live_news(headline, api_token="dummy_token_12345")

        self.assertEqual(result["status"], "INSUFFICIENT EVIDENCE")
        self.assertIn("No related articles were returned by the searched sources", result["message"])
        self.assertIn("does not mean the claim is false", result["message"])
        self.assertEqual(result["articles"], [])

    # -------------------------------------------------------------------------
    # 5. API Returning An Error Test
    # -------------------------------------------------------------------------
    @patch("utils.news_api._make_api_request")
    def test_api_returning_http_error(self, mock_get):
        """Test graceful degradation when TheNewsAPI returns 401 or 500 error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = '{"error": "Unauthorized API token"}'
        mock_get.return_value = mock_response

        headline = "Sample news headline for error test"
        result = verify_live_news(headline, api_token="invalid_key")

        self.assertEqual(result["status"], "VERIFICATION UNAVAILABLE")
        self.assertIn("Verification service notice", result["message"])
        self.assertEqual(result["articles"], [])

    # -------------------------------------------------------------------------
    # 6. Empty User Input Test
    # -------------------------------------------------------------------------
    def test_empty_user_input(self):
        """Test handling of empty, whitespace, and symbol-only text inputs."""
        empty_pred = predict_news("", MODEL_PATH, VECTORIZER_PATH)
        self.assertFalse(empty_pred["success"])
        self.assertEqual(empty_pred["label"], "EMPTY INPUT")

        ws_pred = predict_news("   \n\t  ", MODEL_PATH, VECTORIZER_PATH)
        self.assertFalse(ws_pred["success"])
        self.assertEqual(ws_pred["label"], "EMPTY INPUT")

        symbols_pred = predict_news("!@#$%^&*()", MODEL_PATH, VECTORIZER_PATH)
        self.assertFalse(symbols_pred["success"])
        self.assertEqual(symbols_pred["label"], "INSUFFICIENT TEXT")

        empty_verify = verify_live_news("", api_token="some_token")
        self.assertEqual(empty_verify["status"], "INSUFFICIENT INFORMATION")

    # -------------------------------------------------------------------------
    # 7. Flask Endpoints & Health Check Test
    # -------------------------------------------------------------------------
    def test_health_endpoint(self):
        """Verify /health returns application status without exposing raw token."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["app"], "TruthLens AI – Fake News Detection")
        self.assertIn("status", data)
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["vectorizer_loaded"])
        self.assertIn("thenewsapi_token_loaded", data)
        self.assertIsInstance(data["thenewsapi_token_loaded"], bool)
        # Verify raw token is NEVER exposed
        self.assertNotIn("token", str(data).lower().replace("thenewsapi_token_loaded", ""))

    def test_home_page_renders(self):
        """Verify home page loads successfully with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"TRUTHLENS", response.data)
        self.assertIn(b"AI", response.data)

    def test_about_and_how_it_works_pages(self):
        """Verify navigation pages load."""
        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 200)
        self.assertIn(b"About", res_about.data)

        res_how = self.client.get("/how-it-works")
        self.assertEqual(res_how.status_code, 200)
        self.assertIn(b"How", res_how.data)

    def test_post_analyze_empty_input(self):
        """Verify submitting empty form returns clean validation error without crash."""
        response = self.client.post("/analyze", data={"news": ""}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter a news headline", response.data)

    # -------------------------------------------------------------------------
    # 8. SQLite Scan History Test
    # -------------------------------------------------------------------------
    def test_database_history_crud(self):
        """Verify SQLite insertion, retrieval, and clearing."""
        record_id = add_scan_record(
            db_path=TEST_DB_PATH,
            headline="Test Database Record Headline",
            prediction_label="REAL NEWS",
            confidence=88.5,
            fake_prob=11.5,
            real_prob=88.5,
            live_status="RELATED SOURCES FOUND – NOT VERIFIED",
            articles_count=3,
            top_source="Reuters"
        )
        self.assertIsNotNone(record_id)

        history = get_scan_history(TEST_DB_PATH, limit=10)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["headline"], "Test Database Record Headline")
        self.assertEqual(history[0]["prediction_label"], "REAL NEWS")

        cleared = clear_scan_history(TEST_DB_PATH)
        self.assertTrue(cleared)
        after_clear = get_scan_history(TEST_DB_PATH)
        self.assertEqual(len(after_clear), 0)


if __name__ == "__main__":
    unittest.main()
