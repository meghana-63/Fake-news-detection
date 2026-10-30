"""TruthLens AI - Intelligent Fake News Detection & Live News Verification.

Flask Application serving ML predictions, live news search via TheNewsAPI,
transparent evidence comparison, and SQLite-backed scan history.
"""

import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash
)

from utils.prediction import (
    predict_news,
    load_model_and_vectorizer,
    get_model_status
)
from utils.news_api import verify_live_news
from utils.evidence import build_evidence_analysis
from database.db import (
    init_db,
    add_scan_record,
    get_scan_history,
    clear_scan_history,
    get_history_stats
)

# -----------------------------------------------------------------------------
# Configuration & Initialization
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("truthlens")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "truthlens-ai-secure-key-2026")

# Directory paths
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "history.db")

# Initialize database schema
init_db(DB_PATH)

# Pre-load model and vectorizer at startup
load_model_and_vectorizer(MODEL_PATH, VECTORIZER_PATH)


def get_api_token() -> str:
    """Retrieve TheNewsAPI token from environment."""
    return os.getenv("THENEWSAPI_TOKEN", "").strip()


# -----------------------------------------------------------------------------
# Web Routes
# -----------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """Render the TruthLens AI home dashboard."""
    stats = get_history_stats(DB_PATH)
    return render_template(
        "index.html",
        stats=stats,
        page_title="TruthLens AI – Misinformation Detection & Verification"
    )


@app.route("/predict", methods=["GET", "POST"])
@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """Handle news analysis requests.

    Supports both standard HTML form submissions and asynchronous JSON requests.
    """
    if request.method == "GET":
        return redirect(url_for("home"))

    # Extract input text from JSON or Form submission
    if request.is_json:
        data = request.get_json() or {}
        news_input = data.get("news") or data.get("headline") or ""
    else:
        news_input = (
            request.form.get("news")
            or request.form.get("text")
            or request.form.get("headline")
            or request.form.get("article")
            or ""
        )

    news_text = str(news_input).strip()

    if not news_text:
        error_msg = "Please enter a news headline or article to analyze."
        if request.is_json:
            return jsonify({"success": False, "error": error_msg}), 400
        stats = get_history_stats(DB_PATH)
        return render_template("index.html", error=error_msg, stats=stats)

    if len(news_text) > 30000:
        error_msg = "Input text exceeds character limit (30,000 characters). Please provide a shorter excerpt."
        if request.is_json:
            return jsonify({"success": False, "error": error_msg}), 400
        stats = get_history_stats(DB_PATH)
        return render_template("index.html", error=error_msg, news=news_text[:500], stats=stats)

    # 1. Run Machine Learning Pattern Prediction
    prediction = predict_news(news_text, MODEL_PATH, VECTORIZER_PATH)

    # 2. Run Live News Search Verification
    token = get_api_token()
    verification = verify_live_news(news_text, token)

    # 3. Build Evidence Comparison & Synthesis
    evidence = build_evidence_analysis(news_text, prediction, verification)

    # 4. Save analysis to SQLite history
    top_source = (
        verification["articles"][0].get("source")
        if verification.get("articles")
        else None
    )
    add_scan_record(
        db_path=DB_PATH,
        headline=news_text[:250],
        prediction_label=prediction.get("label"),
        confidence=prediction.get("confidence"),
        fake_prob=prediction.get("fake_probability"),
        real_prob=prediction.get("real_probability"),
        live_status=verification.get("status"),
        articles_count=len(verification.get("articles", [])),
        top_source=top_source
    )

    stats = get_history_stats(DB_PATH)

    # Return JSON for AJAX requests
    if request.is_json:
        return jsonify({
            "success": True,
            "headline": news_text,
            "prediction": prediction,
            "verification": verification,
            "evidence": evidence,
            "stats": stats
        })

    # Render HTML template for standard form submission
    return render_template(
        "index.html",
        news=news_text,
        prediction=prediction,
        verification=verification,
        evidence=evidence,
        stats=stats
    )


@app.route("/history", methods=["GET"])
def history_view():
    """Display past news scans saved in SQLite."""
    history = get_scan_history(DB_PATH, limit=100)
    stats = get_history_stats(DB_PATH)
    return render_template(
        "history.html",
        history=history,
        stats=stats,
        page_title="Scan History – TruthLens AI"
    )


@app.route("/history/clear", methods=["POST"])
def clear_history_action():
    """Clear all records from scan history."""
    success = clear_scan_history(DB_PATH)
    if request.is_json:
        return jsonify({"success": success})
    flash("Scan history cleared successfully." if success else "Failed to clear history.")
    return redirect(url_for("history_view"))


@app.route("/about", methods=["GET"])
def about():
    """Render the About page describing project background and methodology."""
    return render_template(
        "about.html",
        page_title="About – TruthLens AI"
    )


@app.route("/how-it-works", methods=["GET"])
def how_it_works():
    """Render the How It Works page detailing the ML pipeline and live search."""
    return render_template(
        "how_it_works.html",
        page_title="How It Works – TruthLens AI"
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint reporting application and dependency status.

    Never exposes the raw API token.
    """
    token_loaded = bool(get_api_token())
    model_info = get_model_status(MODEL_PATH, VECTORIZER_PATH)

    status_code = 200 if (model_info["model_loaded"] and model_info["vectorizer_loaded"]) else 503

    return jsonify({
        "app": "TruthLens AI – Fake News Detection",
        "status": "healthy" if status_code == 200 else "degraded",
        "model_loaded": model_info["model_loaded"],
        "vectorizer_loaded": model_info["vectorizer_loaded"],
        "thenewsapi_token_loaded": token_loaded,
        "database_connected": os.path.exists(DB_PATH),
        "disclaimer": "Predictions are probabilistic and not proof of factual truth."
    }), status_code


# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal application error: %s", error)
    return render_template("500.html"), 500


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
