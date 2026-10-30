"""Machine Learning prediction module for TruthLens AI.

Loads the pre-trained Logistic Regression classifier and TF-IDF vectorizer,
applies the required preprocessing pipeline, and outputs probabilistic
predictions along with transparent uncertainty and limitation disclosures.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

import joblib

from utils.text_processing import clean_text

logger = logging.getLogger(__name__)

# Cached model instances
_CACHED_MODEL: Any = None
_CACHED_VECTORIZER: Any = None
_CACHED_MODEL_PATH: Optional[str] = None
_CACHED_VECTORIZER_PATH: Optional[str] = None
_MODEL_LOAD_ERROR: Optional[str] = None


def load_model_and_vectorizer(
    model_path: str,
    vectorizer_path: str,
    force_reload: bool = False
) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """Load model and TF-IDF vectorizer from disk with path-aware caching.

    Args:
        model_path: Absolute or relative path to fake_news_model.pkl.
        vectorizer_path: Absolute or relative path to tfidf_vectorizer.pkl.
        force_reload: Set True to bypass cached objects.

    Returns:
        Tuple of (model, vectorizer, error_message).
    """
    global _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_MODEL_PATH, _CACHED_VECTORIZER_PATH, _MODEL_LOAD_ERROR

    if (
        not force_reload
        and _CACHED_MODEL is not None
        and _CACHED_VECTORIZER is not None
        and _CACHED_MODEL_PATH == model_path
        and _CACHED_VECTORIZER_PATH == vectorizer_path
    ):
        return _CACHED_MODEL, _CACHED_VECTORIZER, None

    if not os.path.exists(model_path):
        err = f"Model file not found at: {model_path}"
        logger.warning(err)
        return None, None, err

    if not os.path.exists(vectorizer_path):
        err = f"Vectorizer file not found at: {vectorizer_path}"
        logger.warning(err)
        return None, None, err

    try:
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)

        _CACHED_MODEL = loaded_model
        _CACHED_VECTORIZER = loaded_vectorizer
        _CACHED_MODEL_PATH = model_path
        _CACHED_VECTORIZER_PATH = vectorizer_path
        logger.info("Successfully loaded ML model and TF-IDF vectorizer.")
        return _CACHED_MODEL, _CACHED_VECTORIZER, None

    except Exception as exc:
        _MODEL_LOAD_ERROR = f"Error loading model artifacts: {str(exc)}"
        logger.exception(_MODEL_LOAD_ERROR)
        _CACHED_MODEL = None
        _CACHED_VECTORIZER = None
        return None, None, _MODEL_LOAD_ERROR


def get_model_status(model_path: str, vectorizer_path: str) -> Dict[str, Any]:
    """Return health check info regarding model and vectorizer loading state."""
    model, vectorizer, error = load_model_and_vectorizer(model_path, vectorizer_path)
    return {
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "error": error,
        "classes": list(map(int, model.classes_)) if model is not None and hasattr(model, "classes_") else None
    }


def predict_news(
    text: str,
    model_path: str,
    vectorizer_path: str
) -> Dict[str, Any]:
    """Classify input text using the trained TF-IDF + Logistic Regression model.

    Dataset labels:
        0 = FAKE
        1 = REAL

    Args:
        text: Raw user headline or news content.
        model_path: Path to model pickle file.
        vectorizer_path: Path to vectorizer pickle file.

    Returns:
        Structured prediction outcome dictionary.
    """
    news_text = str(text or "").strip()

    if not news_text:
        return {
            "success": False,
            "label": "EMPTY INPUT",
            "confidence": None,
            "fake_probability": None,
            "real_probability": None,
            "status": "error",
            "message": "Please enter a valid headline or article text to analyze.",
            "limitation": "No text provided for analysis."
        }

    model, vectorizer, load_error = load_model_and_vectorizer(model_path, vectorizer_path)

    if model is None or vectorizer is None:
        return {
            "success": False,
            "label": "MODEL UNAVAILABLE",
            "confidence": None,
            "fake_probability": None,
            "real_probability": None,
            "status": "unavailable",
            "message": (
                "The machine learning model could not be loaded. "
                "Ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are placed in the model directory."
            ),
            "limitation": "Pattern classification is currently offline."
        }

    try:
        # Preprocess text matching exact training pipeline
        cleaned = clean_text(news_text)

        if not cleaned:
            return {
                "success": False,
                "label": "INSUFFICIENT TEXT",
                "confidence": None,
                "fake_probability": None,
                "real_probability": None,
                "status": "warning",
                "message": "Input contains only symbols or numbers. Provide readable news text.",
                "limitation": "Text cannot be vectorized without meaningful words."
            }

        # Transform through TF-IDF
        features = vectorizer.transform([cleaned])

        # Predict class and probabilities
        raw_pred = int(model.predict(features)[0])
        label = "REAL NEWS" if raw_pred == 1 else "FAKE NEWS"

        confidence: Optional[float] = None
        fake_prob: Optional[float] = None
        real_prob: Optional[float] = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            classes = list(model.classes_)

            fake_idx = classes.index(0) if 0 in classes else 0
            real_idx = classes.index(1) if 1 in classes else 1

            fake_prob = round(float(probs[fake_idx]) * 100, 2)
            real_prob = round(float(probs[real_idx]) * 100, 2)

            # Confidence is the probability of the predicted class
            predicted_idx = classes.index(raw_pred) if raw_pred in classes else raw_pred
            confidence = round(float(probs[predicted_idx]) * 100, 2)

        return {
            "success": True,
            "label": label,
            "raw_prediction": raw_pred,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "status": "classified",
            "message": (
                f"Model classified input as {label} with {confidence}% pattern confidence."
                if confidence is not None
                else f"Model classified input as {label}."
            ),
            "limitation": (
                "This is a machine-learning prediction based on patterns learned from training data. "
                "It is not proof of truth."
            )
        }

    except Exception as exc:
        logger.exception("Prediction failed on input: %s", exc)
        return {
            "success": False,
            "label": "MODEL ERROR",
            "confidence": None,
            "fake_probability": None,
            "real_probability": None,
            "status": "error",
            "message": f"An error occurred while evaluating the text: {str(exc)}",
            "limitation": "Classification failed due to internal processing error."
        }
