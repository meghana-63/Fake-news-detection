from flask import Flask, render_template, request
import joblib
import re
import string
import requests
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher

app = Flask(__name__)

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


# ==========================================
# LOAD ML MODEL
# ==========================================
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# ==========================================
# TEXT CLEANING
# ==========================================
def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# CREATE SEARCH QUERY
# ==========================================
def create_search_query(news):

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        news
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Use first 15 words for live search
    words = text.split()

    query = " ".join(words[:15])

    return query[:200]


# ==========================================
# LIVE NEWS VERIFICATION
# ==========================================
def verify_live_news(news):

    if not GNEWS_API_KEY:
        return {
            "status": "API KEY MISSING",
            "message": "Live news verification is not configured.",
            "articles": []
        }

    query = create_search_query(news)

    if not query:
        return {
            "status": "NOT VERIFIED",
            "message": "No searchable news text was provided.",
            "articles": []
        }

    url = "https://gnews.io/api/v4/search"

    params = {
        "q": query,
        "lang": "en",
        "country": "in",
        "max": 5,
        "apikey": GNEWS_API_KEY
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return {
                "status": "VERIFICATION UNAVAILABLE",
                "message": "Live news service could not be reached.",
                "articles": []
            }

        data = response.json()

        articles = data.get("articles", [])

        if not articles:
            return {
                "status": "NOT FOUND",
                "message": (
                    "No matching current news article "
                    "was found."
                ),
                "articles": []
            }

        matched_articles = []

        for article in articles:

            title = article.get(
                "title",
                ""
            )

            description = article.get(
                "description",
                ""
            )

            source = article.get(
                "source",
                {}
            )

            source_name = source.get(
                "name",
                "Unknown source"
            )

            article_url = article.get(
                "url",
                "#"
            )

            published_at = article.get(
                "publishedAt",
                ""
            )

            searchable_article = (
                title + " " + description
            )

            similarity = SequenceMatcher(
                None,
                news.lower(),
                searchable_article.lower()
            ).ratio()

            matched_articles.append({
                "title": title,
                "description": description,
                "source": source_name,
                "url": article_url,
                "published": published_at,
                "similarity": round(
                    similarity * 100,
                    1
                )
            })

        # Sort by similarity
        matched_articles.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        best_match = matched_articles[0]

        if best_match["similarity"] >= 25:
            status = "CURRENT SOURCES FOUND"
            message = (
                "Related current news reports "
                "were found."
            )
        else:
            status = "WEAK MATCH"
            message = (
                "Current articles were found, "
                "but the match is weak."
            )

        return {
            "status": status,
            "message": message,
            "articles": matched_articles[:3]
        }

    except requests.exceptions.RequestException:

        return {
            "status": "VERIFICATION UNAVAILABLE",
            "message": (
                "Could not connect to the live "
                "news service."
            ),
            "articles": []
        }

    except Exception:

        return {
            "status": "VERIFICATION ERROR",
            "message": (
                "An unexpected verification error occurred."
            ),
            "articles": []
        }


# ==========================================
# HOME PAGE
# ==========================================
@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICT
# ==========================================
@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    news = request.form.get(
        "news",
        ""
    ).strip()

    # --------------------------------------
    # EMPTY INPUT
    # --------------------------------------
    if not news:

        return render_template(
            "index.html",
            prediction="Please enter some news.",
            confidence="",
            warning="",
            live_status="",
            live_message="",
            live_articles=[]
        )

    # --------------------------------------
    # MINIMUM WORD CHECK
    # --------------------------------------
    word_count = len(
        news.split()
    )

    if word_count < 5:

        return render_template(
            "index.html",
            prediction="INSUFFICIENT INFORMATION",
            confidence="",
            warning=(
                "Please enter a complete news "
                "headline or article."
            ),
            live_status="",
            live_message="",
            live_articles=[]
        )

    # ======================================
    # ML PREDICTION
    # ======================================

    cleaned_news = clean_text(
        news
    )

    news_tfidf = vectorizer.transform(
        [cleaned_news]
    )

    prediction = model.predict(
        news_tfidf
    )[0]

    probabilities = model.predict_proba(
        news_tfidf
    )[0]

    fake_probability = (
        probabilities[0] * 100
    )

    real_probability = (
        probabilities[1] * 100
    )

    if prediction == 1:

        result = "REAL NEWS"
        confidence = real_probability

    else:

        result = "FAKE NEWS"
        confidence = fake_probability

    # ======================================
    # LIVE NEWS VERIFICATION
    # ======================================

    live_result = verify_live_news(
        news
    )

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    return render_template(
        "index.html",

        prediction=result,

        confidence=f"{confidence:.2f}%",

        warning="",

        live_status=live_result["status"],

        live_message=live_result["message"],

        live_articles=live_result["articles"]
    )


# ==========================================
# RUN APPLICATION
# ==========================================
if __name__ == "__main__":

    app.run(
        debug=True
    )