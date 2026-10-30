# TruthLens AI – Intelligent Fake News Detection & Live News Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

TruthLens AI is an intelligent news-analysis and media literacy platform that combines **Machine Learning text classification** with **real-time live news cross-referencing via TheNewsAPI**. It provides readers, researchers, and journalists with structural pattern predictions, external publisher coverage, and side-by-side evidence comparison without claiming to act as an infallible arbiter of truth.

---

## 1. Key Features

- **Machine Learning Classification**: Evaluates linguistic patterns, vocabulary n-grams, and stylistic markers using a trained Logistic Regression model and a 50,000-feature TF-IDF vectorizer.
- **Probabilistic Confidence Meter**: Displays predicted class probabilities (Real vs. Fake), confidence percentage, and limitation disclosures.
- **Real-Time Live News Verification**: Queries **TheNewsAPI** using multi-tiered search strategies (full headline + extracted topical keywords) to discover corroborating or related coverage from indexed global news sources.
- **Evidence Comparison Matrix**: Compares user-submitted claims against external articles, reporting publication dates, news sources, text similarity scores, and coverage alignment categories.
- **Objective Synthesis Engine**: Generates a balanced explanation of linguistic pattern findings alongside live external reporting.
- **Persistent Scan History**: SQLite-backed audit log storing past analyses with real-time client-side search filtering and single-click history clearing.
- **Modern Dark Glassmorphism Interface**: Deep navy/black background, cyan and blue accents, soft ambient glows, animated multi-step loading indicators, and a responsive layout.
- **System Health Endpoint (`/health`)**: Diagnostic JSON endpoint monitoring model, vectorizer, database, and API token readiness without exposing secrets.

---

## 2. Ethical Transparency & Limitations

> [!IMPORTANT]
> **A machine learning prediction must never be presented as confirmed factual truth.**
> Linguistic models detect vocabulary and phrasing patterns learned from training data (e.g. sensationalist phrasing, clickbait markers). They cannot verify empirical facts in the physical world.

> [!NOTE]
> **Live article search results do not automatically prove a claim is true or false.**
> The absence of search results ("INSUFFICIENT EVIDENCE") indicates that an event is not indexed by mainstream publishers; it does not mean the event did not occur.

---

## 3. Technology Stack

- **Frontend**: HTML5, CSS3 (Glassmorphism, CSS Grid, Flexbox), Vanilla JavaScript (ES6+)
- **Backend**: Python 3, Flask Microframework
- **Machine Learning**: Scikit-learn, Pandas, NumPy, Joblib, TF-IDF Vectorizer, Logistic Regression
- **Live Search**: TheNewsAPI REST API, Requests library, Python Dotenv
- **Database**: SQLite3 (`database/history.db`)
- **Testing**: Python `unittest` framework with automated mocking and real connectivity suites

---

## 4. Project Structure

```
TruthLens-AI/
│
├── app.py                      # Main Flask application and routing
├── requirements.txt            # Python package dependencies
├── .env                        # Local environment configuration (API tokens, secret key)
├── .env.example                # Example environment template
├── README.md                   # Complete documentation and setup guide
│
├── model/
│   ├── fake_news_model.pkl     # Pre-trained Logistic Regression model
│   └── tfidf_vectorizer.pkl    # Pre-trained 50,000-feature TF-IDF vectorizer
│
├── templates/
│   ├── index.html              # Main dashboard (Sections A through E, Quick Samples, Loading State)
│   ├── history.html            # SQLite scan history viewer and search
│   ├── about.html              # About project, methodology, and ethical manifesto
│   ├── how_it_works.html       # Step-by-step pipeline breakdown
│   ├── 404.html                # 404 Not Found error page
│   └── 500.html                # 500 Internal Server Error page
│
├── static/
│   ├── style.css               # Premium cybernetic dark styling system
│   └── script.js               # Reactive text counting, quick samples, progress loading
│
├── database/
│   ├── db.py                   # SQLite schema initialization and CRUD helpers
│   └── history.db              # SQLite persistent scan database
│
├── utils/
│   ├── prediction.py           # ML inference, confidence calibration, error fallbacks
│   ├── news_api.py             # Multi-query TheNewsAPI retrieval & deduplication
│   ├── text_processing.py      # Normalization, keyword extraction, similarity calculation
│   └── evidence.py             # Evidence matrix comparison and objective synthesis
│
└── tests/
    └── test_truthlens.py       # Automated unit tests covering all required scenarios
```

---

## 5. Installation & Local Setup

### Step 1: Clone or Open Project Directory

```bash
cd TruthLens-AI
```

### Step 2: Set Up Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root (or copy from `.env.example`):

```env
THENEWSAPI_TOKEN=your_thenewsapi_token_here
FLASK_SECRET_KEY=your_secure_random_flask_secret_key
```

> **Note:** If you do not have a TheNewsAPI token yet, get a free key at [thenewsapi.com](https://www.thenewsapi.com/). If no token is provided, TruthLens AI will gracefully display `VERIFICATION UNAVAILABLE` with clear instructions without crashing.

### Step 5: Verify Model Artifacts

Ensure the model files are located in the `model/` folder:
- `model/fake_news_model.pkl`
- `model/tfidf_vectorizer.pkl`

*(If missing, the app will display `MODEL UNAVAILABLE` without terminating).*

### Step 6: Launch the Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 6. Running Automated Tests

To run the complete automated test suite (covering model prediction, missing keys, article parsing, zero results, API errors, empty inputs, database CRUD, and Flask routes):

```bash
python -m unittest tests/test_truthlens.py
```

Expected output:
```
Ran 12 tests in 0.234s

OK
```

---

## 7. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Home dashboard interface with news analyzer |
| `/analyze` | `POST` | Analyze headline (supports form submission & JSON API) |
| `/history` | `GET` | View past analysis history |
| `/history/clear` | `POST` | Clear all SQLite history records |
| `/about` | `GET` | About page and ethical principles |
| `/how-it-works`| `GET` | Detailed pipeline architecture documentation |
| `/health` | `GET` | System health check (model, vectorizer, token readiness) |

### Sample Health Check Response (`GET /health`)

```json
{
  "app": "TruthLens AI – Fake News Detection",
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "thenewsapi_token_loaded": true,
  "database_connected": true,
  "disclaimer": "Predictions are probabilistic and not proof of factual truth."
}
```

---

## 8. License

Distributed under the MIT License. Developed for transparent news evaluation and education.
