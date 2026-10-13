import pandas as pd
import re
import string
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Fake articles:", len(fake))
print("Real articles:", len(true))


# ==========================================
# 2. ADD LABELS
# ==========================================

fake["label"] = 0
true["label"] = 1


# ==========================================
# 3. COMBINE DATASETS
# ==========================================

df = pd.concat(
    [fake, true],
    ignore_index=True
)

df = df[
    ["title", "text", "label"]
]

df = df.dropna()


# ==========================================
# 4. COMBINE TITLE + ARTICLE
# ==========================================

df["content"] = (
    df["title"].astype(str)
    + " "
    + df["text"].astype(str)
)


# ==========================================
# 5. TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["content"] = df["content"].apply(clean_text)


# ==========================================
# 6. REMOVE VERY SHORT ARTICLES
# ==========================================

df = df[
    df["content"].str.split().str.len() >= 10
]

print("\nTotal usable articles:", len(df))


# ==========================================
# 7. SHUFFLE DATA
# ==========================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 8. INPUT + OUTPUT
# ==========================================

X = df["content"]
y = df["label"]


# ==========================================
# 9. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 10. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    max_df=0.7,
    min_df=2
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print("\nTF-IDF conversion completed.")


# ==========================================
# 11. LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed.")


# ==========================================
# 12. PREDICTION
# ==========================================

y_pred = model.predict(
    X_test_tfidf
)


# ==========================================
# 13. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 50)
print("           MODEL PERFORMANCE")
print("=" * 50)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake",
            "Real"
        ]
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 14. SAVE MODEL
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    "model/fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


print("\n")
print("=" * 50)
print("       MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print(
    "model/fake_news_model.pkl"
)

print(
    "model/tfidf_vectorizer.pkl"
)