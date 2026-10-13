import pandas as pd
import joblib
import re
import string

# Load model
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

print("Model classes:", model.classes_)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Take one article from each dataset
fake_text = str(fake.iloc[0]["title"]) + " " + str(fake.iloc[0]["text"])
true_text = str(true.iloc[0]["title"]) + " " + str(true.iloc[0]["text"])


def test_article(name, text, actual):
    cleaned = clean_text(text)

    tfidf = vectorizer.transform([cleaned])

    prediction = model.predict(tfidf)[0]
    probabilities = model.predict_proba(tfidf)[0]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Actual:", actual)
    print("Predicted:", "REAL" if prediction == 1 else "FAKE")

    print("Fake probability:", round(probabilities[0] * 100, 2), "%")
    print("Real probability:", round(probabilities[1] * 100, 2), "%")


test_article(
    "FAKE DATASET ARTICLE",
    fake_text,
    "FAKE"
)

test_article(
    "REAL DATASET ARTICLE",
    true_text,
    "REAL"
)