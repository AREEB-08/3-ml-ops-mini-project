# ==========================================================
# Sentiment Analysis Flask Application (Inference Only)
# ==========================================================
#
# This application is used ONLY for inference (prediction).
#
# It does NOT perform:
#   - Model Training
#   - MLflow Tracking
#   - DVC Operations
#   - DagsHub Authentication
#
# It simply processes:
# User Input -> Text Preprocessing -> Vectorizer -> Model -> Prediction
#
# ==========================================================


# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import re
import pickle

import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


# ==========================================================
# Download & Force Load Required NLTK Resources
# ==========================================================

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
wordnet.ensure_loaded()


# ==========================================================
# Initialize Text Processing Objects
# ==========================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ==========================================================
# Initialize Flask Application
# ==========================================================

app = Flask(__name__)


# ==========================================================
# Model & Vectorizer Paths
# ==========================================================

# Directory Layout
# deployment/
# ├── app.py
# ├── models/
# │     ├── model.pkl
# │     └── vectorizer.pkl
# └── templates/
#       └── index.html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


# ==========================================================
# Load Model and Vectorizer
# ==========================================================

print("Loading trained model...")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)
print("Model loaded successfully.")

print("Loading vectorizer...")
with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)
print("Vectorizer loaded successfully.")


# ==========================================================
# Text Preprocessing Functions
# ==========================================================

def lower_case(text: str) -> str:
    """Convert text to lowercase."""
    return " ".join([word.lower() for word in text.split()])


def remove_stop_words(text: str) -> str:
    """Remove English stopwords."""
    words = [word for word in str(text).split() if word not in stop_words]
    return " ".join(words)


def remove_numbers(text: str) -> str:
    """Remove numerical characters."""
    return "".join([char for char in text if not char.isdigit()])


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation symbols and extra spaces."""
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_text(text: str) -> str:
    """Convert words into their base lemmatized form."""
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])


def normalize_text(text: str) -> str:
    """Complete preprocessing pipeline."""
    text = lower_case(text)
    text = remove_stop_words(text)
    text = remove_numbers(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = lemmatize_text(text)
    return text


# ==========================================================
# Flask Routes
# ==========================================================

@app.route("/")
def home():
    """Display the Home Page with clean initial state."""
    return render_template(
        "index.html",
        result=None,
        user_text=""
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Predict sentiment class for incoming text and preserve user input."""
    if request.method == "POST":
        user_text = request.form.get("text", "")

        # Check for empty or whitespace-only input
        if not user_text.strip():
            return render_template(
                "index.html",
                result="Please enter text.",
                user_text=user_text
            )

        # Preprocess text
        cleaned_text = normalize_text(user_text)

        print("\n" + "=" * 60)
        print("Original Text :", user_text)
        print("Cleaned Text  :", cleaned_text)

        # Convert Text -> Features
        features = vectorizer.transform([cleaned_text])
        features_array = features.toarray()

        print("Feature Shape :", features_array.shape)

        # Predict
        prediction = model.predict(features_array)
        raw_prediction = int(prediction[0])

        print("Raw Prediction:", raw_prediction)

        # Map label to template string: 1 = Positive, 0 = Negative
        sentiment = "Positive" if raw_prediction == 1 else "Negative"

        print("Displayed Result:", sentiment)
        print("=" * 60 + "\n")

        # Pass user_text back so <textarea> retains user input
        return render_template(
            "index.html",
            result=sentiment,
            user_text=user_text
        )


# ==========================================================
# Application Entry Point
# ==========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )