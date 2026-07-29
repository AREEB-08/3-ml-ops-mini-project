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
# It simply:
#
# User Input
#      │
#      ▼
# Text Preprocessing
#      │
#      ▼
# Vectorizer (vectorizer.pkl)
#      │
#      ▼
# Trained Model (model.pkl)
#      │
#      ▼
# Prediction
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
# Download Required NLTK Resources
# ==========================================================

# These commands are safe to call repeatedly.
# If the resources already exist, NLTK will not download them again.



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

# Project Structure
#
# deployment/
# │
# ├── app.py
# ├── models/
# │     ├── model.pkl
# │     └── vectorizer.pkl
# └── templates/
#       └── index.html
#

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
    """
    Convert text to lowercase.
    """
    return text.lower()


def remove_stop_words(text: str) -> str:
    """
    Remove English stopwords.
    """
    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(words)


def remove_numbers(text: str) -> str:
    """
    Remove numerical characters.
    """
    return "".join(
        char
        for char in text
        if not char.isdigit()
    )


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.
    """

    url_pattern = re.compile(
        r"https?://\S+|www\.\S+"
    )

    return url_pattern.sub("", text)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation symbols.
    """

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def lemmatize_text(text: str) -> str:
    """
    Convert words into their base form.
    """

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


def normalize_text(text: str) -> str:
    """
    Complete preprocessing pipeline.
    """

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
    """
    Display the Home Page.
    """
    return render_template(
        "index.html",
        result=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict sentiment of user input.
    """

    # ---------------------------------------------
    # Read User Input
    # ---------------------------------------------
    user_text = request.form.get("text", "")

    if not user_text.strip():
        return render_template(
            "index.html",
            result="Please enter some text."
        )

    # ---------------------------------------------
    # Text Cleaning
    # ---------------------------------------------
    cleaned_text = normalize_text(user_text)

    print("\n" + "=" * 60)
    print("Original Text :", user_text)
    print("Cleaned Text  :", cleaned_text)

    # ---------------------------------------------
    # Convert Text → Numerical Features
    # ---------------------------------------------
    features = vectorizer.transform([cleaned_text])

    print("Feature Shape :", features.shape)

    # ---------------------------------------------
    # Predict
    # ---------------------------------------------
    prediction = model.predict(features)

    print("Raw Prediction:", prediction)

    prediction = int(prediction[0])

    print("Prediction Int:", prediction)

    # ---------------------------------------------
    # Convert Numeric Label to Text
    # ---------------------------------------------
    if prediction == 1:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative ☹️"

    print("Displayed Result:", sentiment)
    print("=" * 60 + "\n")

    return render_template(
        "index.html",
        result=sentiment
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