# ==========================================================
# Sentiment Analysis Flask Application (Inference Only)
# ==========================================================
#
# Description
# ----------------------------------------------------------
# Lightweight web application built for serving real-time sentiment 
# predictions. Runs strictly in inference mode without reliance on 
# heavy MLOps tracking tools (DVC, MLflow, DagsHub) during runtime.
#
# Architecture Pipeline
# ----------------------------------------------------------
# User Request (Text)
# └── Text Preprocessing (Normalize/Stopwords/Lemmatize)
#     └── Vectorizer (TF-IDF Feature Transformation)
#         └── Machine Learning Model (Classification Binary)
#             └── Rendered Template Output (Jinja2)
#
# Project Layout
# ----------------------------------------------------------
# deployment/
# ├── app.py
# ├── models/
# │   ├── model.pkl
# │   └── vectorizer.pkl
# └── templates/
#     └── index.html
# ==========================================================


# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import re
import pickle

import nltk
from flask import Flask, render_template, request, redirect, url_for
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


# ==========================================================
# Download & Force Load Required NLTK Resources
# ==========================================================

# Silently download NLTK token and vocabulary dependencies
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Ensure WordNet lexicon is fully initialized in memory prior to requests
wordnet.ensure_loaded()


# ==========================================================
# Global Objects & Utilities Initialization
# ==========================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ==========================================================
# Initialize Flask Application
# ==========================================================

app = Flask(__name__)


# ==========================================================
# Base Directory Resolution & Artifact Paths
# ==========================================================

# Resolve root directory relative to current script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


# ==========================================================
# Load Model and Feature Extractor Artifacts
# ==========================================================

print("Loading trained binary model artifact...")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)
print("Model loaded successfully.")

print("Loading trained TF-IDF vectorizer artifact...")
with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)
print("Vectorizer loaded successfully.")


# ==========================================================
# Text Preprocessing & Normalization Functions
# ==========================================================

def lower_case(text: str) -> str:
    """Convert input string characters to lowercase."""
    return " ".join([word.lower() for word in text.split()])


def remove_stop_words(text: str) -> str:
    """Filter out non-informative English stop words."""
    words = [word for word in str(text).split() if word not in stop_words]
    return " ".join(words)


def remove_numbers(text: str) -> str:
    """Strip all numeric digits from text."""
    return "".join([char for char in text if not char.isdigit()])


def remove_urls(text: str) -> str:
    """Remove standard HTTP/HTTPS and WWW hyperlinks using Regex."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def remove_punctuation(text: str) -> str:
    """Strip punctuation symbols and collapse extra whitespace."""
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_text(text: str) -> str:
    """Reduce tokens to their core dictionary root forms (lemmas)."""
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])


def normalize_text(text: str) -> str:
    """
    Sequential text cleaning pipeline.
    Applies lowercasing, stopword removal, digit stripping, 
    URL removal, punctuation stripping, and lemmatization.
    """
    text = lower_case(text)
    text = remove_stop_words(text)
    text = remove_numbers(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = lemmatize_text(text)
    return text


# ==========================================================
# Flask Web Application Routes
# ==========================================================

@app.route("/")
def home():
    """
    Renders the primary application UI in a clean state.
    Passes empty string for `user_text` to clear input textarea.
    """
    return render_template(
        "index.html",
        result=None,
        user_text=""
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Processes user input text and predicts sentiment polarity.
    Redirects GET requests back to home route to prevent direct route access errors.
    """

    # Redirect GET requests back to home page
    if request.method == "GET":
        return redirect(url_for("home"))

    # Extract raw user input from form submission
    user_text = request.form.get("text", "")

    # Validate non-empty payload input
    if not user_text.strip():
        return render_template(
            "index.html",
            result="Please enter text.",
            user_text=user_text
        )

    # Clean and normalize raw user text
    cleaned_text = normalize_text(user_text)

    # Convert normalized string to sparse matrix and convert to dense array
    features = vectorizer.transform([cleaned_text])
    features_array = features.toarray()

    # Predict class label using model
    prediction = model.predict(features_array)
    raw_prediction = int(prediction[0])

    # Map output integer label to human-readable string: 1 -> Positive, 0 -> Negative
    sentiment = "Positive" if raw_prediction == 1 else "Negative"

    # Render template with prediction outcome and persist user text
    return render_template(
        "index.html",
        result=sentiment,
        user_text=user_text
    )


# ==========================================================
# Application Entry Point
# ==========================================================

if __name__ == "__main__":
    # Launch application server exposing host across all network interfaces
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )