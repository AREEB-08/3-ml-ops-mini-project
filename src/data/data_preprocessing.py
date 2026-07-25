# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import re
import string
import logging
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================================
# Configure Logger
# ==========================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    # Console Logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Error File Logger
    file_handler = logging.FileHandler("transformation_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================================================
# Download Required NLTK Resources
# ==========================================================

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# ==========================================================
# Initialize NLP Utilities
# ==========================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ==========================================================
# Text Cleaning Functions
# ==========================================================

def lower_case(text: str) -> str:
    """
    Convert text to lowercase.
    """
    return text.lower()


def remove_stop_words(text: str) -> str:
    """
    Remove English stop words.
    """
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def remove_numbers(text: str) -> str:
    """
    Remove all numeric characters.
    """
    return "".join([char for char in text if not char.isdigit()])


def remove_punctuations(text: str) -> str:
    """
    Remove punctuation symbols.
    """
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = text.replace("؛", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.
    """
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def lemmatization(text: str) -> str:
    """
    Perform lemmatization.
    """
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


# ==========================================================
# Normalize Text
# ==========================================================

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply complete text preprocessing pipeline.

    Steps:
        1. Lowercase conversion
        2. Stopword removal
        3. Number removal
        4. Punctuation removal
        5. URL removal
        6. Lemmatization

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """

    try:

        logger.info("Starting text normalization...")

        df["content"] = df["content"].apply(lower_case)
        logger.info("Lowercase conversion completed.")

        df["content"] = df["content"].apply(remove_stop_words)
        logger.info("Stopword removal completed.")

        df["content"] = df["content"].apply(remove_numbers)
        logger.info("Number removal completed.")

        df["content"] = df["content"].apply(remove_punctuations)
        logger.info("Punctuation removal completed.")

        df["content"] = df["content"].apply(remove_urls)
        logger.info("URL removal completed.")

        df["content"] = df["content"].apply(lemmatization)
        logger.info("Lemmatization completed.")

        logger.info("Text normalization completed successfully.")

        return df

    except Exception as e:
        logger.exception("Error during text normalization.")
        raise


# ==========================================================
# Save Processed Data
# ==========================================================

def save_processed_data(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        output_path: str) -> None:
    """
    Save processed train and test datasets.

    Args:
        train_df (pd.DataFrame): Processed training data.
        test_df (pd.DataFrame): Processed testing data.
        output_path (str): Output directory.
    """

    os.makedirs(output_path, exist_ok=True)

    train_df.to_csv(
        os.path.join(output_path, "train_processed.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(output_path, "test_processed.csv"),
        index=False
    )

    logger.info("Processed datasets saved successfully.")


# ==========================================================
# Main Function
# ==========================================================

def main():
    """
    Execute complete data preprocessing pipeline.
    """

    try:

        logger.info("Loading datasets...")

        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")

        logger.info("Datasets loaded successfully.")

        # Normalize train and test datasets
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        # Save processed datasets
        save_processed_data(
            train_processed,
            test_processed,
            "./data/interim"
        )

        logger.info("Data preprocessing pipeline completed successfully.")

    except Exception as e:

        logger.exception("Data preprocessing pipeline failed.")

        print(f"Error: {e}")


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()