# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


# ==========================================================
# Configure Logger
# ==========================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Prevent duplicate log handlers
if not logger.handlers:

    # Console Logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Error File Logger
    file_handler = logging.FileHandler("model_building_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================================================
# Load Feature Engineered Dataset
# ==========================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load feature engineered dataset.

    Args:
        file_path (str): Path of processed CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """

    try:

        df = pd.read_csv(file_path)

        logger.info("Dataset loaded successfully from %s", file_path)

        return df

    except pd.errors.ParserError as e:

        logger.error("CSV parsing failed: %s", e)

        raise

    except Exception as e:

        logger.exception("Unexpected error while loading dataset.")

        raise


# ==========================================================
# Train Logistic Regression Model
# ==========================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        LogisticRegression: Trained model.
    """

    try:

        logger.info("Training Logistic Regression model...")

        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            penalty="l2",
            random_state=42
        )

        model.fit(X_train, y_train)

        logger.info("Model training completed successfully.")

        return model

    except Exception as e:

        logger.exception("Model training failed.")

        raise


# ==========================================================
# Save Trained Model
# ==========================================================

def save_model(model, file_path: str) -> None:
    """
    Save trained model as a pickle file.

    Args:
        model: Trained machine learning model.
        file_path (str): Output path.
    """

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logger.info("Model saved successfully at %s", file_path)

    except Exception as e:

        logger.exception("Failed to save model.")

        raise


# ==========================================================
# Main Pipeline
# ==========================================================

def main():
    """
    Execute complete model building pipeline.
    """

    try:

        logger.info("Starting Model Building Pipeline...")

        # Load feature engineered dataset
        train_data = load_data(
            "./data/processed/train_bow.csv"
        )

        # Separate features and labels
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train model
        model = train_model(
            X_train,
            y_train
        )

        # Save trained model
        save_model(
            model,
            "./models/model.pkl"
        )

        logger.info("Model Building Pipeline Completed Successfully.")

    except Exception as e:

        logger.exception("Model Building Pipeline Failed.")

        print(f"Error: {e}")


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()