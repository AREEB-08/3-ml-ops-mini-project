# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import pickle
import logging
import yaml
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


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
    file_handler = logging.FileHandler("feature_engineering_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================================================
# Load Parameters
# ==========================================================

def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from params.yaml.

    Args:
        params_path (str): Path of params.yaml.

    Returns:
        dict: Configuration parameters.
    """

    try:

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.info("Parameters loaded successfully.")

        return params

    except FileNotFoundError:

        logger.error("Parameter file not found.")

        raise

    except yaml.YAMLError as e:

        logger.error("YAML parsing error: %s", e)

        raise

    except Exception as e:

        logger.exception("Unexpected error while loading parameters.")

        raise


# ==========================================================
# Load Processed Dataset
# ==========================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load processed dataset.

    Args:
        file_path (str): CSV file path.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """

    try:

        df = pd.read_csv(file_path)

        df = df.fillna("")

        logger.info("Dataset loaded from %s", file_path)

        return df

    except pd.errors.ParserError as e:

        logger.error("CSV parsing failed: %s", e)

        raise

    except Exception as e:

        logger.exception("Error while loading dataset.")

        raise


# ==========================================================
# Apply Bag of Words
# ==========================================================

def apply_bow(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
):
    """
    Apply CountVectorizer (Bag of Words).

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        max_features (int): Maximum vocabulary size.

    Returns:
        tuple:
            train_df (pd.DataFrame)
            test_df (pd.DataFrame)
    """

    try:

        logger.info("Applying Bag of Words...")

        vectorizer = CountVectorizer(
            max_features=max_features
        )

        # Split features and labels
        X_train = train_data["content"]
        y_train = train_data["sentiment"]

        X_test = test_data["content"]
        y_test = test_data["sentiment"]

        # Learn vocabulary from train data
        X_train_bow = vectorizer.fit_transform(X_train)

        # Transform test data
        X_test_bow = vectorizer.transform(X_test)

        # Convert sparse matrix into dataframe
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train.values

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test.values

        # Save vectorizer
        os.makedirs("models", exist_ok=True)

        with open("models/vectorizer.pkl", "wb") as file:
            pickle.dump(vectorizer, file)

        logger.info("Vectorizer saved successfully.")

        logger.info("Feature Engineering completed.")

        return train_df, test_df

    except Exception as e:

        logger.exception("Bag of Words transformation failed.")

        raise


# ==========================================================
# Save Feature Engineered Dataset
# ==========================================================

def save_data(df: pd.DataFrame, file_path: str):
    """
    Save transformed dataframe.

    Args:
        df (pd.DataFrame): Dataframe to save.
        file_path (str): Destination path.
    """

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        df.to_csv(
            file_path,
            index=False
        )

        logger.info("Dataset saved at %s", file_path)

    except Exception as e:

        logger.exception("Error while saving dataset.")

        raise


# ==========================================================
# Main Pipeline
# ==========================================================

def main():
    """
    Execute complete feature engineering pipeline.
    """

    try:

        logger.info("Starting Feature Engineering Pipeline...")

        # Load configuration
        params = load_params("params.yaml")

        max_features = params["feature_engineering"]["max_features"]

        # Load processed datasets
        train_data = load_data(
            "./data/interim/train_processed.csv"
        )

        test_data = load_data(
            "./data/interim/test_processed.csv"
        )

        # Apply Bag of Words
        train_df, test_df = apply_bow(
            train_data,
            test_data,
            max_features
        )

        # Save transformed datasets
        save_data(
            train_df,
            "./data/processed/train_bow.csv"
        )

        save_data(
            test_df,
            "./data/processed/test_bow.csv"
        )

        logger.info("Feature Engineering Pipeline Completed Successfully.")

    except Exception as e:

        logger.exception("Feature Engineering Pipeline Failed.")

        print(f"Error: {e}")


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()