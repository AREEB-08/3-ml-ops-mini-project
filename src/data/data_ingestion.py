# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import logging
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split


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

    # File Logger (Only ERROR logs)
    file_handler = logging.FileHandler("errors.log")
    file_handler.setLevel(logging.ERROR)

    # Log Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================================================
# Load Parameters from params.yaml
# ==========================================================

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML configuration file.

    Args:
        params_path (str): Path to params.yaml

    Returns:
        dict: Dictionary containing configuration parameters.
    """

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.info("Parameters loaded successfully.")

        return params

    except FileNotFoundError:
        logger.error("Parameter file not found: %s", params_path)
        raise

    except yaml.YAMLError as e:
        logger.error("YAML parsing error: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error while loading parameters: %s", e)
        raise


# ==========================================================
# Load Dataset
# ==========================================================

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        data_url (str): URL or local path of dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    try:
        df = pd.read_csv(data_url)

        logger.info("Dataset loaded successfully.")

        return df

    except pd.errors.ParserError as e:
        logger.error("CSV parsing failed: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error while loading dataset: %s", e)
        raise


# ==========================================================
# Preprocess Dataset
# ==========================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset.

    Steps:
        1. Remove unnecessary columns.
        2. Keep only happiness and sadness tweets.
        3. Convert labels into binary values.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    try:

        # Remove unnecessary column
        df = df.drop(columns=["tweet_id"])

        # Keep only required classes
        sentiments = ["happiness", "sadness"]

        final_df = df[df["sentiment"].isin(sentiments)].copy()

        # Encode target labels
        final_df["sentiment"] = final_df["sentiment"].replace(
            {
                "happiness": 1,
                "sadness": 0
            }
        )

        logger.info("Data preprocessing completed successfully.")

        return final_df

    except KeyError as e:
        logger.error("Missing required column: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise


# ==========================================================
# Save Train and Test Dataset
# ==========================================================

def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save train and test datasets.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        output_path (str): Output directory.
    """

    try:

        raw_data_path = os.path.join(output_path, "raw")

        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(raw_data_path, "train.csv"),
            index=False
        )

        test_data.to_csv(
            os.path.join(raw_data_path, "test.csv"),
            index=False
        )

        logger.info("Train and test datasets saved successfully.")

    except Exception as e:
        logger.error("Failed to save dataset: %s", e)
        raise


# ==========================================================
# Main Function
# ==========================================================

def main():
    """
    Execute the complete data ingestion pipeline.
    """

    try:

        # Load configuration
        params = load_params("params.yaml")

        data_params = params["data_ingestion"]

        test_size = data_params["test_size"]

        dataset_url = data_params["dataset_url"]

        output_path = data_params["output_path"]

        # Load dataset
        df = load_data(dataset_url)

        # Preprocess dataset
        final_df = preprocess_data(df)

        # Split dataset
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        logger.info("Dataset split completed.")

        # Save datasets
        save_data(
            train_data,
            test_data,
            output_path
        )

        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:

        logger.exception("Data ingestion pipeline failed.")

        print(f"Error: {e}")


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()