# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import json
import pickle
import logging

import dagshub
import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ==========================================================
# Authenticate with DagsHub
# ==========================================================
# The first time you run this script, it will ask you to
# authenticate with DagsHub. After successful login,
# the credentials are stored locally and reused in
# future executions.
# ==========================================================

dagshub.auth.login()

# ==========================================================
# Configure MLflow Tracking
# ==========================================================

REPO_OWNER = "AREEB-08"
REPO_NAME = "3-ml-ops-mini-project"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)

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
    file_handler = logging.FileHandler(
        "model_evaluation_errors.log"
    )
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
# Load Trained Model
# ==========================================================

def load_model(model_path: str):
    """
    Load the trained machine learning model.

    Args:
        model_path (str): Path of the saved model.

    Returns:
        Trained machine learning model.
    """

    try:

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded successfully.")

        return model

    except Exception:

        logger.exception("Unable to load model.")

        raise


# ==========================================================
# Load Test Dataset
# ==========================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the processed test dataset.

    Args:
        file_path (str): CSV file path.

    Returns:
        pd.DataFrame
    """

    try:

        df = pd.read_csv(file_path)

        logger.info("Test dataset loaded successfully.")

        return df

    except Exception:

        logger.exception("Unable to load dataset.")

        raise


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Args:
        model : Trained model
        X_test : Test features
        y_test : True labels

    Returns:
        Dictionary containing evaluation metrics.
    """

    try:

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {

            "accuracy":
                accuracy_score(y_test, predictions),

            "precision":
                precision_score(y_test, predictions),

            "recall":
                recall_score(y_test, predictions),

            "auc":
                roc_auc_score(
                    y_test,
                    probabilities
                )

        }

        logger.info("Model evaluated successfully.")

        return metrics

    except Exception:

        logger.exception("Model evaluation failed.")

        raise


# ==========================================================
# Save Metrics
# ==========================================================

def save_metrics(metrics: dict, file_path: str):
    """
    Save evaluation metrics into a JSON file.

    Args:
        metrics (dict): Evaluation metrics.
        file_path (str): Output JSON path.
    """

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "w") as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.info("Metrics saved successfully.")

    except Exception:

        logger.exception("Unable to save metrics.")

        raise


# ==========================================================
# Save Experiment Information
# ==========================================================

def save_experiment_info(
    run_id: str,
    model_name: str,
    file_path: str
):
    """
    Save MLflow experiment details.

    Args:
        run_id (str): MLflow Run ID.
        model_name (str): Name of trained model.
        file_path (str): Output JSON path.
    """

    try:

        experiment = {

            "run_id": run_id,
            "model_name": model_name

        }

        with open(file_path, "w") as file:

            json.dump(
                experiment,
                file,
                indent=4
            )

        logger.info("Experiment information saved.")

    except Exception:

        logger.exception("Unable to save experiment information.")

        raise


# ==========================================================
# Main Pipeline
# ==========================================================

def main():
    """
    Execute complete Model Evaluation pipeline.
    """

    try:

        logger.info(
            "Starting Model Evaluation Pipeline..."
        )

        # Create MLflow Experiment
        mlflow.set_experiment("dvc-pipeline")

        # Start MLflow Run
        with mlflow.start_run() as run:

            # Load trained model
            model = load_model(
                "./models/model.pkl"
            )

            # Load processed test dataset
            test_data = load_data(
                "./data/processed/test_bow.csv"
            )

            # Separate Features and Labels
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate model
            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            # Save evaluation metrics
            save_metrics(
                metrics,
                "./reports/metrics.json"
            )

            # Save experiment information
            save_experiment_info(
                run.info.run_id,
                "LogisticRegression",
                "./reports/experiment_info.json"
            )

            # --------------------------------------------------
            # Log Metrics to MLflow
            # --------------------------------------------------

            mlflow.log_metrics(metrics)

            # --------------------------------------------------
            # Log Model Parameters
            # --------------------------------------------------

            mlflow.log_params(
                model.get_params()
            )

            # --------------------------------------------------
            # Log Trained Model
            # --------------------------------------------------

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

            # --------------------------------------------------
            # Log Artifacts
            # --------------------------------------------------

            mlflow.log_artifact(
                "./reports/metrics.json"
            )

            mlflow.log_artifact(
                "./reports/experiment_info.json"
            )

            if os.path.exists(
                "model_evaluation_errors.log"
            ):

                mlflow.log_artifact(
                    "model_evaluation_errors.log"
                )

            logger.info(
                "Model Evaluation Pipeline Completed Successfully."
            )

    except Exception:

        logger.exception(
            "Model Evaluation Pipeline Failed."
        )

        raise


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":

    main()