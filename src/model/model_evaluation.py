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

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ==========================================================
# Global Configuration & Constants
# ==========================================================

REPO_OWNER = "AREEB-08"
REPO_NAME = "3-ml-ops-mini-project"
EXPERIMENT_NAME = "Sentiment Analysis DVC Pipeline v1"
MODEL_NAME = "LogisticRegression"

# ==========================================================
# Configure Logger
# ==========================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:

    # Console Logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Error File Logger
    file_handler = logging.FileHandler("model_evaluation_errors.log")
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
    Load the trained machine learning model from a pickle file.

    Args:
        model_path (str): Path of the saved model file.

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
    Load the processed test dataset from a CSV file.

    Args:
        file_path (str): CSV file path.

    Returns:
        pd.DataFrame: Loaded pandas DataFrame.
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

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the trained model performance on test data.

    Args:
        model: Trained scikit-learn model object.
        X_test: Test feature matrix.
        y_test: True target labels.

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, auc).
    """
    try:
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "auc": roc_auc_score(y_test, probabilities),
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
        metrics (dict): Dictionary of evaluation metrics.
        file_path (str): Output JSON file path.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.info("Metrics saved successfully.")

    except Exception:
        logger.exception("Unable to save metrics.")
        raise


# ==========================================================
# Save Experiment Information
# ==========================================================

def save_experiment_info(run_id: str, model_name: str, file_path: str):
    """
    Save MLflow experiment metadata.

    Args:
        run_id (str): Active MLflow Run ID.
        model_name (str): Name of the evaluated model.
        file_path (str): Output JSON file path.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        experiment = {
            "run_id": run_id,
            "model_name": model_name,
        }

        with open(file_path, "w") as file:
            json.dump(experiment, file, indent=4)

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
        logger.info("Starting Model Evaluation Pipeline...")

        # --------------------------------------------------
        # DagsHub & MLflow Initialization
        # --------------------------------------------------
        logger.info("Initializing DagsHub and MLflow tracking...")

        dagshub.init(
            repo_owner=REPO_OWNER,
            repo_name=REPO_NAME,
            mlflow=True
        )

        mlflow.set_tracking_uri(
            f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
        )

        logger.info("DagsHub initialized successfully.")
        logger.info(f"MLflow Tracking URI : {mlflow.get_tracking_uri()}")

        # Set Descriptive MLflow Experiment Name
        mlflow.set_experiment(EXPERIMENT_NAME)

        # --------------------------------------------------
        # Run Evaluation Workflows inside MLflow Context
        # --------------------------------------------------
        with mlflow.start_run() as run:

            logger.info(f"MLflow Run ID: {run.info.run_id}")

            # Load trained model
            model = load_model("./models/model.pkl")

            # Load processed test dataset
            test_data = load_data("./data/processed/test_bow.csv")

            # Separate Features and Labels
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)

            # Save evaluation metrics locally
            save_metrics(metrics, "./reports/metrics.json")

            # Save experiment info locally
            save_experiment_info(
                run.info.run_id,
                MODEL_NAME,
                "./reports/experiment_info.json"
            )

            # --------------------------------------------------
            # Log Metrics, Params & Artifacts to MLflow
            # --------------------------------------------------

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model parameters
            mlflow.log_params(model.get_params())

           # Log trained model artifact AND register it in one step
            # mlflow.sklearn.log_model(
            #     sk_model=model,
            #     artifact_path="model",
            #     registered_model_name="sentiment-analysis-model"
            # )
#              
            # new ml config so that it will not make new version and ci will not track them 
            mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="model",
                            
                        )
            # Log JSON artifacts
            mlflow.log_artifact("./reports/metrics.json")
            mlflow.log_artifact("./reports/experiment_info.json")

            # Log error file if it was created
            if os.path.exists("model_evaluation_errors.log"):
                mlflow.log_artifact("model_evaluation_errors.log")

            # --------------------------------------------------
            # Summary Output
            # --------------------------------------------------
            logger.info("=" * 60)
            logger.info("Model Evaluation Completed Successfully")
            logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
            logger.info(f"Precision : {metrics['precision']:.4f}")
            logger.info(f"Recall    : {metrics['recall']:.4f}")
            logger.info(f"AUC       : {metrics['auc']:.4f}")
            logger.info(f"Run ID    : {run.info.run_id}")
            logger.info("=" * 60)

    except Exception:
        logger.exception("Model Evaluation Pipeline Failed.")
        raise


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()