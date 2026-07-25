# ==========================================================
# Import Required Libraries
# ==========================================================

import json
import logging

import dagshub
import mlflow
from mlflow import MlflowClient


# ==========================================================
# Authenticate with DagsHub
# ==========================================================
# The first time you run this script, DagsHub will ask you
# to authenticate. After successful authentication,
# credentials are stored locally and reused.
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
        "model_registration_errors.log"
    )
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================================================
# Load Experiment Information
# ==========================================================

def load_experiment_info(file_path: str) -> dict:
    """
    Load experiment information saved during model evaluation.

    Args:
        file_path (str): Path of experiment_info.json.

    Returns:
        dict: Experiment information.
    """

    try:

        with open(file_path, "r") as file:
            experiment_info = json.load(file)

        logger.info("Experiment information loaded successfully.")

        return experiment_info

    except Exception:

        logger.exception("Unable to load experiment information.")

        raise


# ==========================================================
# Register Model
# ==========================================================

def register_model(
    model_name: str,
    experiment_info: dict
):
    """
    Register the trained model in the MLflow Model Registry.

    Args:
        model_name (str): Name of the registered model.
        experiment_info (dict): Experiment details.
    """

    try:

        # Construct model URI
        model_uri = (
            f"runs:/{experiment_info['run_id']}/model"
        )

        logger.info("Registering model...")

        # Register Model
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logger.info(
            "Model registered successfully."
        )

        logger.info(
            "Model Version : %s",
            registered_model.version
        )

        # --------------------------------------------------
        # Move Model to Staging
        # --------------------------------------------------

        client = MlflowClient()

        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging"
        )

        logger.info(
            "Model moved to Staging successfully."
        )

    except Exception:

        logger.exception(
            "Model registration failed."
        )

        raise


# ==========================================================
# Main Pipeline
# ==========================================================

def main():
    """
    Execute complete model registration pipeline.
    """

    try:

        logger.info(
            "Starting Model Registration Pipeline..."
        )

        # Load experiment information
        experiment_info = load_experiment_info(
            "./reports/experiment_info.json"
        )

        # Register model
        register_model(
            model_name="sentiment-analysis-model",
            experiment_info=experiment_info
        )

        logger.info(
            "Model Registration Pipeline Completed Successfully."
        )

    except Exception:

        logger.exception(
            "Model Registration Pipeline Failed."
        )


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":

    main()