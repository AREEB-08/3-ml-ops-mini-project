# ==========================================================
# Import Required Libraries
# ==========================================================

import json
import logging

import dagshub
import mlflow
from mlflow import MlflowClient

# ==========================================================
# Global Configuration & Constants
# ==========================================================

REPO_OWNER = "AREEB-08"
REPO_NAME = "3-ml-ops-mini-project"
REGISTERED_MODEL_NAME = "sentiment-analysis-model"

# ==========================================================
# Configure Logger
# ==========================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("model_registration_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================================================
# Register Model Function
# ==========================================================

def register_model(model_name: str, experiment_info: dict):
    """
    Register the logged model from the run into MLflow Registry and set alias.
    """
    try:
        run_id = experiment_info["run_id"]
        model_uri = f"runs:/{run_id}/model"

        logger.info(f"Registering model from run_id '{run_id}' as '{model_name}'...")

        # Register the model version
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        version = registered_model.version
        logger.info(f"Model registered successfully! Assigned Version: {version}")

        # Set Alias (@staging)
        client = MlflowClient()
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="staging",
                version=version
            )
            logger.info(f"Assigned alias '@staging' to model version {version}.")
        except Exception as e:
            logger.warning(f"Failed to set alias: {e}")

    except Exception:
        logger.exception("Model registration failed.")
        raise


# ==========================================================
# Main Pipeline
# ==========================================================

def main():
    """
    Execute complete model registration pipeline stage.
    """
    try:
        logger.info("Starting Model Registration Pipeline Stage...")

        # Initialize DagsHub & MLflow tracking
        dagshub.init(
            repo_owner=REPO_OWNER,
            repo_name=REPO_NAME,
            mlflow=True
        )

        mlflow.set_tracking_uri(
            f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
        )

        logger.info("DagsHub initialized successfully.")

        # Read run_id from reports/experiment_info.json
        with open("./reports/experiment_info.json", "r") as f:
            experiment_info = json.load(f)

        # Register the model
        register_model(REGISTERED_MODEL_NAME, experiment_info)

        logger.info("=" * 60)
        logger.info("Model Registration Pipeline Completed Successfully")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Model Registration Pipeline Failed.")
        raise


if __name__ == "__main__":
    main()