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

    # Console Logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Error File Logger
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
# Load Experiment Information
# ==========================================================

def load_experiment_info(file_path: str) -> dict:
    """
    Load experiment information saved during model evaluation.

    Args:
        file_path (str): Path of experiment_info.json.

    Returns:
        dict: Experiment information dictionary containing run_id and model_name.
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
# Register Model & Assign Stage / Alias
# ==========================================================

def register_model(model_name: str, experiment_info: dict):
    """
    Register the trained model in the MLflow Model Registry and assign staging alias.

    Args:
        model_name (str): Name of the registered model in MLflow.
        experiment_info (dict): Dictionary containing run_id details.
    """
    try:
        run_id = experiment_info["run_id"]
        model_uri = f"runs:/{run_id}/model"

        logger.info(f"Registering model from run_id '{run_id}' as '{model_name}'...")

        # Register Model in MLflow Registry
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        version = registered_model.version
        logger.info(f"Model registered successfully. Version: {version}")

        # Initialize MLflow Client
        client = MlflowClient()

        # --------------------------------------------------
        # Assign Staging Alias / Stage Transition
        # --------------------------------------------------
        
        # Modern MLflow Alias System (Recommended)
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="staging",
                version=version
            )
            logger.info(f"Assigned alias '@staging' to model version {version}.")
        except Exception as e:
            logger.warning(f"Could not set alias: {e}")

        # Legacy Stage Transition (Fallback for older server UI views)
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            logger.info(f"Model version {version} moved to Stage 'Staging'.")
        except Exception as e:
            logger.warning(f"Stage transition skipped or failed: {e}")

    except Exception:
        logger.exception("Model registration failed.")
        raise


# ==========================================================
# Main Pipeline
# ==========================================================

def main():
    """
    Execute complete model registration pipeline.
    """
    try:
        logger.info("Starting Model Registration Pipeline...")

        # --------------------------------------------------
        # DagsHub & MLflow Initialization
        # Replaces deprecated dagshub.auth.login()
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

        # Load experiment metadata generated in evaluation stage
        experiment_info = load_experiment_info("./reports/experiment_info.json")

        # Register model to registry
        register_model(
            model_name=REGISTERED_MODEL_NAME,
            experiment_info=experiment_info
        )

        logger.info("=" * 60)
        logger.info("Model Registration Pipeline Completed Successfully")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Model Registration Pipeline Failed.")
        raise


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()