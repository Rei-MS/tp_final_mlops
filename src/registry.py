"""MLflow model registry utilities."""

from mlflow.tracking import MlflowClient

from src.config import MODEL_NAME, TRACKING_URI


def assign_champion_alias(run_id: str) -> str:
    """Finds the registered model version from a run ID and sets the 'champion' alias.

    Args:
        run_id: MLflow run ID associated with the registered model artifact.

    Returns:
        str: The registered model version string assigned as champion.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)

    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    matching_versions = [v for v in model_versions if v.run_id == run_id]

    registered_version = max(matching_versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(
        name=MODEL_NAME, alias="champion", version=registered_version.version
    )

    return str(registered_version.version)
