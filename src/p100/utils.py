import json

from google.cloud import secretmanager


def get_wandb_api_key() -> str:
    """
    Retrieve the W&B API key from Google Secret Manager.
    """
    client = secretmanager.SecretManagerServiceClient()

    # Construct the secret name
    secret_name = "wandb-api-key"
    project_id = "level-oxygen-447714-d3"
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

    # Access the secret value
    response = client.access_secret_version(name=secret_path)
    secret_payload = response.payload.data.decode("UTF-8")

    # Parse the JSON and extract the key
    secret_dict = json.loads(secret_payload)
    api_key = secret_dict.get("wandb-api-key")

    return api_key
