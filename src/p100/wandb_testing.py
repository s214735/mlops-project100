import os
import torch
import wandb
from model import ResNetModel  # Ensure this matches your model definition
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_model(artifact_name, entity, project):
    """
    Load a model checkpoint directly from a W&B artifact.
    """
    # Check if necessary environment variables are set
    if not os.getenv("WANDB_API_KEY"):
        raise EnvironmentError("WANDB_API_KEY environment variable is not set.")
    
    print(f"Loading artifact: {artifact_name} from project {entity}/{project}")

    # Initialize W&B API
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    
    # Construct artifact reference
    artifact_reference = f"{entity}/{project}/{artifact_name}:latest"

    # Fetch artifact
    artifact = api.artifact(artifact_reference)
    artifact_dir = artifact.download()  # Download to a temporary directory

    # Locate the checkpoint file in the artifact
    checkpoint_path = None
    for file_name in artifact.files():
        if file_name.name.endswith(".ckpt"):  # Adjust this to match your checkpoint extension
            checkpoint_path = os.path.join(artifact_dir, file_name.name)
            break

    if not checkpoint_path:
        raise FileNotFoundError("No valid model checkpoint file found in the artifact.")
    
    print(f"Model checkpoint found at: {checkpoint_path}")

    # Load the model from the checkpoint
    return ResNetModel.load_from_checkpoint(checkpoint_path)


def test_model_speed(artifact_name, entity, project, max_time=5):
    """
    Test the model's inference speed.
    """
    print(f"Testing model inference speed for artifact: {artifact_name}")
    model = load_model(artifact_name, entity, project)

    # Create a test input tensor
    test_input = torch.rand(1, 3, 224, 224)  

    # Measure inference time
    start = time.time()
    for _ in range(100):
        model(test_input)  
    end = time.time()
    elapsed_time = end - start

    print(f"Model inference time: {elapsed_time:.2f} seconds")
    assert elapsed_time < max_time, f"Model inference time too slow: {elapsed_time:.2f} seconds"
    print("Model speed test passed!")


# Example usage (adjust to match your environment)
if __name__ == "__main__":
    ENTITY = os.getenv("WANDB_ENTITY")
    PROJECT = os.getenv("WANDB_PROJECT")
    MODEL_NAME = os.getenv("MODEL_NAME")
    print(f"{ENTITY}/{PROJECT}/{MODEL_NAME}:latest\n")
    test_model_speed(MODEL_NAME, ENTITY, PROJECT, max_time=5.0)