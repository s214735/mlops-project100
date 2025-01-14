import wandb
import os
import time
import torch
from src.p100.model import ResNetModel

def load_model(artifact):
    logdir = "../../models"
    model_checkpoint = f"{artifact}:latest"
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    return ResNetModel.load_from_checkpoint(f"{logdir}/{file_name}")

def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1