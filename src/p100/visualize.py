import wandb
from p100.utils import get_wandb_api_key


def main():
    wandb.login(key=get_wandb_api_key())
    run = wandb.init()
    artifact = run.use_artifact("MLOperations/pokemon_classifier/pokemon_classifier_model:v11", type="model")
    model = artifact.download()
    print(model)
