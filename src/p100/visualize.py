import wandb


def main():
    run = wandb.init()
    artifact = run.use_artifact("MLOperations/pokemon_classifier/pokemon_classifier_model:v11", type="model")
    model = artifact.download()
    print(model)
