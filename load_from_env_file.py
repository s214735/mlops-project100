from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.getenv("WANDB_API_KEY")
project = os.getenv("WANDB_PROJECT")
entity = os.getenv("WANDB_ENTITY")
model_name = os.getenv("MODEL_NAME")

print(f"API Key: {api_key}")
print(f"Project: {project}")
print(f"Entity: {entity}")
print(f"Model Name: {model_name}")
