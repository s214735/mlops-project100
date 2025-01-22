import json
import os
from contextlib import asynccontextmanager

import anyio
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms

from p100.data import PokeDataset

BUCKET_NAME = "mlops_bucket100"
PREFIX = "data/processed/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, imagenet_classes
    # Load model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    async with await anyio.open_file("app/imagenet-simple-labels.json") as f:
        file_content = await f.read()
        imagenet_classes = json.loads(file_content)

    yield

    # Clean up
    del model
    del transform
    del imagenet_classes


app = FastAPI(lifespan=lifespan)


def predict_image(image_path: str) -> str:
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    _, predicted_idx = torch.max(output, 1)
    return output.softmax(dim=-1), imagenet_classes[predicted_idx.item()]


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        # Ensure the 'temp' directory exists
        os.makedirs("temp", exist_ok=True)

        # Save the file temporarily
        temp_path = f"temp/{file.filename}"
        contents = await file.read()
        async with await anyio.open_file(temp_path, "wb") as f:
            await f.write(contents)

        # Predict the image
        probabilities, prediction = predict_image(temp_path)

        # Return the response
        return {
            "filename": file.filename,
            "prediction": prediction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/data/")
async def show_data():
    """
    Return dataset information instead of printing.
    """
    try:
        # Initialize datasets
        train_dataset = PokeDataset(BUCKET_NAME, mode="train", transform=transforms.ToTensor())
        test_dataset = PokeDataset(BUCKET_NAME, mode="test", transform=transforms.ToTensor())
        val_dataset = PokeDataset(BUCKET_NAME, mode="val", transform=transforms.ToTensor())

        # Compute statistics
        def dataset_stats(dataset):
            return {
                "num_images": len(dataset),
                "image_shape": list(dataset[0][0].shape) if len(dataset) > 0 else None,
                "num_classes": len(np.unique(dataset.targets)),
                "min_label": min(dataset.targets),
                "max_label": max(dataset.targets),
            }

        train_stats = dataset_stats(train_dataset)
        test_stats = dataset_stats(test_dataset)
        val_stats = dataset_stats(val_dataset)

        # Return as JSON
        return {
            "train": train_stats,
            "test": test_stats,
            "val": val_stats,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))  # Use the PORT env variable or default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)
