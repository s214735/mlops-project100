"""API for the MNIST model."""

import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

# from p100.data import default_img_transform
# from p100.model import load_from_checkpoint
# from p100.utils import HydraRichLogger
from .hello import helloWorld

load_dotenv()

# logger = HydraRichLogger(level=os.getenv("LOG_LEVEL", "INFO"))

models = {}
data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup and clean up at shutdown.
    logger.info(f"Loading model from {os.getenv('MODEL_CHECKPOINT')}...")
    if os.getenv("MODEL_CHECKPOINT") is None:
        logger.error("No model checkpoint found.")
        sys.exit(1)

    mnist_model = load_from_checkpoint(os.getenv("MODEL_CHECKPOINT"), logdir="models")
    models["mnist"] = mnist_model
    logger.info("Model loaded.")

    yield  # Wait for the application to finish

    logger.info("Cleaning up...")
    del mnist_model
    """
    try:
        hello = helloWorld()
        data["hello"] = hello

        yield

    except asyncio.CancelledError:
        # Handle the cancellation gracefully during shutdown
        print("Lifespan shutdown was cancelled.")
    finally:
        # Perform cleanup tasks
        if "hello" in data:
            del data["hello"]
        print("Lifespan cleanup complete.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    """Root endpoint of the API."""
    return {"message": "Welcome to the MNIST model inference API!"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/hello")
def hello():
    return {"data": data["hello"]}


@app.get("/items/{item_id}")
def getitem(item_id: int):
    return {"item_id": item_id}


"""
@app.get("/modelstats")
def modelstats():

    return {
        "model architecture": str(models["mnist"]),
    }


@app.post("/predict")
async def predict(image: bytes = File(...)):

    pil_image = Image.open(BytesIO(image))
    image_data = pil_to_tensor(pil_image)
    logger.info("Image loaded.")
    input_tensor = default_img_transform(image_data)
    probs, preds = models["mnist"].inference(input_tensor)

    # Return the predicted label
    return {"prediction": int(preds[0]), "probabilities": probs[0].tolist()}
"""
