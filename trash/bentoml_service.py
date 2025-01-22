from __future__ import annotations

import bentoml
import numpy as np
from onnxruntime import InferenceSession


@bentoml.service(resources={"cpu": "1"}, traffic={"timeout": 10})

class ImageClassifierService:
    """Image classifier service using ONNX model."""

    def __init__(self) -> None:
        self.model = InferenceSession("resnet18.onnx")

    @bentoml.api
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict the class of the input image."""
        output = self.model.run(None, {"input": image.astype(np.float32)})
        return output[0]


# Create an instance of the service and assign it to `svc`
svc = ImageClassifierService()

