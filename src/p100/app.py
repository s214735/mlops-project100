import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2

os.environ["BACKEND"] = "0.0.0.0"


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/level-oxygen-447714-d3/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "backend":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    files = {
        "file": ("uploaded_image.jpg", image, "image/jpeg")  # (filename, content, MIME type)
    }
    try:
        response = requests.post(predict_url, files=files, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from backend: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
    return None


def fetch_dataloader_info(backend):
    """Fetch dataloader information from the backend."""
    data_url = f"{backend}/data/"
    try:
        response = requests.post(data_url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from backend: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        result = classify_image(uploaded_file, backend=backend)
        if result is not None:
            prediction = result["prediction"]
            st.image(uploaded_file, caption="Uploaded Image")
            st.write("Prediction:", prediction)
            if "probabilities" in result:
                probabilities = result["probabilities"]
                data = {"Class": [f"Class {i}" for i in range(len(probabilities))], "Probability": probabilities}
                df = pd.DataFrame(data)
                df.set_index("Class", inplace=True)
                st.bar_chart(df, y="Probability")
        else:
            st.error("Failed to get prediction.")

    # Add a button to fetch dataloader information
    if st.button("Show Dataloader Info"):
        dataloader_info = fetch_dataloader_info(backend=backend)
        if dataloader_info is not None:
            st.write("### Dataloader Information")
            for key, value in dataloader_info.items():
                st.write(f"**{key.capitalize()} Dataset**")
                st.json(value)
        else:
            st.error("Failed to fetch dataloader information.")


if __name__ == "__main__":
    main()
