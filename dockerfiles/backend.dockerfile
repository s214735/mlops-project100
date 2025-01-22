FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_backend.txt /app/requirements_backend.txt
COPY src/p100/backend.py /app/p100/backend.py
COPY utils/imagenet-simple-labels.json /app/imagenet-simple-labels.json
COPY src/p100/data.py /app/data.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt

EXPOSE 8080

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
