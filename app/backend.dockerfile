FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY app/requirements_backend.txt /app/requirements_backend.txt
COPY app/backend.py /app/backend.py
COPY app/imagenet-simple-labels.json /app/imagenet-simple-labels.json
COPY app/src/data.py /app/src/data.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt

EXPOSE 8080

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
