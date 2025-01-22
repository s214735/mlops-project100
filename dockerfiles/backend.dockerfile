FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/p100

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY src/p100/backend.py /app/backend.py
COPY utils/pokemon-labels.json /app/pokemon-labels.json

COPY src/p100/data.py /app/p100/data.py
COPY src/p100/evaluate.py /app/p100/evaluate.py
COPY src/p100/model.py /app/p100/model.py
COPY src/p100/utils.py /app/p100/utils.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
