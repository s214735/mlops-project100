FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/

# sys.path.append("src/p100")

ENTRYPOINT ["python", "-u", "src/p100/test.py"]