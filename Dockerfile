FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/checkpoints /app/logs

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/checkpoints/best_model
ENV MODEL_NAME=google/vit-base-patch16-224
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
