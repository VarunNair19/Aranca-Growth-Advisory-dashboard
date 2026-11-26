FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for building/running Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (keeps image smaller & faster) then requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create writable directories for SQLite and Hugging Face model cache
RUN mkdir -p /app/data /app/cache /app/logs && chmod -R 777 /app
ENV HF_HOME=/app/cache

EXPOSE 7860

CMD python -m scripts.seed_db ; gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 120