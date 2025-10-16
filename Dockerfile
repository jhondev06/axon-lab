# AXON minimal Dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy minimal files first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Environment for Telegram (optional, to be passed at runtime)
# ENV TELEGRAM_BOT_TOKEN="" \
#     TELEGRAM_CHAT_ID=""

# Create necessary directories (in case volumes not mounted)
RUN mkdir -p data/raw data/processed outputs/artifacts outputs/metrics outputs/ledgers outputs/reports outputs/figures knowledge

# Default command: run full pipeline
CMD ["python", "main.py"]