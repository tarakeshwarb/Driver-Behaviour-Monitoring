FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for C-extensions (e.g. hmmlearn, scipy) and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code and trained model
COPY . .

# Expose web console port
EXPOSE 8000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Launch FastAPI & static dashboard console
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
