FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration and dependencies
COPY pyproject.toml ./

# Install python dependencies (installing PyTorch CPU-only to keep image size small)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn pydantic pandas numpy scikit-learn

# Copy application source code and model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Cloud Run dynamically assigns a PORT environment variable
ENV PORT=8080

# Expose port and run uvicorn
EXPOSE 8080
CMD sh -c "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"