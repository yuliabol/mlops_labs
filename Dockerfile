# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into a local directory
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final
FROM python:3.11-slim

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
# Copy DVC files if needed for dvc repro inside container
COPY dvc.yaml dvc.lock .dvcignore ./
COPY .dvc/ ./.dvc/

# Set Python path to include the current directory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command (can be overridden by Airflow)
CMD ["python", "src/train.py", "data/prepared/train.csv", "data/prepared/test.csv", "data/models"]
