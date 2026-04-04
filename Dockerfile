# Use Python 3.11 slim — smaller image, faster builds
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
# So if your code changes but requirements don't, it won't reinstall everything
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create directories that are gitignored but needed at runtime
RUN mkdir -p data models

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501
