# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# 1. Install system dependencies for OpenCV (Corrected)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install CPU-only PyTorch & Torchvision (Fast, ~200MB instead of 2.5GB)
# We do this BEFORE copying requirements to cache this layer
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 4. Copy the source code
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the API (with quotes fixed)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]