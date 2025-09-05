FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Make sure /app/data exists for mounting
RUN mkdir -p /app/data

CMD ["python", "src/dashboard.py"]
