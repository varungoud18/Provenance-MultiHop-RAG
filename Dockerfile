FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling C++ libraries (FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose default port
EXPOSE 8000

# Set environment defaults
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Start FastAPI server
CMD ["sh", "-c", "uvicorn src.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
