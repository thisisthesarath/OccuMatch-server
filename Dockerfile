# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies for pandas, pyarrow, faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Set PORT env var (Cloud Run sets it automatically)
ENV PORT=8080

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
