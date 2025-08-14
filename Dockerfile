# Use the latest official Python slim image
FROM python:3.11-slim-bullseye

WORKDIR /app

# Upgrade pip and install system deps
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Run the Uvicorn server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]