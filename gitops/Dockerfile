FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Configure git to trust the repository
RUN git config --global --add safe.directory /repo

# Run gitops service
CMD ["python", "-u", "main.py"]
