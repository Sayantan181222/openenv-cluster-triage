# Use a highly specific, updated tag rather than just 'slim'
FROM python:3.10.14-slim-bookworm

WORKDIR /app

# THE FIX: Force Debian to install the latest security patches
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install your Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your environment and agent code
COPY . .

# Run the baseline agent
CMD ["python", "-u", "inference.py"]