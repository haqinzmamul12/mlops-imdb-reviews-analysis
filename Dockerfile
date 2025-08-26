# Base image for Airflow
FROM apache/airflow:2.9.0

# Switch to root to install extra dependencies if needed
USER root

# Install system-level dependencies (optional)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential libpq-dev && \
#     rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements.txt (only for dependencies, not DAGs)
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt
