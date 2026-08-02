# ==========================================================
# Base Image
# ==========================================================
# Use the official lightweight Python 3.12 image.
# The "slim" variant significantly reduces the final image size.
# ==========================================================

FROM python:3.12-slim


# ==========================================================
# Metadata (Optional)
# ==========================================================

LABEL maintainer="Mohammad Areeb Uddin"
LABEL project="Sentiment Analysis MLOps"
LABEL description="Flask Inference Application"


# ==========================================================
# Environment Variables
# ==========================================================
# PYTHONDONTWRITEBYTECODE
#   Prevents Python from creating .pyc files.
#
# PYTHONUNBUFFERED
#   Sends logs directly to the terminal without buffering.
# ==========================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# ==========================================================
# Working Directory
# ==========================================================
# All subsequent commands execute inside /app.
# ==========================================================

WORKDIR /app


# ==========================================================
# Install System Dependencies
# ==========================================================
# gcc is required because some Python packages may compile
# native extensions during installation.
#
# After installation, apt cache is removed to reduce image size.
# ==========================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*


# ==========================================================
# Copy Requirements File
# ==========================================================
# Copy only requirements.txt first.
# Docker caches this layer, so dependencies are not
# reinstalled unless requirements.txt changes.
# ==========================================================

COPY requirements.txt .


# ==========================================================
# Install Python Dependencies
# ==========================================================

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt


# ==========================================================
# Download NLTK Resources
# ==========================================================
# Download during image build instead of every container start.
# ==========================================================

RUN python -c "import nltk; \
nltk.download('stopwords'); \
nltk.download('wordnet')"


# ==========================================================
# Copy Application Source Code
# ==========================================================
# Copies:
#
# app.py
# templates/
# models/
#
# into the container.
# ==========================================================

COPY . .


# ==========================================================
# Expose Application Port
# ==========================================================

EXPOSE 5000


# ==========================================================
# Start Flask Application
# ==========================================================
# Gunicorn is used instead of Flask's development server.
#
# 1 Workers
# Bind to all interfaces on port 5000.
# ==========================================================

CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "app:app"]