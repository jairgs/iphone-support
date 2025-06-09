# Base image
FROM python:3.11-slim

# Reduce size: avoid cache & install only what’s needed
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app
COPY . .

# Expose Gradio’s default port
EXPOSE 8502

# Run the app
CMD ["python", "chatbot_gradio.py"]
