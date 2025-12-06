# ai-tutor-langgraph/Dockerfile

FROM python:3.12-slim

# Avoid .pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside the container
WORKDIR /app

# System dependencies (for building some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default API host/port (can be overridden by env vars in Azure)
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Start the FastAPI app
CMD ["uvicorn", "ai_tutor.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
