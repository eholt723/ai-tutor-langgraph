FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps – now including compiler + cmake for llama-cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Use runtime requirements
COPY requirements-runtime.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "ai_tutor.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
