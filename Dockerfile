FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# musl libc required by the pre-built llama-cpp-python wheel (compiled against Alpine/musl)
RUN apt-get update && apt-get install -y --no-install-recommends musl && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir llama-cpp-python==0.3.2 \
        --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/

COPY . .

RUN chmod +x startup.sh

# HF Spaces requires port 7860
EXPOSE 7860

# Downloads models from HF Hub on first start, then launches uvicorn
CMD ["./startup.sh"]
