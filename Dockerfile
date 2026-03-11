FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Disable all optional llama.cpp features — minimizes compile time significantly
ENV CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF \
    -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
    -DGGML_F16C=OFF -DGGML_FMA=OFF"

WORKDIR /app

# Minimal build deps — no libopenblas-dev (BLAS is disabled above)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir llama-cpp-python

COPY . .

RUN chmod +x startup.sh

# HF Spaces requires port 7860
EXPOSE 7860

# Downloads models from HF Hub on first start, then launches uvicorn
CMD ["./startup.sh"]
