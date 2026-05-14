FROM python:3.12-slim

ARG CUDA=130

WORKDIR /app

# System deps (ffmpeg needed by torchcodec for audio load)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu${CUDA}

# App deps
COPY requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face cache
ENV HF_HOME=/data/huggingface

COPY wyoming_granite_stt.py .

EXPOSE 10300

ENTRYPOINT ["python", "wyoming_granite_stt.py"]
