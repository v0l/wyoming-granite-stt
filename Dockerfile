FROM nvcr.io/nvidia/pytorch:26.04-py3
ARG CUDA=132

WORKDIR /app

# System deps (ffmpeg needed by torchcodec for audio load)
RUN apt update && apt install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# App deps
COPY requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu${CUDA} && \
    pip install --no-cache-dir --no-build-isolation git+https://github.com/pytorch/audio.git@main#egg=torchaudio

# Hugging Face cache
ENV HF_HOME=/data/huggingface

COPY wyoming_granite_stt.py .

EXPOSE 10300

ENTRYPOINT ["python", "wyoming_granite_stt.py"]
