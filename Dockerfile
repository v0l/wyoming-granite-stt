FROM nvcr.io/nvidia/pytorch:24.12-py3
ARG CUDA=126

WORKDIR /app

# System deps (ffmpeg needed by torchcodec for audio load)
RUN apt update && apt install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# App deps
COPY requirements-docker.txt ./requirements.txt
# Uninstall pre-installed torchaudio (wrong CUDA version) and install correct one
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y torchaudio && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu${CUDA}

# Hugging Face cache
ENV HF_HOME=/data/huggingface

COPY wyoming_granite_stt.py .

EXPOSE 10300

ENTRYPOINT ["python", "wyoming_granite_stt.py"]
