FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install remaining deps (torch/torchaudio/torchcodec already in base image)
COPY requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face cache directory
ENV HF_HOME=/data/huggingface

# Copy server code
COPY wyoming_granite_stt.py .

EXPOSE 10300

ENTRYPOINT ["python", "wyoming_granite_stt.py"]
