FROM python:3.12-slim

# FFmpeg provides ffprobe/ffmpeg, used to validate uploads before transcription.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home app
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

USER app
# Model weights are downloaded to this cache on first start. Mount a volume here to keep them.
ENV HF_HOME=/home/app/.cache/huggingface

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
