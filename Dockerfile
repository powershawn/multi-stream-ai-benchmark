FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY benchmark/ ./benchmark/
COPY tests/ ./tests/
COPY config.json ./
RUN python -m unittest discover -s tests -v
RUN mkdir -p /tmp/ultralytics
ENTRYPOINT ["python", "-m", "benchmark"]
CMD ["run", "--config", "/app/config.json"]
