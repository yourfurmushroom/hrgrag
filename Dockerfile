FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORTABLE_NO_VENV=1 \
    PYTHON_BIN=python3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install torch && \
    python3 -m pip install -r /app/requirements.txt && \
    python3 -m nltk.downloader punkt

COPY . /app

RUN chmod +x \
    /app/setup_env.sh \
    /app/download_datasets.sh \
    /app/bootstrap_all.sh \
    /app/run_pipeline.sh \
    /app/auto_benchmark.sh \
    /app/run_all_benchmarks.sh \
    /app/docker_entrypoint.sh

ENTRYPOINT ["/app/docker_entrypoint.sh"]
