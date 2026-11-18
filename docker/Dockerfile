# HeisenBig shared dev image
#   Python 3.12 + Java 17 + PySpark + PyTorch (CPU) + Ray + endf + JupyterLab
# Multi-arch: builds natively on Apple Silicon (arm64) and Intel/AMD (amd64).
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Java 17 is required by Spark 4.x
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless procps curl ca-certificates git tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /usr/lib/jvm/default-java-17

ENV JAVA_HOME=/usr/lib/jvm/default-java-17

# PyTorch CPU wheels (avoids pulling multi-GB CUDA wheels on amd64)
ARG TORCH_VERSION=2.15.0
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Non-root user (UID 1000 matches the default user on most Linux hosts; Docker Desktop
# on macOS maps ownership transparently).
RUN useradd -m -u 1000 -s /bin/bash dev && mkdir -p /workspace && chown dev:dev /workspace
USER dev
WORKDIR /workspace

ENV PYTHONPATH=/workspace/src \
    PYSPARK_PYTHON=python \
    PYSPARK_DRIVER_PYTHON=python \
    RAY_USAGE_STATS_ENABLED=0

# 8888 JupyterLab | 4040 Spark UI | 8265 Ray dashboard
EXPOSE 8888 4040 8265

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/workspace"]
