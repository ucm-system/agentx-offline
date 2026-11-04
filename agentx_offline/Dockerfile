# AgentX offline test image.
#
# Build context = this directory (agentx_offline/). Prepare the cache first:
#   pwsh scripts/prepare_agentx_cache.ps1
#   bash scripts/prepare_agentx_cache.sh
#
# Build (dataset + tokenizer baked in, ~3.9 GB):
#   docker build -t agentx-offline:1.0.7 .
# Slim variant (mount /opt/hf at runtime instead):
#   docker build --build-arg WITH_CACHE=0 -t agentx-offline:slim .
#
# Cross-build for ARM64 (agentx deps include `crick`, which has no arm64 Linux
# wheel, so the builder stage installs a Rust toolchain and compiles it; the
# final stage only installs prebuilt wheels):
#   docker buildx build --platform linux/arm64 --build-arg BASE_IMAGE=llmperf:arm64 \
#     -t agentx-offline:1.0.7-arm64 .
ARG BASE_IMAGE=python:3.11-slim

# ---------- wheelhouse (needs cargo for crick on arm64) ----------
FROM ${BASE_IMAGE} AS wheels
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} PIP_NO_CACHE_DIR=1
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/*.sources /etc/apt/sources.list 2>/dev/null || true; \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential cargo rustc && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip wheel --no-cache-dir -r /tmp/requirements.txt -w /tmp/wheels

# ---------- runtime ----------
FROM ${BASE_IMAGE}
ARG WITH_CACHE=1
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

WORKDIR /opt/agentx

COPY --from=wheels /tmp/wheels /tmp/wheels
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-index --find-links /tmp/wheels -r /tmp/requirements.txt && \
    python -c "import agentx_cli, aiperf; print('agentx-harness ready')" && \
    rm -rf /tmp/wheels

COPY runner.py report.py entrypoint.sh prefetch_cache.py /opt/agentx/
COPY assets /opt/agentx/assets
COPY patch_aiperf_tokenizer.py /tmp/patch_aiperf_tokenizer.py
RUN chmod +x /opt/agentx/entrypoint.sh && python /tmp/patch_aiperf_tokenizer.py

# Dataset Arrow cache + DeepSeek-V3 tokenizer (built by the prepare script).
COPY --chown=10001:10001 cache/ /opt/hf/
RUN if [ "$WITH_CACHE" != "1" ]; then rm -rf /opt/hf/datasets; fi

RUN useradd --create-home --uid 10001 agentx && \
    mkdir -p /results && chown -R agentx:agentx /results /opt/agentx
USER agentx
VOLUME ["/results"]

ENTRYPOINT ["/opt/agentx/entrypoint.sh"]
