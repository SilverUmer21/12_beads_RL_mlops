# syntax=docker/dockerfile:1.7
# BarahAI inference server: FastAPI + ONNX Runtime, with the model baked in.
#
# Two stages: the first installs the package and its serving dependencies into a
# virtual environment; the second copies only that environment and the model
# files into a clean slim image. No PyTorch, no build tools, no source tree.
#
#   docker build -t barahai-api .          (needs models/serve: `dvc pull models/serve`)

ARG PYTHON_IMAGE=python:3.12-slim-trixie

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

FROM ${PYTHON_IMAGE}
ARG MODEL_VERSION=unknown
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="barahai-api" \
      org.opencontainers.image.description="Barah Goti RL model inference server" \
      org.opencontainers.image.source="https://github.com/SilverUmer21/12_beads_RL_mlops" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      ai.barahai.model.version="${MODEL_VERSION}"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BARAHAI_MODEL_DIR=/app/models \
    BARAHAI_DB=/data/games.db
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app:app /data
COPY --from=build /opt/venv /opt/venv
COPY models/serve /app/models
USER app
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status != 200)"]
# One worker: Prometheus metrics live in process memory. Requests run in a thread pool.
CMD ["uvicorn", "barahai.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
