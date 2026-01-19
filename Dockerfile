ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0.0

# Build stage
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md LICENSE ./
COPY cka/ ./cka/

ARG PYTORCH_VERSION
RUN pip install torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu && \
    pip install .

# Runtime stage
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --shell /bin/bash cka
USER cka

CMD ["python", "-c", "from pytorch_cka import CKA; print('pytorch-cka is ready! Import with: from pytorch_cka import CKA')"]
