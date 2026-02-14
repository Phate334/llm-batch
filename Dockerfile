# An example using multi-stage image builds to create a final image without uv.

# First, build the application in the `/app` directory.
# See `Dockerfile` for details.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ARG VLLM_VERSION=v0.15.1
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV VLLM_VERSION=${VLLM_VERSION}

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /tmp/vllm-requirements \
    && curl -fsSL \
    "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_VERSION}/requirements/common.txt" \
    -o /tmp/vllm-requirements/common.txt \
    && curl -fsSL \
    "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_VERSION}/requirements/cpu.txt" \
    -o /tmp/vllm-requirements/cpu.txt \
    && uv pip install -r /tmp/vllm-requirements/cpu.txt \
    && uv pip install --no-deps "vllm[bench]==${VLLM_VERSION}"


# Then, use a final image without uv
FROM python:3.13-slim-bookworm
# It is important to use the image that matches the builder, as the path to the
# Python executable must be the same, e.g., using `python:3.11-slim-bookworm`
# will fail.

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
    && useradd --system --gid 999 --uid 999 --create-home nonroot

# Copy the application from the builder
COPY --from=builder --chown=nonroot:nonroot /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Use the non-root user to run our application
USER nonroot

# Use `/app` as the working directory
WORKDIR /app

CMD ["mitmdump", "-s", "./src/openai_logger.py"]
