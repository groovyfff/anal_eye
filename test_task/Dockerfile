# =============================================================================
# A.E. Brain runtime image
# Target hardware: 4x NVIDIA Tesla P100 (Pascal, sm_60) -> CUDA 11.8 + fp16
# =============================================================================
# CUDA 11.8 has first-class Pascal support and matches the torch cu118 wheels.
# The cudnn8 runtime image is enough for inference; we add build-essential only
# to compile the TA-Lib C library.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Pascal P100 = sm_60. Build/select kernels for the whole P100/P40 era.
    TORCH_CUDA_ARCH_LIST="6.0;6.1"

# Ubuntu 22.04 ships Python 3.10, which satisfies requires-python >=3.10,<3.13.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        build-essential wget ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# TA-Lib C library 0.4.0 (the Python TA-Lib==0.4.28 wheel binds against it).
# ---------------------------------------------------------------------------
RUN cd /tmp \
    && wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    # NOTE: build serially. TA-Lib 0.4.0's `gen_code` tool has a dependency-file
    # race under `make -j` (mv: cannot stat '.deps/gen_code-gen_code.Tpo').
    && make \
    && make install \
    && cd / && rm -rf /tmp/ta-lib*
ENV LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH}

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies. Order matters:
#   1) numpy<2 FIRST so TA-Lib binds to the correct C-ABI.
#   2) PyTorch CUDA 11.8 build (Pascal-compatible) from the torch index.
#   3) the rest from requirements.txt.
# ---------------------------------------------------------------------------
# CRITICAL: pin numpy<2 for *build isolation* too. PEP 517 builds (e.g. the
# TA-Lib 0.4.28 wheel) otherwise pull the latest numpy 2.x to compile against,
# producing C-API errors (NPY_DEFAULT/NPY_C_CONTIGUOUS undeclared) and an
# ABI-incompatible extension. PIP_CONSTRAINT is honoured by isolated builds.
RUN echo "numpy<2.0,>=1.26.4" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install "numpy<2.0,>=1.26.4"

RUN python3 -m pip install \
        torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt ./
# torch is already installed from the cu118 index above; drop the generic line
# so pip doesn't pull a CPU wheel over it.
RUN grep -v -E '^torch' requirements.txt > /tmp/reqs.txt \
    && python3 -m pip install -r /tmp/reqs.txt

# ---------------------------------------------------------------------------
# Project source + editable install (provides the `ae-brain` console script).
# ---------------------------------------------------------------------------
COPY pyproject.toml README.md ./
COPY ae_brain ./ae_brain
COPY tests ./tests
RUN python3 -m pip install -e . --no-deps

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Artifacts (trained models) live here; mount a volume to persist them.
ENV AEB_MODEL_ARTIFACTS_DIR=/app/artifacts
RUN mkdir -p /app/artifacts /app/data

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["run"]
