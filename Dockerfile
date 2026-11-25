# syntax=docker/dockerfile:1
#
# Edge0 inference server (edge0-infer + the edge0-strata CUDA engine) for Linux NVIDIA GPUs.
# Upstream project: https://github.com/Edge0-AI/Edge0  (Apache-2.0)
#
# Build:
#   docker build -t edge0-infer:cuda13 .
#   docker build -t edge0-infer:cuda13 --build-arg CUDA_ARCHITECTURES="89;120" .
#
# Build steps follow edge0-strata/EDGE0.md "Build (Linux, NVIDIA)" and
# edge0-infer/docs/getting-started.md in the upstream repository.

ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=24.04

############################ build stage ############################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# Upstream repository and revision (branch, tag or full commit SHA).
ARG EDGE0_REPO=https://github.com/Edge0-AI/Edge0.git
ARG EDGE0_REF=main
# llama.cpp commit that provides ggml for edge0-strata (given in edge0-strata/EDGE0.md).
# If you change EDGE0_REF, check EDGE0.md at that revision for the matching commit.
ARG LLAMA_CPP_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
# CUDA compute capabilities to compile for, separated by ";" (sm_80 or newer).
# 120 = RTX 50-series (Blackwell), 89 = RTX 40-series, 86 = RTX 30-series, 90 = H100, 80 = A100.
# "native" can't be used: no GPU is visible during `docker build`. Each extra one adds build time.
ARG CUDA_ARCHITECTURES=120

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates cmake ninja-build build-essential pkg-config \
      python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/edge0
RUN git init -q . \
    && git remote add origin "${EDGE0_REPO}" \
    && git fetch -q --depth 1 origin "${EDGE0_REF}" \
    && git checkout -q FETCH_HEAD \
    && git log -1 --format='Edge0 revision: %H (%cd)'

# edge0-strata: the CUDA engine that runs Edge0 145B and Edge0 120B.
# The switch is STRATA_ENABLE_CUDA (GGML_CUDA has no effect here).
WORKDIR /opt/edge0/edge0-strata
RUN mkdir -p third_party/llama.cpp \
    && curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.tar.gz" \
       | tar -xz --strip-components=1 -C third_party/llama.cpp \
    && cmake -S . -B build -G Ninja \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
         -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
         -DSTRATA_ENABLE_CUDA=ON \
         -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
    && cmake --build build --target strata -j"$(nproc)" \
    && test -x build/strata

# The engine's Python server environment (edge0-infer's default: edge0-strata/.venv/bin/python).
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -q -r requirements.txt

# edge0-infer: the CLI and OpenAI/Anthropic-compatible server. The cpu preset is the
# documented Linux build; for 145B/120B it hands the GPU work to edge0-strata.
WORKDIR /opt/edge0/edge0-infer
RUN cmake --preset cpu \
    && cmake --build --preset cpu --target edge0-infer -j"$(nproc)" \
    && test -x build/cpu/bin/edge0-infer

# Keep only what runs: the two binaries, the engine's Python server and model templates.
RUN mkdir -p /out/edge0/edge0-infer/build/cpu/bin /out/edge0/edge0-strata/build \
    && cp /opt/edge0/LICENSE /opt/edge0/NOTICE /out/edge0/ \
    && { cp -a /opt/edge0/licenses /out/edge0/ 2>/dev/null || true; } \
    && cp /opt/edge0/edge0-infer/build/cpu/bin/edge0-infer /out/edge0/edge0-infer/build/cpu/bin/ \
    && cd /opt/edge0/edge0-strata \
    && tar --exclude=./build --exclude=./third_party --exclude=./.git -cf - . | tar -xf - -C /out/edge0/edge0-strata \
    && cp build/strata /out/edge0/edge0-strata/build/ \
    && git -C /opt/edge0 rev-parse HEAD > /out/edge0/REVISION

############################ runtime stage ############################
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

LABEL org.opencontainers.image.title="edge0-infer (CUDA)" \
      org.opencontainers.image.description="Edge0 inference server (OpenAI- and Anthropic-compatible) with the edge0-strata CUDA engine for Edge0 145B / 120B" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/edge0 /opt/edge0
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && ln -s /opt/edge0/edge0-infer/build/cpu/bin/edge0-infer /usr/local/bin/edge0-infer

ENV EDGE0_STRATA_DIR=/opt/edge0/edge0-strata \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PORT=8080

WORKDIR /opt/edge0
EXPOSE 8080
VOLUME ["/models"]

# The server only answers once the model is loaded; the first load can take a long time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
