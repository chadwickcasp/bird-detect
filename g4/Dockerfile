# Linux x86_64 environment for the Google Tensor G4 AOT compiler.
# The compiler plugin ships only as a Linux x86_64 shared library.
# On Apple Silicon this image runs under emulation, which may be slow or refuse the plugin.
#
# LiteRT issue 7598 confirmed ai-edge-litert 2.1.5 with its Tensor SDK for G4. That SDK installs only
# from the access-controlled Tensor ML SDK tarball. Put the tarball in g4/ and build from the repository root:
#   docker build --platform linux/amd64 --build-arg SDK_TARBALL=litert_plugin_compiler.tar.gz -t bird-detect-g4 g4
#   docker run --rm --platform linux/amd64 -v "$PWD":/work bird-detect-g4
#
# Without SDK_TARBALL the image installs the SDK package with no compiler, so a run records where the compile
# stops. The public SDK URLs in 2.2.0 and 2.3.0 returned 404 on October 10, 2026 (docs/g4_smoke.md).
FROM --platform=linux/amd64 python:3.11-slim

ARG LITERT_VERSION=2.1.5
ARG SDK_TARBALL=""
# A bind mount keeps the access-controlled tarball out of every image layer. It needs BuildKit.
RUN --mount=type=bind,source=.,target=/tmp/g4-context \
    if [ -n "$SDK_TARBALL" ]; then export GOOGLE_TENSOR_SDK_BETA="/tmp/g4-context/$SDK_TARBALL"; \
    else export SKIP_SDK_DOWNLOAD=1; fi \
    && pip install --no-cache-dir \
        "ai-edge-litert==${LITERT_VERSION}" \
        "ai-edge-litert-sdk-google-tensor==${LITERT_VERSION}"

# Releases before 2.1.6 register the Google Tensor backend only with this variable set.
ENV GOOGLE_TENSOR_BACKEND_ENABLED=1
WORKDIR /work
ENTRYPOINT ["python", "g4/compile_tensor_g4.py"]
