#!/usr/bin/env bash
# Copies libLiteRtDispatch_GoogleTensor.so from the LiteRT v2.1.6 NPU runtime zip into jniLibs.
# v2.1.6 matches the smoke app runtime and the on-device compiler plugin. The .so is gitignored.
set -euo pipefail

JNI_LIBS="$(cd "$(dirname "$0")" && pwd)/src/main/jniLibs/arm64-v8a"
ZIP_URL="https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.6/litert_npu_runtime_libraries.zip"
LIBRARY="google_tensor_runtime/src/main/jni/arm64-v8a/libLiteRtDispatch_GoogleTensor.so"

mkdir -p "${JNI_LIBS}"
TMP_ZIP="$(mktemp "${TMPDIR:-/tmp}/litert_npu_runtime.XXXXXX")"
trap 'rm -f "$TMP_ZIP"' EXIT
curl -fL --retry 3 -o "${TMP_ZIP}" "${ZIP_URL}"
unzip -o -j "${TMP_ZIP}" "${LIBRARY}" -d "${JNI_LIBS}"
