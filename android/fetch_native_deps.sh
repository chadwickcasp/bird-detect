#!/usr/bin/env bash
# Downloads the prebuilt ncnn and opencv-mobile packages that app/src/main/jni/CMakeLists.txt expects.
# Versions match nihui/ncnn-android-yolov8 at commit f1ac75ec (see docs/third_party.md).
set -euo pipefail

JNI_DIR="$(cd "$(dirname "$0")" && pwd)/app/src/main/jni"
NCNN_ZIP="ncnn-20260526-android-vulkan.zip"
NCNN_URL="https://github.com/Tencent/ncnn/releases/download/20260526/${NCNN_ZIP}"
OPENCV_ZIP="opencv-mobile-4.13.0-android.zip"
OPENCV_URL="https://github.com/nihui/opencv-mobile/releases/download/v35/${OPENCV_ZIP}"

fetch_and_unzip() {
  local url="$1" zip="$2" dir="$3"
  if [ -d "${JNI_DIR}/${dir}" ]; then
    echo "${dir} already present"
    return
  fi
  curl -fL --retry 3 -o "${JNI_DIR}/${zip}" "${url}"
  unzip -q "${JNI_DIR}/${zip}" -d "${JNI_DIR}"
  rm "${JNI_DIR}/${zip}"
}

fetch_and_unzip "${NCNN_URL}" "${NCNN_ZIP}" "ncnn-20260526-android-vulkan"
fetch_and_unzip "${OPENCV_URL}" "${OPENCV_ZIP}" "opencv-mobile-4.13.0-android"
