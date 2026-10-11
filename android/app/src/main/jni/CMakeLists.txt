project(birddetect)

cmake_minimum_required(VERSION 3.10)

# Both prebuilt packages are downloaded by android/fetch_native_deps.sh.
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.13.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260526-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(birddetect SHARED bird_detect_jni.cpp bird_detector.cpp yolo26_det.cpp ndkcamera.cpp)
target_compile_features(birddetect PRIVATE cxx_std_17)

target_link_libraries(birddetect ncnn ${OpenCV_LIBS} camera2ndk mediandk jnigraphics)
