cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)

find_package(OpenGL    REQUIRED)
find_package(glfw3     REQUIRED)
find_package(GLEW      REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(OpenCV    REQUIRED COMPONENTS core imgproc)

find_path(FREENECT_INCLUDE_DIR
    NAMES libfreenect/libfreenect.h
    PATHS /opt/homebrew/include /usr/local/include
)

find_library(FREENECT_LIBRARY
    NAMES freenect
    PATHS /opt/homebrew/lib /usr/local/lib
)

if(NOT FREENECT_INCLUDE_DIR OR NOT FREENECT_LIBRARY)
    message(FATAL_ERROR "libfreenect not found. Install via: brew install libfreenect")
endif()

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../imgui)

add_executable(freenect_test
    main.cpp
    pose/rtmdet_onnxruntime.cpp
    pose/rtmpose_onnxruntime.cpp
    pose/rtmpose_tracker_onnxruntime.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(freenect_test PRIVATE
    ${FREENECT_INCLUDE_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/pose
)

target_compile_definitions(freenect_test PRIVATE
    MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models"
)

target_link_libraries(freenect_test PRIVATE
    ${FREENECT_LIBRARY}
    glfw
    GLEW::GLEW
    OpenGL::GL
    onnxruntime::onnxruntime
    opencv_core
    opencv_imgproc
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
    "-framework CoreML"
    "-framework Foundation"
)
