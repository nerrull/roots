# scope_monitor -- standalone imgui app that reads SignalScope's shared-memory
# buffers (see ../mi_common/signal_scope_shm.h) and displays waveform,
# spectrogram and spectral-envelope views for whichever capture instance is
# selected.
#
# Not part of the repo's top-level CMake superbuild (that one configures
# CPlantBox/mirror_app for the Mac/Metal toolchain via OBJCXX and isn't
# relevant here). Build standalone:
#
#   cmake -S wwise_plugins/scope_monitor -B wwise_plugins/scope_monitor/build
#   cmake --build wwise_plugins/scope_monitor/build --config Release
#   wwise_plugins/scope_monitor/build/Release/scope_monitor.exe
#
# GLFW is fetched via CMake FetchContent (source build, no system package
# manager) to keep this buildable on a bare Windows/Visual Studio install.
cmake_minimum_required(VERSION 3.17)
project(scope_monitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../imgui)
if(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
    message(FATAL_ERROR "imgui submodule not checked out -- run: git submodule update --init imgui")
endif()

add_library(scope_monitor_imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(scope_monitor_imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(scope_monitor_imgui PUBLIC glfw)

add_executable(scope_monitor
    src/main.cpp)
target_include_directories(scope_monitor PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../mi_common)
target_link_libraries(scope_monitor PRIVATE scope_monitor_imgui glfw)

if(WIN32)
    target_link_libraries(scope_monitor PRIVATE opengl32)
elseif(APPLE)
    target_link_libraries(scope_monitor PRIVATE "-framework OpenGL")
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(scope_monitor PRIVATE OpenGL::GL)
endif()
