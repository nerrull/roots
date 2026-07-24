# mirror_app — unified Metal app (neural mirror + 3D root scene).
#
# Added to the JardinsRacine superbuild via add_subdirectory in the parent
# CMakeLists. Built from the repo root, e.g.:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --target mirror_app -j
#   ./build/mirror_app/mirror_app
#
# Scope right now: the Metal window shell only. MLX (neural mirror), libmediapipe
# (face tracking) and the Metal root renderer link in as their subsystems land —
# see the marked sections below.

project(mirror_app LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)
find_package(glfw3 REQUIRED)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../imgui)

# ImGui core + GLFW (C++) + Metal (ObjC++) backends. The Metal backend uses ARC.
add_library(mirror_imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_metal.mm)
target_include_directories(mirror_imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
set_source_files_properties(${IMGUI_DIR}/backends/imgui_impl_metal.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(mirror_imgui PUBLIC glfw
    "-framework Metal" "-framework MetalKit" "-framework QuartzCore"
    "-framework Cocoa" "-framework Foundation" "-framework GameController")

# --- app ---------------------------------------------------------------------
add_executable(mirror_app
    src/main.mm
    src/metal_context.mm)
target_include_directories(mirror_app PRIVATE src)
target_link_libraries(mirror_app PRIVATE mirror_imgui glfw
    "-framework Metal" "-framework MetalKit" "-framework QuartzCore"
    "-framework Cocoa" "-framework Foundation")

# Neuromirror sibling repo (MSL sources, face_landmarker.task, canonical model).
set(NEUROMIRROR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../neuromirror"
    CACHE PATH "Path to the neuromirror repo checkout")
target_compile_definitions(mirror_app PRIVATE
    NEUROMIRROR_DIR="${NEUROMIRROR_DIR}"
    SDF_VIEWER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../sdf_viewer"
    MIRROR_APP_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")

# --- subsystems (wired in as they land) --------------------------------------
# TODO(mirror): MLX C++ (find_package(mlx)) for mlp_forward — neural mirror.
# TODO(tracking): prebuilt libmediapipe .dylib for face_tracker.
# TODO(roots): link sdfsim (GL-free CPlantBox sim) + MetalRootRenderer sources.
