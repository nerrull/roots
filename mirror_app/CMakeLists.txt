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
    src/metal_context.mm
    src/mirror_scene.mm
    src/metal_root_renderer.mm
    src/root_scene.mm
    src/fullscreen_present.mm)
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
    MIRROR_APP_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    MIRROR_APP_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src"
    MIRROR_APP_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")

# --- MLX (neural mirror) -----------------------------------------------------
# The MLX C++ library + CMake package ships inside neuromirror's venv (pip mlx).
if(NOT MLX_DIR)
    file(GLOB _mlx_cfg_dirs
        "${NEUROMIRROR_DIR}/.venv/lib/python*/site-packages/mlx/share/cmake/MLX")
    if(_mlx_cfg_dirs)
        list(GET _mlx_cfg_dirs 0 MLX_DIR)
    endif()
endif()
find_package(MLX CONFIG REQUIRED)

# mirror_core: the neural-mirror compute (MLX C++ port of the fused MLP), shared
# by the app and the parity test. Pure C++ (no ObjC).
add_library(mirror_core STATIC
    src/mlp_forward.cpp
    src/mirror_render.cpp
    src/pond_state.cpp)
target_include_directories(mirror_core PUBLIC src ${MLX_INCLUDE_DIRS})
target_link_libraries(mirror_core PUBLIC mlx)

target_link_libraries(mirror_app PRIVATE mirror_core)

# --- tests -------------------------------------------------------------------
add_executable(mlp_parity_test tests/mlp_parity_test.cpp)
target_link_libraries(mlp_parity_test PRIVATE mirror_core)

add_executable(pond_parity_test tests/pond_parity_test.cpp)
target_link_libraries(pond_parity_test PRIVATE mirror_core)

# --- subsystems (wired in as they land) --------------------------------------
# TODO(tracking): prebuilt libmediapipe .dylib for face_tracker.
# TODO(roots): link sdfsim (GL-free CPlantBox sim) + MetalRootRenderer sources.
