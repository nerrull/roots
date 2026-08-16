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
    src/transition_scene.mm
    src/fit_view_scene.mm
    src/midi_in.mm
    src/ui_params.cpp
    src/text_sdf.cpp
    src/text_overlay.mm
    src/fullscreen_present.mm)
target_include_directories(mirror_app PRIVATE src)
target_link_libraries(mirror_app PRIVATE mirror_imgui glfw
    "-framework Metal" "-framework MetalKit" "-framework QuartzCore"
    "-framework CoreMIDI"
    # CoreText/CoreGraphics rasterise the text overlay's glyphs. Cocoa pulls
    # both in transitively, but the overlay uses them directly.
    "-framework CoreText" "-framework CoreGraphics"
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
    src/mlp_backward.cpp
    src/mirror_train.cpp
    src/fit_target.cpp
    src/face_mask.cpp
    src/face_tracker.cpp
    src/face_basis.cpp
    src/face_fit.cpp
    src/mirror_render.cpp
    src/screen_layout.cpp
    src/show_timeline.cpp
    src/pond_state.cpp)
target_include_directories(mirror_core PUBLIC src ${MLX_INCLUDE_DIRS})
# external/ is gitignored: face_basis.bin and the MediaPipe dylib are both
# generated, not committed. PUBLIC so the tests find the basis too.
target_compile_definitions(mirror_core PUBLIC
    MIRROR_APP_EXTERNAL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/external"
    # show_timeline resolves shows/ against this, the same way RootScene
    # resolves presets/ -- scripts are edited in the source tree during an
    # install, not copied into a build directory.
    MIRROR_APP_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(mirror_core PUBLIC mlx)

# --- MediaPipe face tracking -------------------------------------------------
# Uses MediaPipe's OFFICIAL C Tasks API (mediapipe/tasks/c/vision/face_landmarker)
# built as a shared library, not cpvrlab/libmediapipe: that wrapper pins
# MediaPipe v0.8.11, which predates the Tasks API entirely (legacy 468-point
# face_mesh, no blendshapes, no pose matrix) and is GPL-3.0. Upstream's C API
# gives 478 landmarks + 52 blendshapes + a 4x4 transform under Apache-2.0.
#
# Build it once with (from external/mediapipe, needs a JDK on PATH):
#   bazel build --config darwin_arm64 -c opt --strip always \
#     --define MEDIAPIPE_DISABLE_GPU=1 --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
#     //mediapipe/tasks/c/vision/face_landmarker:libface_landmarker.dylib
set(_mp ${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe)
set(_mp_lib ${_mp}/bazel-bin/mediapipe/tasks/c/vision/face_landmarker/libface_landmarker.dylib)
if(EXISTS ${_mp_lib})
    # Bazel builds the dylib with install_name "libface_landmarker.dylib" -- a
    # bare filename, which dyld treats as a literal path and never resolves
    # against an rpath. Copy it out of bazel-bin (which is a symlink into a
    # cache anyway) and rewrite the id to @rpath so the usual mechanism works.
    set(_mp_stage ${CMAKE_CURRENT_BINARY_DIR}/mediapipe_lib)
    file(MAKE_DIRECTORY ${_mp_stage})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            ${_mp_lib} ${_mp_stage}/libface_landmarker.dylib)
    execute_process(COMMAND install_name_tool -id @rpath/libface_landmarker.dylib
                            ${_mp_stage}/libface_landmarker.dylib
                    ERROR_QUIET)
    target_include_directories(mirror_core PUBLIC ${_mp})
    target_link_libraries(mirror_core PUBLIC ${_mp_stage}/libface_landmarker.dylib)
    # PUBLIC on a static lib propagates to whatever links mirror_core.
    target_link_options(mirror_core PUBLIC "-Wl,-rpath,${_mp_stage}")
    target_compile_definitions(mirror_core PUBLIC MIRROR_HAVE_MEDIAPIPE=1)
    message(STATUS "mirror_app: MediaPipe face tracking enabled")
else()
    target_compile_definitions(mirror_core PUBLIC MIRROR_HAVE_MEDIAPIPE=0)
    message(STATUS "mirror_app: MediaPipe face tracking disabled "
                   "(no libface_landmarker.dylib -- see the bazel command above)")
endif()

# --- Kinect v2 as a live fitting target --------------------------------------
# Reuses kinect_v2_validate's KinectSource rather than a second copy: it already
# carries the USB-reset policy and the independent latest-wins listeners that
# module exists to get right. Optional -- without freenect2 the app builds and
# runs, just without the camera target.
set(_kv2 ${CMAKE_CURRENT_SOURCE_DIR}/../kinect_v2_validate)
list(APPEND CMAKE_PREFIX_PATH ${_kv2}/external/libfreenect2/install)
find_package(freenect2 QUIET)
if(freenect2_FOUND AND EXISTS ${_kv2}/src/demo/kinect_source.cpp)
    target_sources(mirror_core PRIVATE
        src/kinect_target.cpp
        ${_kv2}/src/demo/kinect_source.cpp)
    target_include_directories(mirror_core PUBLIC
        ${_kv2}/src/demo ${freenect2_INCLUDE_DIRS})
    target_link_libraries(mirror_core PUBLIC ${freenect2_LIBRARIES})
    target_compile_definitions(mirror_core PUBLIC MIRROR_HAVE_KINECT=1)
    message(STATUS "mirror_app: Kinect fit target enabled")
else()
    target_compile_definitions(mirror_core PUBLIC MIRROR_HAVE_KINECT=0)
    message(STATUS "mirror_app: Kinect fit target disabled (no freenect2)")
endif()

target_link_libraries(mirror_app PRIVATE mirror_core)

# --- mirror_sim (live CPlantBox root growth) ---------------------------------
# GL-free "mask relay" growth, reusing sdf_viewer's MaskCavities/RootAttractors
# headers and CPlantBox directly. Pure C++ (CPlantBox types hidden behind pimpl),
# so it links into the ObjC++ app cleanly.
add_library(mirror_sim STATIC src/root_sim.cpp)
target_include_directories(mirror_sim PRIVATE
    src ${CMAKE_CURRENT_SOURCE_DIR}/../sdf_viewer)
target_link_libraries(mirror_sim PUBLIC CPlantBox)

target_link_libraries(mirror_app PRIVATE mirror_sim)
target_compile_definitions(mirror_app PRIVATE
    ROOTSIM_PARAM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../CPlantBox/modelparameter/structural/rootsystem/")

# --- tests -------------------------------------------------------------------
add_executable(mlp_parity_test tests/mlp_parity_test.cpp)
target_link_libraries(mlp_parity_test PRIVATE mirror_core)

add_executable(pond_parity_test tests/pond_parity_test.cpp)
target_link_libraries(pond_parity_test PRIVATE mirror_core)

# Fused ripple kernel vs. the op-graph reference it replaced. Needs a Metal
# device but no fixtures, so unlike the two above it runs without a dump step.
add_executable(ripple_parity_test tests/ripple_parity_test.cpp)
target_link_libraries(ripple_parity_test PRIVATE mirror_core)

# --- subsystems (wired in as they land) --------------------------------------
# TODO(tracking): prebuilt libmediapipe .dylib for face_tracker.
# TODO(roots): link sdfsim (GL-free CPlantBox sim) + MetalRootRenderer sources.

# The fused MLP at hidden widths other than the fixture's 64. hidden_dim is a
# kernel template parameter, so each width is a different kernel with a
# different simdgroup decomposition; the fixture test cannot see any of that.
add_executable(mlp_width_test tests/mlp_width_test.cpp)
target_link_libraries(mlp_width_test PRIVATE mirror_core)

# The fused backward against an fp32 MLX-op backprop, across widths, depths,
# hybrid splits, activations and ragged row counts. No fixtures needed.
add_executable(mlp_backward_test tests/mlp_backward_test.cpp)
target_link_libraries(mlp_backward_test PRIVATE mirror_core)

# End-to-end fit: loss falls AND the rendered image moves toward the target,
# including at a resolution the fit never saw.
add_executable(fit_test tests/fit_test.cpp)
target_link_libraries(fit_test PRIVATE mirror_core)

# Resampling + fitting a moving target (the camera case), with a synthetic feed
# so it needs no sensor.
add_executable(fit_target_test tests/fit_target_test.cpp)
target_link_libraries(fit_target_test PRIVATE mirror_core)

# The PBD cloth solver, headless. No Metal, no GPU -- which is why the solver was
# written GL-free in the first place.
add_executable(cloth_test tests/cloth_test.cpp)
target_include_directories(cloth_test PRIVATE src)

# Landmark outline -> training mask. Pure geometry; no MediaPipe needed.
add_executable(face_mask_test tests/face_mask_test.cpp)
target_link_libraries(face_mask_test PRIVATE mirror_core)

# Morphable-model fit against synthetic landmarks with a known ground-truth
# identity. Needs external/face_basis.bin; skips cleanly without it.
add_executable(face_fit_test tests/face_fit_test.cpp)
target_link_libraries(face_fit_test PRIVATE mirror_core)

# Composition size / letterbox arithmetic, and the camera crop that follows it.
# Pure arithmetic, no Metal: exactly the part that is wrong by a few pixels in a
# way a screenshot does not show.
add_executable(screen_layout_test tests/screen_layout_test.cpp)
target_link_libraries(screen_layout_test PRIVATE mirror_core)

# The show's running order: floors, ceilings, debounced events, text cues. Pure
# state machine, driven at a fixed step -- the alternative is discovering a bad
# debounce with somebody standing in front of the piece.
add_executable(show_timeline_test tests/show_timeline_test.cpp)
target_link_libraries(show_timeline_test PRIVATE mirror_core)

# The text overlay's distance transform, against an analytically known shape.
# Pure geometry: no Metal, no fonts, no MLX.
add_executable(text_sdf_test tests/text_sdf_test.cpp src/text_sdf.cpp)
target_include_directories(text_sdf_test PRIVATE src)
