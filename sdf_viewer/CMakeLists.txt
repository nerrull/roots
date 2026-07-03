cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer Homebrew packages on Apple Silicon / Intel Mac
list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)

find_package(OpenGL REQUIRED)
find_package(glfw3  REQUIRED)
find_package(GLEW   REQUIRED)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../imgui)

add_executable(sdf_viewer
    main.cpp
    RootRenderer.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(sdf_viewer PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# Bake shader and parameter paths into the binary so they resolve at runtime
target_compile_definitions(sdf_viewer PRIVATE
    SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    PARAM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../CPlantBox/modelparameter"
)

target_link_libraries(sdf_viewer PRIVATE
    CPlantBox
    glfw
    GLEW::GLEW
    OpenGL::GL
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
)

# --- mask_relay_gui: interactive mask-cavity/attractor root relay viewer ---
add_executable(mask_relay_gui
    render_relay_gui.cpp
    RootRenderer.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(mask_relay_gui PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_compile_definitions(mask_relay_gui PRIVATE
    SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    FACE_OBJ_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/canonical_face_model.obj"
)

target_link_libraries(mask_relay_gui PRIVATE
    CPlantBox
    glfw
    GLEW::GLEW
    OpenGL::GL
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
)
