cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)

find_package(OpenGL REQUIRED)
find_package(glfw3  REQUIRED)
find_package(GLEW   REQUIRED)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../imgui)

# ---------- Compile Metal shader at build time ----------
set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/bubble.metal)
set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/bubble.air)
set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/bubble.metallib)

add_custom_command(
    OUTPUT  ${METAL_AIR}
    COMMAND xcrun -sdk macosx metal -c ${METAL_SRC} -o ${METAL_AIR}
    DEPENDS ${METAL_SRC}
    VERBATIM
)
add_custom_command(
    OUTPUT  ${METAL_LIB}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
    DEPENDS ${METAL_AIR}
    VERBATIM
)
add_custom_target(bubble_metallib DEPENDS ${METAL_LIB})

# ---------- Executable ----------
add_executable(bubble_renderer
    main.cpp
    BubbleRenderer.mm
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_dependencies(bubble_renderer bubble_metallib)

set_source_files_properties(BubbleRenderer.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

target_include_directories(bubble_renderer PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(bubble_renderer PRIVATE
    METALLIB_PATH="${METAL_LIB}"
)

target_link_libraries(bubble_renderer PRIVATE
    glfw
    GLEW::GLEW
    OpenGL::GL
    "-framework Metal"
    "-framework Foundation"
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
    "-framework QuartzCore"
)
