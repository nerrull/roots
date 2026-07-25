cmake_minimum_required(VERSION 3.16)
project(kinect_v2_validate LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)

# We build libfreenect2 from source into external/libfreenect2/install
# (see setup.sh). Point CMake at that local install first.
set(_local_prefix "${CMAKE_CURRENT_SOURCE_DIR}/external/libfreenect2/install")
list(APPEND CMAKE_PREFIX_PATH "${_local_prefix}" /opt/homebrew /usr/local)

find_package(freenect2 REQUIRED)

# --- headless validator ------------------------------------------------------
add_executable(kinect_v2_validate src/main.cpp)
target_include_directories(kinect_v2_validate PRIVATE ${freenect2_INCLUDE_DIRS})
target_link_libraries(kinect_v2_validate PRIVATE ${freenect2_LIBRARIES})

# --- ImGui demo viewer -------------------------------------------------------
# Optional: only configured if GLFW and the vendored ImGui are both present.
find_package(glfw3 QUIET)
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../imgui)

if(NOT glfw3_FOUND)
  message(STATUS "kinect_v2_demo: skipped (glfw3 not found — brew install glfw)")
elseif(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
  message(STATUS "kinect_v2_demo: skipped (no ImGui at ${IMGUI_DIR})")
else()
  # ImGui core + GLFW (C++) and Metal (ObjC++, needs ARC) backends. Metal keeps
  # the UI off OpenGL entirely, so we never share a GL context with
  # libfreenect2's OpenGL depth pipeline.
  add_library(kinect_demo_imgui STATIC
      ${IMGUI_DIR}/imgui.cpp
      ${IMGUI_DIR}/imgui_draw.cpp
      ${IMGUI_DIR}/imgui_tables.cpp
      ${IMGUI_DIR}/imgui_widgets.cpp
      ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
      ${IMGUI_DIR}/backends/imgui_impl_metal.mm)
  target_include_directories(kinect_demo_imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
  set_source_files_properties(${IMGUI_DIR}/backends/imgui_impl_metal.mm
      PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_link_libraries(kinect_demo_imgui PUBLIC glfw
      "-framework Metal" "-framework MetalKit" "-framework QuartzCore"
      "-framework Cocoa" "-framework Foundation" "-framework GameController")

  add_executable(kinect_v2_demo
      src/demo/demo_main.mm
      src/demo/kinect_source.cpp
      src/demo/audio_capture.mm
      src/demo/dsp.cpp
      src/demo/person_tracker.cpp)
  set_source_files_properties(src/demo/demo_main.mm src/demo/audio_capture.mm
      PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_include_directories(kinect_v2_demo PRIVATE
      src/demo ${freenect2_INCLUDE_DIRS})
  target_link_libraries(kinect_v2_demo PRIVATE
      ${freenect2_LIBRARIES} kinect_demo_imgui glfw
      "-framework Metal" "-framework QuartzCore" "-framework Cocoa"
      "-framework AudioToolbox" "-framework CoreAudio")
endif()

# --- tests -------------------------------------------------------------------
# Headless and hardware-free: they run on synthetic signals, so they work in CI
# and without a sensor plugged in. Run both with `ctest --test-dir build`.
enable_testing()

add_executable(kinect_v2_dsp_test src/demo/test_dsp.cpp src/demo/dsp.cpp)
target_include_directories(kinect_v2_dsp_test PRIVATE src/demo)
add_test(NAME dsp COMMAND kinect_v2_dsp_test)

add_executable(kinect_v2_tracker_test
    src/demo/test_person_tracker.cpp src/demo/person_tracker.cpp)
target_include_directories(kinect_v2_tracker_test PRIVATE
    src/demo ${freenect2_INCLUDE_DIRS})
target_link_libraries(kinect_v2_tracker_test PRIVATE ${freenect2_LIBRARIES})
add_test(NAME person_tracker COMMAND kinect_v2_tracker_test)
