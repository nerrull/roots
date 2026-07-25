cmake_minimum_required(VERSION 3.16)
project(kinect_v2_validate LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)

# --- optional SpeechAnalyzer backend -----------------------------------------
#
# `SpeechAnalyzer`/`SpeechTranscriber` (WWDC25) is the current speech-to-text
# API, but it is Swift-only and needs the **macOS 26 SDK** to compile and
# macOS 26 to run. Older SDKs get the SFSpeechRecognizer backend alone, which is
# a complete implementation rather than a stub -- see src/demo/transcriber.h.
#
# Detected from the SDK, not from the host OS: what can be *compiled* is a
# property of the SDK. The runtime check lives in the Swift `#available` guards.
if(CMAKE_OSX_SYSROOT)
  set(_sdk_path "${CMAKE_OSX_SYSROOT}")
else()
  execute_process(COMMAND xcrun --show-sdk-path --sdk macosx
                  OUTPUT_VARIABLE _sdk_path OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
endif()
execute_process(COMMAND xcrun --show-sdk-version --sdk macosx
                OUTPUT_VARIABLE _sdk_version OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)

set(KV2_SPEECH_ANALYZER OFF)
if(_sdk_version AND _sdk_version VERSION_GREATER_EQUAL 26.0)
  # Swift is enabled only when it will actually be used: turning the language on
  # unconditionally makes every configure depend on a working Swift toolchain.
  include(CheckLanguage)
  check_language(Swift)
  if(CMAKE_Swift_COMPILER)
    enable_language(Swift)
    set(KV2_SPEECH_ANALYZER ON)
    message(STATUS "transcription: SpeechAnalyzer + SFSpeechRecognizer "
                   "(SDK ${_sdk_version})")
  else()
    message(STATUS "transcription: SFSpeechRecognizer only "
                   "(SDK ${_sdk_version} is new enough, but no Swift compiler)")
  endif()
else()
  message(STATUS "transcription: SFSpeechRecognizer only "
                 "(SDK ${_sdk_version} < 26.0, so SpeechAnalyzer is unavailable; "
                 "build against the macOS 26 SDK to enable it)")
endif()

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
      src/demo/person_tracker.cpp
      src/demo/transcriber.cpp
      src/demo/transcriber_sf.mm
      src/demo/voice_clone.mm)
  set_source_files_properties(src/demo/demo_main.mm src/demo/audio_capture.mm
      src/demo/transcriber_sf.mm src/demo/voice_clone.mm
      PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_include_directories(kinect_v2_demo PRIVATE
      src/demo ${freenect2_INCLUDE_DIRS})
  target_link_libraries(kinect_v2_demo PRIVATE
      ${freenect2_LIBRARIES} kinect_demo_imgui glfw
      "-framework Metal" "-framework QuartzCore" "-framework Cocoa"
      "-framework AudioToolbox" "-framework CoreAudio"
      "-framework Speech" "-framework AVFoundation")

  if(KV2_SPEECH_ANALYZER)
    target_sources(kinect_v2_demo PRIVATE
        src/demo/transcriber_analyzer.swift
        src/demo/transcriber_analyzer_shim.cpp)
    target_compile_definitions(kinect_v2_demo PRIVATE KV2_WITH_SPEECH_ANALYZER=1)
    # The Swift half only exports @_cdecl symbols, so no bridging header or
    # generated interface is needed in either direction.
    set_target_properties(kinect_v2_demo PROPERTIES
        Swift_COMPILATION_MODE Wholemodule)
  else()
    target_compile_definitions(kinect_v2_demo PRIVATE KV2_WITH_SPEECH_ANALYZER=0)
  endif()

  # TCC reads the microphone and speech-recognition usage strings out of the
  # binary's embedded plist. A CLI tool has no bundle to hold one, so the
  # linker stashes it in a __TEXT section instead -- without this, asking for
  # speech-recognition authorisation on a bare executable does not work.
  target_link_options(kinect_v2_demo PRIVATE
      "-Wl,-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_SOURCE_DIR}/src/demo/Info.plist")
  set_property(TARGET kinect_v2_demo APPEND PROPERTY LINK_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/src/demo/Info.plist")
endif()

# --- tests -------------------------------------------------------------------
# Headless and hardware-free: they run on synthetic signals, so they work in CI
# and without a sensor plugged in. Run both with `ctest --test-dir build`.
enable_testing()

add_executable(kinect_v2_dsp_test src/demo/test_dsp.cpp src/demo/dsp.cpp)
target_include_directories(kinect_v2_dsp_test PRIVATE src/demo)
add_test(NAME dsp COMMAND kinect_v2_dsp_test)

# Transcript accumulation, WAV round-tripping and reference capture. Links the
# real voice_clone.mm and transcriber backends -- it needs no microphone,
# permission, speech model, or Python server, but it does exercise shipping code
# rather than a copy of it.
add_executable(kinect_v2_voice_test
    src/demo/test_voice.cpp
    src/demo/transcriber.cpp
    src/demo/transcriber_sf.mm
    src/demo/voice_clone.mm)
set_source_files_properties(src/demo/transcriber_sf.mm src/demo/voice_clone.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_include_directories(kinect_v2_voice_test PRIVATE src/demo)
target_link_libraries(kinect_v2_voice_test PRIVATE
    "-framework Foundation" "-framework Speech" "-framework AVFoundation")
if(KV2_SPEECH_ANALYZER)
  target_sources(kinect_v2_voice_test PRIVATE
      src/demo/transcriber_analyzer.swift
      src/demo/transcriber_analyzer_shim.cpp)
  target_compile_definitions(kinect_v2_voice_test PRIVATE
      KV2_WITH_SPEECH_ANALYZER=1)
else()
  target_compile_definitions(kinect_v2_voice_test PRIVATE
      KV2_WITH_SPEECH_ANALYZER=0)
endif()
add_test(NAME voice COMMAND kinect_v2_voice_test)

add_executable(kinect_v2_tracker_test
    src/demo/test_person_tracker.cpp src/demo/person_tracker.cpp)
target_include_directories(kinect_v2_tracker_test PRIVATE
    src/demo ${freenect2_INCLUDE_DIRS})
target_link_libraries(kinect_v2_tracker_test PRIVATE ${freenect2_LIBRARIES})
add_test(NAME person_tracker COMMAND kinect_v2_tracker_test)
