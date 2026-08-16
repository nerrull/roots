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
  # Deliberately *not* enable_language(Swift): CMake implements Swift only for
  # the Ninja and Xcode generators, so `enable_language` hard-errors under the
  # Makefiles generator this project otherwise builds fine with. Since the Swift
  # half is a single file exporting a handful of @_cdecl symbols, invoking
  # swiftc directly is both simpler and generator-independent -- see
  # kv2_add_speech_analyzer() below.
  find_program(KV2_SWIFTC swiftc)
  if(KV2_SWIFTC)
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

if(KV2_SPEECH_ANALYZER)
  set(_sa_swift "${CMAKE_CURRENT_SOURCE_DIR}/src/demo/transcriber_analyzer.swift")
  set(_sa_obj "${CMAKE_CURRENT_BINARY_DIR}/transcriber_analyzer.o")

  # Pinned to the Swift 5 language mode. Under Swift 6's strict concurrency the
  # AVAudioConverter input block trips Sendable diagnostics -- warnings today,
  # errors if the toolchain default ever moves -- and the block is documented to
  # run synchronously on the calling thread, so there is nothing to fix.
  # -parse-as-library: without it swiftc treats a lone file as a script and
  # synthesises a `main`, which then collides with the real one at link time.
  set(_sa_flags -parse-as-library -swift-version 5 -O -wmo
                -module-name kv2_speech -sdk "${_sdk_path}")
  if(CMAKE_OSX_DEPLOYMENT_TARGET)
    # Match the C++ half, so `#available(macOS 26.0, *)` stays a real runtime
    # check rather than a statically-true one.
    set(_sa_target "${CMAKE_SYSTEM_PROCESSOR}-apple-macos${CMAKE_OSX_DEPLOYMENT_TARGET}")
    list(APPEND _sa_flags -target "${_sa_target}")
  endif()

  add_custom_command(OUTPUT "${_sa_obj}"
      COMMAND "${KV2_SWIFTC}" -c "${_sa_swift}" -o "${_sa_obj}" ${_sa_flags}
      DEPENDS "${_sa_swift}"
      COMMENT "Compiling Swift SpeechAnalyzer backend"
      VERBATIM)
  add_custom_target(kv2_speech_analyzer_swift DEPENDS "${_sa_obj}")
  # swiftc stamps LC_LINKER_OPTION load commands for swiftCore, _Concurrency and
  # friends into the object, so the ordinary clang link picks the Swift runtime
  # up on its own -- no swiftc-driven link step, no -L needed.
  set_source_files_properties("${_sa_obj}" PROPERTIES
      EXTERNAL_OBJECT TRUE GENERATED TRUE)

  # Adds the backend to `target`: the prebuilt Swift object, the C++ shim that
  # adapts its C ABI to the Transcriber interface, and the feature macro that
  # transcriber_sf.mm keys the kAuto preference off.
  function(kv2_add_speech_analyzer target)
    target_sources(${target} PRIVATE
        "${_sa_obj}" src/demo/transcriber_analyzer_shim.cpp)
    add_dependencies(${target} kv2_speech_analyzer_swift)
    target_compile_definitions(${target} PRIVATE KV2_WITH_SPEECH_ANALYZER=1)
  endfunction()
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
    kv2_add_speech_analyzer(kinect_v2_demo)
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
  kv2_add_speech_analyzer(kinect_v2_voice_test)
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
