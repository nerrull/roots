cmake_minimum_required(VERSION 3.17)
project(JardinsRacine LANGUAGES CXX)

set(PIAFMUNCH OFF CACHE BOOL "Disable optional PiafMunch" FORCE)
set(CPLANTBOX_PYTHON_BINDINGS OFF CACHE BOOL "Disable CPlantBox Python bindings" FORCE)
add_subdirectory(CPlantBox)

# Expose CPlantBox headers as PUBLIC so sdf_viewer inherits them via target_link_libraries
set(_CPB ${CMAKE_CURRENT_SOURCE_DIR}/CPlantBox/src)
target_include_directories(CPlantBox PUBLIC
    ${_CPB}/structural
    ${_CPB}/functional
    ${_CPB}/external/eigen
    ${_CPB}/external/tinyxml2
    ${_CPB}/external
    ${_CPB}
)

add_subdirectory(sdf_viewer)
add_subdirectory(freenect_test)
add_subdirectory(bubble_renderer)
