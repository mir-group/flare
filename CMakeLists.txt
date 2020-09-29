cmake_minimum_required(VERSION 3.14.5)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(flare LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FLARE_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling modules")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "[flare] CMAKE_BUILD_TYPE not specified, setting it to "
                 "Release. Use `-DCMAKE_BUILD_TYPE=...` to overwrite.")
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#

include(FetchContent)

# googletest
################################################################################

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/googletest
    UPDATE_COMMAND ""
)
FetchContent_MakeAvailable(googletest)

# Kokkos
################################################################################

#FetchContent_Declare(kokkos
#    GIT_REPOSITORY https://github.com/kokkos/kokkos
#    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/kokkos
#    UPDATE_COMMAND ""
#)
#FetchContent_GetProperties(kokkos)
#FetchContent_Populate(kokkos)
#add_subdirectory(${kokkos_SOURCE_DIR} ${kokkos_BINARY_DIR})
#include_directories(${Kokkos_SOURCE_DIR})


# Eigen3
################################################################################

ExternalProject_Add(
    eigen_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/Eigen3"
    URL "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.tar.gz"
    URL_HASH MD5=77a2c934eaf35943c43ee600a83b72df
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(eigen_project SOURCE_DIR)
add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 SYSTEM INTERFACE ${SOURCE_DIR})
include_directories(${SOURCE_DIR})

# pybind11
###############################################################################
ExternalProject_Add(
    pybind11_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/pybind11"
    URL "https://github.com/pybind/pybind11/archive/v2.3.0.tar.gz"
    URL_HASH MD5=e2120c5b0e3c20a93f2dfcdc55026ba8
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
if (NOT FLARE_PYTHON_VERSION)
    set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5)
endif()
find_package(PythonLibsNew ${FLARE_PYTHON_VERSION} REQUIRED)

add_library(pybind11 INTERFACE)
ExternalProject_Get_Property(pybind11_project SOURCE_DIR)
target_include_directories(pybind11 SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_include_directories(pybind11 SYSTEM INTERFACE ${PYTHON_INCLUDE_DIRS})

if(APPLE)
      TARGET_LINK_LIBRARIES(pybind11 INTERFACE "-undefined dynamic_lookup")
      message(STATUS "Building in conda environment on MAC")
else()
      target_link_libraries(pybind11 INTERFACE ${PYTHON_LIBRARIES})
endif()

# Greatly reduces the code bloat
target_compile_options(pybind11 INTERFACE "-fvisibility=hidden")
###############################################################################

# Specify source files.
set(FLARE_SOURCES
    src/flare_pp/y_grad.cpp
    src/flare_pp/radial.cpp
    src/flare_pp/cutoffs.cpp
    src/flare_pp/single_bond.cpp
    src/flare_pp/descriptor.cpp
    src/flare_pp/structure.cpp
    src/flare_pp/local_environment.cpp
    src/flare_pp/kernels.cpp
    src/flare_pp/two_body_kernel.cpp
    src/flare_pp/three_body_kernel.cpp
    src/flare_pp/dot_product_kernel.cpp
    src/flare_pp/sparse_gp.cpp
    src/flare_pp/sparse_gp_dtc.cpp
    src/flare_pp/compact_structure.cpp
    src/flare_pp/compact_gp.cpp)

set(PYBIND_SOURCES
    src/ace_binding.cpp)

# Include sources.
include_directories(src/flare_pp)

# Create flare library.
add_library(flare SHARED ${FLARE_SOURCES})
add_dependencies(flare eigen_project)

# Link to OpenMP.
if (DEFINED ENV{OMP_LOC})
    target_link_libraries(flare PUBLIC $ENV{OMP_LOC})
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(flare PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# Link to MKL.
target_compile_definitions(flare PUBLIC EIGEN_USE_MKL_ALL=1)
include_directories($ENV{MKL_INCLUDE})
target_link_libraries(flare PUBLIC $ENV{MKL_LIBS})

# Create pybind module.
add_library(flare_module SHARED ${PYBIND_SOURCES})
target_link_libraries(flare_module PUBLIC flare pybind11)
set_target_properties(flare_module PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                              SUFFIX "${PYTHON_MODULE_EXTENSION}")
add_dependencies(flare_module pybind11_project)
set_target_properties(flare_module PROPERTIES OUTPUT_NAME "_C_flare")

# Add test directory.
add_subdirectory(tests)
