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
    GIT_TAG release-1.11.0
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/googletest
    UPDATE_COMMAND ""
)
FetchContent_MakeAvailable(googletest)

# Json
################################################################################

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/json
    GIT_TAG v3.9.1)

FetchContent_GetProperties(json)
FetchContent_Populate(json)
add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR})
include_directories(${json_SOURCE_DIR})

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
set(PYBIND11_FINDPYTHON ON)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
)
FetchContent_MakeAvailable(pybind11)
###############################################################################

# Specify source files.
set(FLARE_SOURCES
    src/flare_pp/y_grad.cpp
    src/flare_pp/radial.cpp
    src/flare_pp/cutoffs.cpp
    src/flare_pp/structure.cpp
    src/flare_pp/bffs/sparse_gp.cpp
    src/flare_pp/bffs/gp.cpp
    src/flare_pp/descriptors/descriptor.cpp
    src/flare_pp/descriptors/b2.cpp
    src/flare_pp/descriptors/b2_norm.cpp
    src/flare_pp/descriptors/b2_simple.cpp
    src/flare_pp/descriptors/b3.cpp
    src/flare_pp/descriptors/wigner3j.cpp
    src/flare_pp/descriptors/two_body.cpp
    src/flare_pp/descriptors/three_body.cpp
    src/flare_pp/descriptors/three_body_wide.cpp
    src/flare_pp/descriptors/four_body.cpp
    src/flare_pp/kernels/kernel.cpp
    src/flare_pp/kernels/normalized_dot_product.cpp
    src/flare_pp/kernels/dot_product.cpp
    src/flare_pp/kernels/norm_dot_icm.cpp
    src/flare_pp/kernels/squared_exponential.cpp)

set(PYBIND_SOURCES
    src/ace_binding.cpp)

# Include sources.
include_directories(
  src/flare_pp
  src/flare_pp/descriptors
  src/flare_pp/kernels
  src/flare_pp/bffs)

# Create flare library.
# Should be static to avoid issues with pip installation.
# See: https://github.com/pybind/cmake_example/issues/11
add_library(flare STATIC ${FLARE_SOURCES})
set_target_properties(flare PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(flare eigen_project)

# Link to json.
target_link_libraries(flare PUBLIC nlohmann_json::nlohmann_json)

# Add conda include directories
if (DEFINED ENV{CONDA_PREFIX})
  message(STATUS "Adding conda include directories.")
  include_directories($ENV{CONDA_PREFIX}/include)
endif()

# Link to OpenMP.
if (DEFINED ENV{OMP_LOC})
    target_link_libraries(flare PUBLIC $ENV{OMP_LOC})
    message(STATUS "Linking to user-specified OpenMP library.")
elseif(DEFINED ENV{NO_OMP})
    message(STATUS "Building without OpenMP.")
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(flare PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# Check for user-specified MKL package.
if (DEFINED ENV{MKL_INCLUDE} AND DEFINED ENV{MKL_LIBS})
  message(STATUS "Linking Eigen to user-specified MKL libraries.")
  include_directories($ENV{MKL_INCLUDE})
  target_link_libraries(flare PUBLIC $ENV{MKL_LIBS})
  target_compile_definitions(flare PUBLIC EIGEN_USE_MKL_ALL=1)
# Give the option to do a "bare-bones" build without Lapack/Lapacke.
elseif(DEFINED ENV{NO_LAPACK})
  message(STATUS "Building without Lapack.")
else()
  find_package(LAPACK REQUIRED)
  target_link_libraries(flare PUBLIC ${LAPACK_LIBRARIES})

  if (LAPACK_LIBRARIES MATCHES mkl)
    message(STATUS "Linking Eigen to MKL.")
    target_compile_definitions(flare PUBLIC EIGEN_USE_MKL_ALL=1)
  else()
    find_package(LAPACKE)
    if(LAPACKE_FOUND)
      target_link_libraries(flare PUBLIC ${LAPACKE_LIBRARIES})
    endif()
    message(STATUS "Linking Eigen to LAPACKE.")
    target_compile_definitions(flare PUBLIC EIGEN_USE_LAPACKE=1)
  endif()
endif()

# Create pybind module.
pybind11_add_module(flare_module ${PYBIND_SOURCES})
target_link_libraries(flare_module PUBLIC flare)
set_target_properties(flare_module PROPERTIES OUTPUT_NAME "_C_flare")

# Add test directory.
add_subdirectory(ctests)
