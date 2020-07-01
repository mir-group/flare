cmake_minimum_required(VERSION 3.12)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(flare LANGUAGES C CXX)

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

# googletest
################################################################################

# Download and unpack googletest at configure time
configure_file(CMakeGoogle.txt googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()


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
add_library(Eigen3 STATIC IMPORTED)
message(${SOURCE_DIR})
include_directories(${SOURCE_DIR})
# target_include_directories(Eigen3 SYSTEM INTERFACE ${SOURCE_DIR})

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
    Sources/Descriptor/y_grad.cpp
    Sources/Descriptor/radial.cpp
    Sources/Descriptor/cutoffs.cpp
    Sources/Descriptor/single_bond.cpp
    Sources/Descriptor/descriptor.cpp
    Sources/Structure/structure.cpp
    Sources/Environment/local_environment.cpp
    Sources/Kernels/kernels.cpp
    Sources/Sparse_GP/sparse_gp.cpp)

set(PYBIND_SOURCES
    Sources/Descriptor/py_descriptor.cpp
    Sources/Structure/py_structure.cpp
    Sources/pyflare.cpp)

# Include sources.
include_directories(Sources)

# Create shared library for unit tests.
add_library(flare SHARED ${FLARE_SOURCES})

# Create pybind module.
add_library(flare_module SHARED ${FLARE_SOURCES} ${PYBIND_SOURCES})
target_link_libraries(flare_module PRIVATE pybind11)
set_target_properties(flare_module PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                              SUFFIX "${PYTHON_MODULE_EXTENSION}")

add_dependencies(flare_module eigen_project pybind11_project)
set_target_properties(flare_module PROPERTIES OUTPUT_NAME "_C_flare")

# Add test directory.
add_subdirectory(CTests)

