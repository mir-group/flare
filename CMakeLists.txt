# set minimum required version of cmake
cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# set the project name
project(flare_pp VERSION 1.0
	    DESCRIPTION "High performance implementation of the atomic cluster expansion."
	    LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES ("${EIGEN3_INCLUDE_DIR}")

set(ACE_ROOT_DIR $ENV{ACE})

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(timing)
add_subdirectory(googletest)
add_subdirectory(parallel_tests)

# make python bindings
add_subdirectory(pybind11)
pybind11_add_module(ace SHARED src/ace_binding.cpp)
target_link_libraries(ace PRIVATE flare_pp)
target_include_directories(ace PRIVATE ${ACE_INCLUDE_DIR})
