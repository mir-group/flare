# set minimum required version of cmake
cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# set the project name
project(ACE_Tests VERSION 1.0
	    DESCRIPTION "High performance implementation of the atomic cluster expansion."
	    LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES ("${EIGEN3_INCLUDE_DIR}")

find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(ACE_ROOT_DIR $ENV{ACE})

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(timing)
add_subdirectory(pybind11)
add_subdirectory(googletest)
add_subdirectory(parallel_tests)

# make python bindings
pybind11_add_module(ace src/ace_binding.cpp src/y_grad.cpp src/structure.cpp
		    src/local_environment.cpp src/descriptor.cpp
		    src/radial.cpp src/cutoffs.cpp src/single_bond.cpp
		    src/kernels.cpp src/sparse_gp.cpp)
find_package(OpenMP REQUIRED)
target_link_libraries(ace PRIVATE OpenMP::OpenMP_CXX)
