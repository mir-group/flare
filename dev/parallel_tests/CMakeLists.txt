# make library
add_executable(par_test par_test.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(par_test PRIVATE OpenMP::OpenMP_CXX)

