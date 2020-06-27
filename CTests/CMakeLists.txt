add_executable(test test_y_grad.cpp test_simple.cpp)
include_directories(../Sources/)
target_link_libraries(test PUBLIC gtest gtest_main flare)
