add_executable(test Descriptor/test_y_grad.cpp Descriptor/test_radial.cpp)
include_directories(../Sources/)
target_link_libraries(test PUBLIC gtest gtest_main flare)
