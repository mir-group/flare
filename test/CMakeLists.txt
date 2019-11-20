# make executable
add_executable(test test_main.cpp test_y_grad.cpp)
target_include_directories(test PUBLIC ${YGRAD_INCLUDE_DIR})
target_link_libraries(test PUBLIC gtest gtest_main YGrad)
