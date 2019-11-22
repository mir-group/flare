# make executable
add_executable(test test_main.cpp test_y_grad.cpp test_radial.cpp)
target_include_directories(test PUBLIC ${ACE_INCLUDE_DIR})
target_link_libraries(test PUBLIC gtest gtest_main Ace)
