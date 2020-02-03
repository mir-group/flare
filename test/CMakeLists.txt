# make executable
add_executable(test test_main.cpp test_y_grad.cpp test_radial.cpp
               test_structure.cpp test_local_environment.cpp test_bond_env.cpp
	       test_descriptor.cpp test_kernels.cpp)
target_include_directories(test PUBLIC ${ACE_INCLUDE_DIR})
target_link_libraries(test PUBLIC gtest gtest_main Ace)
