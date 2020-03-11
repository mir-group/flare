# make executable
add_executable(tests test_main.cpp test_y_grad.cpp test_radial.cpp
               test_structure.cpp test_local_environment.cpp test_bond_env.cpp
	       test_descriptor.cpp test_kernels.cpp test_sparse_gp.cpp)
target_include_directories(tests PUBLIC ${ACE_INCLUDE_DIR})
target_link_libraries(tests PUBLIC gtest gtest_main flare_pp)
