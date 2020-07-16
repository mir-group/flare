# make executable
add_executable(tests test_y_grad.cpp test_radial.cpp
               test_structure.cpp test_local_environment.cpp test_bond_env.cpp
	       test_descriptor.cpp test_kernels.cpp test_sparse_gp.cpp
	       test_sparse_gp_dtc.cpp)

# target_include_directories(tests PUBLIC ${ACE_INCLUDE_DIR})
include_directories(../src/flare_pp)
target_link_libraries(tests PUBLIC gtest gtest_main flare)
