# make executable
add_executable(tests test_y_grad.cpp test_radial.cpp
               test_structure.cpp test_local_environment.cpp test_single_bond.cpp
	       test_descriptor.cpp test_kernels.cpp test_sparse_gp.cpp
	       test_sparse_gp_dtc.cpp test_compact_structure.cpp
               test_n_body.cpp)

include_directories(../src/flare_pp)
target_link_libraries(tests PUBLIC gtest gtest_main flare)
