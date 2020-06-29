add_executable(test
	Descriptor/test_y_grad.cpp
	Descriptor/test_radial.cpp
	Descriptor/test_bond_env.cpp
	Descriptor/test_descriptor.cpp
	Environment/test_local_environment.cpp
	Kernels/test_kernels.cpp
	Sparse_GP/test_sparse_gp.cpp
	Structure/test_structure.cpp)

include_directories(../Sources/)
target_link_libraries(test PUBLIC gtest gtest_main flare)
