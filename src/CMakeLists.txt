# make library
add_library(Ace y_grad.cpp radial.cpp cutoffs.cpp single_bond.cpp
            structure.cpp local_environment.cpp descriptor.cpp
	    kernels.cpp sparse_gp.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(Ace PRIVATE OpenMP::OpenMP_CXX)

set(ACE_INCLUDE_DIR "${ACE_ROOT_DIR}/src" PARENT_SCOPE)
