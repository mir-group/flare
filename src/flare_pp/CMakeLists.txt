# make library
add_library(flare_pp SHARED y_grad.cpp radial.cpp cutoffs.cpp single_bond.cpp
            structure.cpp local_environment.cpp descriptor.cpp
	        kernels.cpp sparse_gp.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flare_pp PUBLIC OpenMP::OpenMP_CXX)
endif()

set(ACE_INCLUDE_DIR "${ACE_ROOT_DIR}/src/flare_pp" PARENT_SCOPE)
