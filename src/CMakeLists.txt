# make library
add_library(Ace y_grad.cpp radial.cpp cutoffs.cpp single_bond.cpp
            structure.cpp)

set(ACE_INCLUDE_DIR "${ACE_ROOT_DIR}/src" PARENT_SCOPE)
