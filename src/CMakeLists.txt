# make library
add_library(Ace y_grad.cpp y_grad.h radial.cpp radial.h cutoffs.cpp cutoffs.h)

set(ACE_INCLUDE_DIR "${ACE_ROOT_DIR}/src" PARENT_SCOPE)
