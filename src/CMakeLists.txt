# make library
add_library(YGrad y_grad.cpp y_grad.h radial.cpp radial.h)

set(YGRAD_INCLUDE_DIR "${ACE_ROOT_DIR}/src" PARENT_SCOPE)
