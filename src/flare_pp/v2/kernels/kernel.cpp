#include "kernel.h"
#include "cutoffs.h"
#include <cmath>
#include <iostream>

CompactKernel ::CompactKernel(){};
CompactKernel ::CompactKernel(Eigen::VectorXd kernel_hyperparameters) {
  this->kernel_hyperparameters = kernel_hyperparameters;
};
