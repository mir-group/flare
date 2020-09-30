#include "compact_kernel.h"
#include "compact_structure.h"
#include <cmath>
#include <iostream>

CompactKernel ::CompactKernel(){};

CompactKernel ::CompactKernel(double sigma, double power) {

  this->sigma = sigma;
  sig2 = sigma * sigma;
  this->power = power;
}

// TODO: Make compact environments class.
Eigen::VectorXd
CompactKernel ::envs_struc(const std::vector<Eigen::MatrixXd> &envs1,
                           const CompactStructure &struc){

  Eigen::VectorXd kern_mat = Eigen::VectorXd::Zero(1 + struc.noa * 3 + 6);
  int n_species = envs1.size();

  for (int s = 0; s < n_species; s++){
    // Compute dot products. (Parallel with MKL.)
    Eigen::MatrixXd dot_vals = envs1[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_dot =
      envs1[s] * struc.descriptor_force_dervs[s].transpose();
    Eigen::MatrixXd stress_dot =
      envs1[s] * struc.descriptor_stress_dervs[s].transpose();

    // Compute kernels. Possible to parallelize over environments.
  }

  return kern_mat;
}