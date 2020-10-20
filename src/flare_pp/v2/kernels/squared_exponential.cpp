#include "squared_exponential.h"
#include "compact_environments.h"
#include "compact_structure.h"
#include "compact_descriptor.h"
#include <assert.h>
#include <cmath>
#include <iostream>

SquaredExponential ::SquaredExponential(){};

SquaredExponential ::SquaredExponential(double sigma, double ls) {

  this->sigma = sigma;
  this->ls = ls;
  sig2 = sigma * sigma;
  ls2 = ls * ls;

  // Set kernel hyperparameters.
  Eigen::VectorXd hyps(2);
  hyps << sigma, ls;
  kernel_hyperparameters = hyps;
}

Eigen::MatrixXd SquaredExponential ::envs_envs(
  const ClusterDescriptor &envs1, const ClusterDescriptor &envs2) {

  // Check types.
  int n_types_1 = envs1.n_types;
  int n_types_2 = envs2.n_types;
  bool type_check = (n_types_1 == n_types_2);
  assert(("Types don't match.", type_check));

  // Check descriptor size.
  int n_descriptors_1 = envs1.n_descriptors;
  int n_descriptors_2 = envs2.n_descriptors;
  bool descriptor_check = (n_descriptors_1 == n_descriptors_2);
  assert(("Descriptors don't match.", descriptor_check));

  Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(
    envs1.n_clusters, envs2.n_clusters);
  int n_types = n_types_1;

  for (int s = 0; s < n_types; s++) {
    // Compute dot products. (Should be done in parallel with MKL.)
    Eigen::MatrixXd dot_vals =
        envs1.descriptors[s] * envs2.descriptors[s].transpose();

    // Compute kernels.
    int n_sparse_1 = envs1.type_count[s];
    int c_sparse_1 = envs1.cumulative_type_count[s];
    int n_sparse_2 = envs2.type_count[s];
    int c_sparse_2 = envs2.cumulative_type_count[s];

#pragma omp parallel for
    for (int i = 0; i < n_sparse_1; i++) {
      double norm_i = envs1.descriptor_norms[s](i);
      double norm_i2 = norm_i * norm_i;
      double cut_i = envs1.cutoff_values[s](i);
      int ind1 = c_sparse_1 + i;

      for (int j = 0; j < n_sparse_2; j++) {
        double norm_j = envs2.descriptor_norms[s](j);
        double norm_j2 = norm_j * norm_j;
        double cut_j = envs2.cutoff_values[s](i);
        int ind2 = c_sparse_2 + j;

        // Energy kernel.
        double exp_arg =
          (norm_i2 + norm_j2 - 2 * dot_vals(i, j)) / (2 * ls * ls);
        kern_mat(ind1, ind2) += sig2 * cut_i * cut_j * exp(-exp_arg);
      }
    }
  }

  return kern_mat;
}

Eigen::MatrixXd SquaredExponential ::envs_struc(
  const ClusterDescriptor &envs, const DescriptorValues &struc) {

  // Check types.
  int n_types_1 = envs.n_types;
  int n_types_2 = struc.n_types;
  bool type_check = (n_types_1 == n_types_2);
  assert(("Types don't match.", type_check));

  // Check descriptor size.
  int n_descriptors_1 = envs.n_descriptors;
  int n_descriptors_2 = struc.n_descriptors;
  bool descriptor_check = (n_descriptors_1 == n_descriptors_2);
  assert(("Descriptors don't match.", descriptor_check));

  Eigen::MatrixXd kern_mat =
      Eigen::MatrixXd::Zero(envs.n_clusters, 1 + struc.n_atoms * 3 + 6);

  return kern_mat;
}

Eigen::MatrixXd SquaredExponential ::struc_struc(
  DescriptorValues struc1, DescriptorValues struc2) {

  int n_elements_1 = 1 + 3 * struc1.n_atoms + 6;
  int n_elements_2 = 1 + 3 * struc2.n_atoms + 6;
  Eigen::MatrixXd kernel_matrix =
      Eigen::MatrixXd::Zero(n_elements_1, n_elements_2);

  return kernel_matrix;
}

Eigen::VectorXd
SquaredExponential ::self_kernel_struc(DescriptorValues struc) {

  int n_elements = 1 + 3 * struc.n_atoms + 6;
  Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(n_elements);

  return kernel_vector;
}
