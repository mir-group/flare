#include "compact_kernel.h"
#include "compact_structure.h"
#include "compact_environments.h"
#include <cmath>
#include <iostream>

CompactKernel ::CompactKernel(){};

CompactKernel ::CompactKernel(double sigma, double power) {

  this->sigma = sigma;
  sig2 = sigma * sigma;
  this->power = power;
}

Eigen::MatrixXd CompactKernel ::envs_struc(const CompactEnvironments &envs,
                                           const CompactStructure &struc){

  Eigen::MatrixXd kern_mat =
    Eigen::MatrixXd::Zero(envs.n_envs, 1 + struc.noa * 3 + 6);
  int n_species = envs.n_species;

  for (int s = 0; s < n_species; s++){
    // Compute dot products. (Should be done in parallel with MKL.)
    Eigen::MatrixXd dot_vals =
      envs.descriptors[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_dot =
      envs.descriptors[s] * struc.descriptor_force_dervs[s].transpose();
    Eigen::MatrixXd stress_dot =
      envs.descriptors[s] * struc.descriptor_stress_dervs[s].transpose();

    // Compute kernels. Can parallelize over environments.
    int n_sparse = envs.n_atoms[s];
    int n_struc = struc.n_atoms_by_species[s];
    int c_sparse = envs.c_atoms[s];
    for (int i = 0; i < n_sparse; i++){
      double norm_i = envs.descriptor_norms[s][i];
      for (int j = 0; j < n_struc; j++){
          // Energy kernel.
          double norm_j = struc.descriptor_norms[s](j);
          double norm_dot = dot_vals(i, j) / (norm_i * norm_j);
          kern_mat(c_sparse + i, 0) += pow(norm_dot, power);

          // TODO: Force kernel.
          int n_neigh = struc.neighbor_counts[s](j);
          int c_neigh = struc.cumulative_neighbor_counts[s](j);
          int atom_index = struc.atom_indices[s](j);
          for (int k = 0; k < n_neigh; k++){
              int neighbor_index = struc.neighbor_indices[s](c_neigh + k);
              for (int comp = 0; comp < 3; comp++){

              }
          }

          // TODO: Stress kernel.
      }
    }
  }

  return kern_mat;
}