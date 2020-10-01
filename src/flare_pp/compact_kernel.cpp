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

  double empty_thresh = 1e-8;

  for (int s = 0; s < n_species; s++){
    // Compute dot products. (Should be done in parallel with MKL.)
    Eigen::MatrixXd dot_vals =
      envs.descriptors[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_dot =
      envs.descriptors[s] * struc.descriptor_force_dervs[s].transpose();
    Eigen::MatrixXd stress_dot =
      envs.descriptors[s] * struc.descriptor_stress_dervs[s].transpose();
    
    Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s];
    Eigen::VectorXd struc_stress_dot = struc.descriptor_stress_dots[s];

    // Compute kernels. Can parallelize over environments.
    int n_sparse = envs.n_atoms[s];
    int n_struc = struc.n_atoms_by_species[s];
    int c_sparse = envs.c_atoms[s];

    for (int i = 0; i < n_sparse; i++){
      double norm_i = envs.descriptor_norms[s][i];

      // Continue if sparse environment i has no neighbors.
      if (norm_i < empty_thresh)
        continue;
      int sparse_index = c_sparse + i;

      for (int j = 0; j < n_struc; j++){
          // Energy kernel.
          double norm_j = struc.descriptor_norms[s](j);
          // Continue if atom j has no neighbors.
          if (norm_j < empty_thresh)
            continue;
          double norm_j_3 = norm_j * norm_j * norm_j;
          double norm_dot = dot_vals(i, j) / (norm_i * norm_j);
          kern_mat(sparse_index, 0) += pow(norm_dot, power);

          // TODO: Force kernel.
          int n_neigh = struc.neighbor_counts[s](j);
          int c_neigh = struc.cumulative_neighbor_counts[s](j);
          int atom_index = struc.atom_indices[s](j);

          for (int k = 0; k < n_neigh; k++){
              int neighbor_index = struc.neighbor_indices[s](c_neigh + k);

              for (int comp = 0; comp < 3; comp++){
                int force_index = 3 * (c_neigh + k) + comp;
                double f1 =
                  force_dot(i, force_index) / (norm_i * norm_j);
                double f2 =
                  dot_vals(i, j) * struc_force_dot(force_index) /
                  (norm_i * norm_j_3);
                double f3 = f1 - f2;
                double dval = power * pow(norm_dot, power - 1);
                double kern_val = dval * f3;

                kern_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                  sig2 * kern_val;
                kern_mat(sparse_index, 1 + 3 * atom_index + comp) +=
                  sig2 * kern_val;
              }
          }

          // TODO: Stress kernel.
      }
    }
  }

  return kern_mat;
}