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

Eigen::MatrixXd CompactKernel ::envs_envs(const CompactEnvironments &envs1,
                                          const CompactEnvironments &envs2){

  Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(envs1.n_envs, envs2.n_envs);
  int n_species = envs1.n_species;
  double empty_thresh = 1e-8;

  for (int s = 0; s < n_species; s++){
    // Compute dot products. (Should be done in parallel with MKL.)
    Eigen::MatrixXd dot_vals =
      envs1.descriptors[s] * envs2.descriptors[s].transpose();

    // Compute kernels.
    int n_sparse_1 = envs1.n_atoms[s];
    int c_sparse_1 = envs1.c_atoms[s];
    int n_sparse_2 = envs2.n_atoms[s];
    int c_sparse_2 = envs2.c_atoms[s];

#pragma omp parallel for
    for (int i = 0; i < n_sparse_1; i++){
      double norm_i = envs1.descriptor_norms[s][i];

      // Continue if sparse environment i has no neighbors.
      if (norm_i < empty_thresh)
        continue;
      int ind1 = c_sparse_1 + i;

      for (int j = 0; j < n_sparse_2; j++){
          double norm_j = envs2.descriptor_norms[s][j];
          double norm_ij = norm_i * norm_j;

          // Continue if atom j has no neighbors.
          if (norm_j < empty_thresh)
            continue;
          int ind2 = c_sparse_2 + j;

          // Energy kernel.
          double norm_dot = dot_vals(i, j) / norm_ij;
          double dval = power * pow(norm_dot, power - 1);
          kern_mat(ind1, ind2) += sig2 * pow(norm_dot, power);
      }
    }
  }
  return kern_mat;
}

Eigen::MatrixXd CompactKernel ::envs_struc(const CompactEnvironments &envs,
                                           const CompactStructure &struc){

  Eigen::MatrixXd kern_mat =
    Eigen::MatrixXd::Zero(envs.n_envs, 1 + struc.noa * 3 + 6);
  int n_species = envs.n_species;
  double vol_inv = 1 / struc.volume;
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

#pragma omp parallel for
    for (int i = 0; i < n_sparse; i++){
      double norm_i = envs.descriptor_norms[s][i];

      // Continue if sparse environment i has no neighbors.
      if (norm_i < empty_thresh)
        continue;
      int sparse_index = c_sparse + i;

      for (int j = 0; j < n_struc; j++){
          double norm_j = struc.descriptor_norms[s](j);
          double norm_ij = norm_i * norm_j;
          double norm_ij3 = norm_ij * norm_j * norm_j;

          // Continue if atom j has no neighbors.
          if (norm_j < empty_thresh)
            continue;

          // Energy kernel.
          double norm_dot = dot_vals(i, j) / norm_ij;
          double dval = power * pow(norm_dot, power - 1);
          kern_mat(sparse_index, 0) += sig2 * pow(norm_dot, power);

          // Force kernel.
          int n_neigh = struc.neighbor_counts[s](j);
          int c_neigh = struc.cumulative_neighbor_counts[s](j);
          int atom_index = struc.atom_indices[s](j);

          for (int k = 0; k < n_neigh; k++){
              int neighbor_index = struc.neighbor_indices[s](c_neigh + k);

              for (int comp = 0; comp < 3; comp++){
                int force_index = 3 * (c_neigh + k) + comp;
                double f1 = force_dot(i, force_index) / norm_ij;
                double f2 =
                  dot_vals(i, j) * struc_force_dot(force_index) / norm_ij3;
                double f3 = f1 - f2;
                double kern_val = dval * f3;

                kern_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                  sig2 * kern_val;
                kern_mat(sparse_index, 1 + 3 * atom_index + comp) +=
                  sig2 * kern_val;
              }
          }

          // Stress kernel.
          for (int comp = 0; comp < 6; comp++){
              int stress_index = j * 6 + comp;
              double s1 = stress_dot(i, stress_index) / norm_ij;
              double s2 =
                dot_vals(i, j) * struc_stress_dot(stress_index) / norm_ij3;
              double s3 = s1 - s2;
              double kern_val = dval * s3;
              kern_mat(sparse_index, 1 + 3 * struc.noa + comp) +=
                -sig2 * kern_val * vol_inv;
          }
      }
    }
  }

  return kern_mat;
}

Eigen::VectorXd CompactKernel ::self_kernel_struc(
    const CompactStructure &struc){

  int n_elements = 1 + 3 * struc.noa + 6;
  Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(n_elements);

  int n_species = struc.n_species;
  double vol_inv = 1 / struc.volume;
  double empty_thresh = 1e-8;

  for (int s = 0; s < n_species; s++){
    // Compute dot products. (Should be done in parallel with MKL.)
    Eigen::MatrixXd dot_vals =
      struc.descriptors[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_dot =
      struc.descriptor_force_dervs[s] *
      struc.descriptors[s].transpose();
    Eigen::MatrixXd stress_dot =
      struc.descriptor_stress_dervs[s] *
      struc.descriptors[s].transpose();
    Eigen::MatrixXd force_force =
      struc.descriptor_force_dervs[s] *
      struc.descriptor_force_dervs[s].transpose();
    Eigen::MatrixXd stress_stress =
      struc.descriptor_stress_dervs[s] *
      struc.descriptor_stress_dervs[s].transpose();

    Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s];
    Eigen::VectorXd struc_stress_dot = struc.descriptor_stress_dots[s];

    // Compute kernels.
    int n_struc = struc.n_atoms_by_species[s];

    for (int i = 0; i < n_struc; i++){
      double norm_i = struc.descriptor_norms[s](i);

      // Continue if atom i has no neighbors.
      if (norm_i < empty_thresh)
        continue;

      double norm_i2 = norm_i * norm_i;
      double norm_i3 = norm_i2 * norm_i;

      for (int j = i; j < n_struc; j++){
          double norm_j = struc.descriptor_norms[s](j);

          // Continue if atom j has no neighbors.
          if (norm_j < empty_thresh)
            continue;

          double mult_fac;
          if (i == j)
            mult_fac = 1;
          else
            mult_fac = 2;

          double norm_j2 = norm_j * norm_j;
          double norm_j3 = norm_j2 * norm_j;
          double norm_ij = norm_i * norm_j;

          // Energy kernel.
          double norm_dot = dot_vals(i, j) / norm_ij;
          double c1 = (power - 1) * power * pow(norm_dot, power - 2);
          double c2 = power * pow(norm_dot, power - 1);
          kernel_vector(0) += sig2 * mult_fac * pow(norm_dot, power);

          // Force kernel.
          int n_neigh_1 = struc.neighbor_counts[s](i);
          int c_neigh_1 = struc.cumulative_neighbor_counts[s](i);
          int c_ind_1 = struc.atom_indices[s](i);

          int n_neigh_2 = struc.neighbor_counts[s](j);
          int c_neigh_2 = struc.cumulative_neighbor_counts[s](j);
          int c_ind_2 = struc.atom_indices[s](j);

          for (int k = 0; k < n_neigh_1; k++){
            int ind1 = c_neigh_1 + k;
            int n_ind_1 = struc.neighbor_indices[s](ind1);

            for (int l = 0; l < n_neigh_2; l++){
                int ind2 = c_neigh_2 + l;
                int n_ind_2 = struc.neighbor_indices[s](ind2);

                // Check if any pairs of atoms match.
                int match_check = 0;
                if ((c_ind_1 == c_ind_2) || (c_ind_1 == n_ind_2) ||
                    (n_ind_1 == c_ind_2) || (n_ind_1 == n_ind_2))
                  match_check = 1;

                if (match_check == 0)
                  continue;

                for (int m = 0; m < 3; m++){
                    int f_ind_1 = 3 * ind1 + m;
                    int f_ind_2 = 3 * ind2 + m;
                    double v1 = force_dot(f_ind_1, j) / norm_ij - 
                      norm_dot * struc_force_dot(f_ind_1) / norm_i2;
                    double v2 = force_dot(f_ind_2, i) / norm_ij -
                      norm_dot * struc_force_dot(f_ind_2) / norm_j2;
                    double v3 = force_force(f_ind_1, f_ind_2) / norm_ij;
                    double v4 = struc_force_dot(f_ind_1) *
                      force_dot(f_ind_2, i) /  (norm_i3 * norm_j);
                    double v5 = struc_force_dot(f_ind_2) *
                      force_dot(f_ind_1, j) / (norm_i * norm_j3);
                    double v6 = struc_force_dot(f_ind_1) *
                      struc_force_dot(f_ind_2) * norm_dot / (norm_i2 * norm_j2);
                    
                    double kern_val =
                      sig2 * mult_fac *
                      (c1 * v1 * v2 + c2 * (v3 - v4 - v5 + v6));

                    if (c_ind_1 == c_ind_2)
                      kernel_vector(1 + c_ind_1 * 3 + m) += kern_val;
                    if (c_ind_1 == n_ind_2)
                      kernel_vector(1 + c_ind_1 * 3 + m) -= kern_val;
                    if (n_ind_1 == c_ind_2)
                      kernel_vector(1 + n_ind_1 * 3 + m) -= kern_val;
                    if (n_ind_1 == n_ind_2)
                      kernel_vector(1 + n_ind_1 * 3 + m) += kern_val;
                }
            }
          }

          // Stress kernel.
          for (int comp = 0; comp < 6; comp++){
            int s_ind_1 = i * 6 + comp;
            int s_ind_2 = j * 6 + comp;

            double v1 = stress_dot(s_ind_1, j) / norm_ij - 
              norm_dot * struc_stress_dot(s_ind_1) / norm_i2;
            double v2 = stress_dot(s_ind_2, i) / norm_ij -
              norm_dot * struc_stress_dot(s_ind_2) / norm_j2;
            double v3 = stress_stress(s_ind_1, s_ind_2) / norm_ij;
            double v4 = struc_stress_dot(s_ind_1) *
              stress_dot(s_ind_2, i) /  (norm_i3 * norm_j);
            double v5 = struc_stress_dot(s_ind_2) *
              stress_dot(s_ind_1, j) / (norm_i * norm_j3);
            double v6 = struc_stress_dot(s_ind_1) *
              struc_stress_dot(s_ind_2) * norm_dot / (norm_i2 * norm_j2);
            
            double kern_val = sig2 * mult_fac *
              (c1 * v1 * v2 + c2 * (v3 - v4 - v5 + v6));

            kernel_vector(1 + 3 * struc.noa + comp) +=
              kern_val * vol_inv * vol_inv;
          }
      }
    }
  }

  return kernel_vector;
}
