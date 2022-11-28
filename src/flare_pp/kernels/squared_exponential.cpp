#include "squared_exponential.h"
#include "descriptor.h"
#include "structure.h"
#include <iostream>
#include <stdio.h>
#undef NDEBUG
#include <assert.h>
#include <cmath>

SquaredExponential ::SquaredExponential(){};

SquaredExponential ::SquaredExponential(double sigma, double ls) {

  this->sigma = sigma;
  this->ls = ls;
  sig2 = sigma * sigma;
  ls2 = ls * ls;
  kernel_name = "SquaredExponential";

  // Set kernel hyperparameters.
  Eigen::VectorXd hyps(2);
  hyps << sigma, ls;
  kernel_hyperparameters = hyps;
}

Eigen::MatrixXd SquaredExponential ::envs_envs(const ClusterDescriptor &envs1,
                                               const ClusterDescriptor &envs2,
                                               const Eigen::VectorXd &hyps) {

  double sig2 = hyps(0) * hyps(0);
  double ls2 = hyps(1) * hyps(1);

  // Check types.
  int n_types_1 = envs1.n_types;
  int n_types_2 = envs2.n_types;
  assert(n_types_1 == n_types_2);

  // Check descriptor size.
  int n_descriptors_1 = envs1.n_descriptors;
  int n_descriptors_2 = envs2.n_descriptors;
  assert(n_descriptors_1 == n_descriptors_2);

  Eigen::MatrixXd kern_mat =
      Eigen::MatrixXd::Zero(envs1.n_clusters, envs2.n_clusters);
  int n_types = n_types_1;

  for (int s = 0; s < n_types; s++) {
    // Compute dot products.
    Eigen::MatrixXd dot_vals =
        envs1.descriptors[s] * envs2.descriptors[s].transpose();

    // Compute kernels.
    int n_sparse_1 = envs1.n_clusters_by_type[s];
    int c_sparse_1 = envs1.cumulative_type_count[s];
    int n_sparse_2 = envs2.n_clusters_by_type[s];
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
        double cut_j = envs2.cutoff_values[s](j);
        int ind2 = c_sparse_2 + j;

        // Energy kernel.
        double exp_arg = (norm_i2 + norm_j2 - 2 * dot_vals(i, j)) / (2 * ls2);
        kern_mat(ind1, ind2) += sig2 * cut_i * cut_j * exp(-exp_arg);
      }
    }
  }

  return kern_mat;
}

std::vector<Eigen::MatrixXd>
SquaredExponential ::envs_envs_grad(const ClusterDescriptor &envs1,
                                    const ClusterDescriptor &envs2,
                                    const Eigen::VectorXd &hyps) {

  // Set hyperparameters.
  double sig_new = hyps(0);
  double sig2_new = sig_new * sig_new;
  double ls_new = hyps(1);
  double ls2_new = ls_new * ls_new;

  // Check types.
  int n_types_1 = envs1.n_types;
  int n_types_2 = envs2.n_types;
  assert(n_types_1 == n_types_2);

  // Check descriptor size.
  int n_descriptors_1 = envs1.n_descriptors;
  int n_descriptors_2 = envs2.n_descriptors;
  assert(n_descriptors_1 == n_descriptors_2);

  Eigen::MatrixXd kern_mat =
      Eigen::MatrixXd::Zero(envs1.n_clusters, envs2.n_clusters);
  int n_types = n_types_1;
  Eigen::MatrixXd sig_mat =
      Eigen::MatrixXd::Zero(envs1.n_clusters, envs2.n_clusters);
  Eigen::MatrixXd ls_mat =
      Eigen::MatrixXd::Zero(envs1.n_clusters, envs2.n_clusters);

  for (int s = 0; s < n_types; s++) {
    // Compute dot products.
    Eigen::MatrixXd dot_vals =
        envs1.descriptors[s] * envs2.descriptors[s].transpose();

    // Compute kernels.
    int n_sparse_1 = envs1.n_clusters_by_type[s];
    int c_sparse_1 = envs1.cumulative_type_count[s];
    int n_sparse_2 = envs2.n_clusters_by_type[s];
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
        double cut_j = envs2.cutoff_values[s](j);
        int ind2 = c_sparse_2 + j;

        // Energy kernel.
        double val1 = norm_i2 + norm_j2 - 2 * dot_vals(i, j);
        double exp_arg = val1 / (2 * ls2_new);
        double val2 = exp(-exp_arg) * cut_i * cut_j;
        double en_kern = sig2_new * val2;
        kern_mat(ind1, ind2) += en_kern;
        sig_mat(ind1, ind2) += 2 * sig_new * val2;
        ls_mat(ind1, ind2) += 2 * en_kern * exp_arg / ls_new;
      }
    }
  }

  std::vector<Eigen::MatrixXd> kernel_gradients;
  kernel_gradients.push_back(kern_mat);
  kernel_gradients.push_back(sig_mat);
  kernel_gradients.push_back(ls_mat);
  return kernel_gradients;
}

Eigen::MatrixXd SquaredExponential ::envs_struc(const ClusterDescriptor &envs,
                                                const DescriptorValues &struc,
                                                const Eigen::VectorXd &hyps) {

  double sig2 = hyps(0) * hyps(0);
  double ls2 = hyps(1) * hyps(1);

  // Check types.
  int n_types_1 = envs.n_types;
  int n_types_2 = struc.n_types;
  assert(n_types_1 == n_types_2);

  // Check descriptor size.
  int n_descriptors_1 = envs.n_descriptors;
  int n_descriptors_2 = struc.n_descriptors;
  assert(n_descriptors_1 == n_descriptors_2);

  Eigen::MatrixXd kern_mat =
      Eigen::MatrixXd::Zero(envs.n_clusters, 1 + struc.n_atoms * 3 + 6);

  int n_types = envs.n_types;
  double vol_inv = 1 / struc.volume;

  for (int s = 0; s < n_types; s++) {
    // Compute dot products.
    Eigen::MatrixXd dot_vals =
        envs.descriptors[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_dot =
        envs.descriptors[s] * struc.descriptor_force_dervs[s].transpose();

    Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s];

    // Compute kernels, parallelizing over environments.
    int n_sparse = envs.n_clusters_by_type[s];
    int n_struc = struc.n_clusters_by_type[s];
    int c_sparse = envs.cumulative_type_count[s];

#pragma omp parallel for
    for (int i = 0; i < n_sparse; i++) {
      double norm_i = envs.descriptor_norms[s](i);
      double norm_i2 = norm_i * norm_i;
      double cut_i = envs.cutoff_values[s](i);
      int sparse_index = c_sparse + i;

      for (int j = 0; j < n_struc; j++) {
        double norm_j = struc.descriptor_norms[s](j);
        double norm_j2 = norm_j * norm_j;
        double cut_j = struc.cutoff_values[s](j);

        // Energy kernel.
        double exp_arg = (norm_i2 + norm_j2 - 2 * dot_vals(i, j)) / (2 * ls2);
        double exp_val = exp(-exp_arg);
        double en_kern = sig2 * exp_val * cut_i * cut_j;
        kern_mat(sparse_index, 0) += en_kern;

        // Force kernel.
        int n_neigh = struc.neighbor_counts[s](j);
        int c_neigh = struc.cumulative_neighbor_counts[s](j);
        int atom_index = struc.atom_indices[s](j);

        for (int k = 0; k < n_neigh; k++) {
          int neighbor_index = struc.neighbor_indices[s](c_neigh + k);
          int stress_counter = 0;
          int ind = c_neigh + k;

          for (int comp = 0; comp < 3; comp++) {
            int force_index = 3 * ind + comp;
            double cut_derv = struc.cutoff_dervs[s](force_index);
            double f1 = force_dot(i, force_index);
            double f2 = struc_force_dot(force_index);
            double f3 = (f1 - f2) / ls2;
            double force_kern_val =
                sig2 * exp_val * cut_i * (f3 * cut_j + cut_derv);

            kern_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                force_kern_val;
            kern_mat(sparse_index, 1 + 3 * atom_index + comp) += force_kern_val;

            for (int comp2 = comp; comp2 < 3; comp2++) {
              double coord = struc.neighbor_coordinates[s](ind, comp2);
              kern_mat(sparse_index, 1 + 3 * struc.n_atoms + stress_counter) -=
                  force_kern_val * coord * vol_inv;
              stress_counter++;
            }
          }
        }
      }
    }
  }

  return kern_mat;
}

std::vector<Eigen::MatrixXd>
SquaredExponential ::envs_struc_grad(const ClusterDescriptor &envs,
                                     const DescriptorValues &struc,
                                     const Eigen::VectorXd &hyps) {

  // Define hyperparameters.
  double sig_new = hyps(0);
  double sig2_new = sig_new * sig_new;
  double ls_new = hyps(1);
  double ls2_new = ls_new * ls_new;

  // Check types.
  int n_types_1 = envs.n_types;
  int n_types_2 = struc.n_types;
  assert(n_types_1 == n_types_2);

  // Check descriptor size.
  int n_descriptors_1 = envs.n_descriptors;
  int n_descriptors_2 = struc.n_descriptors;
  assert(n_descriptors_1 == n_descriptors_2);

  Eigen::MatrixXd kern_mat =
      Eigen::MatrixXd::Zero(envs.n_clusters, 1 + struc.n_atoms * 3 + 6);
  Eigen::MatrixXd sig_mat =
      Eigen::MatrixXd::Zero(envs.n_clusters, 1 + struc.n_atoms * 3 + 6);
  Eigen::MatrixXd ls_mat =
      Eigen::MatrixXd::Zero(envs.n_clusters, 1 + struc.n_atoms * 3 + 6);

  int n_types = envs.n_types;
  double vol_inv = 1 / struc.volume;

  for (int s = 0; s < n_types; s++) {
    // Compute dot products.
    Eigen::MatrixXd dot_vals =
        envs.descriptors[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_dot =
        envs.descriptors[s] * struc.descriptor_force_dervs[s].transpose();

    Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s];

    // Compute kernels, parallelizing over environments.
    int n_sparse = envs.n_clusters_by_type[s];
    int n_struc = struc.n_clusters_by_type[s];
    int c_sparse = envs.cumulative_type_count[s];

#pragma omp parallel for
    for (int i = 0; i < n_sparse; i++) {
      double norm_i = envs.descriptor_norms[s](i);
      double norm_i2 = norm_i * norm_i;
      double cut_i = envs.cutoff_values[s](i);
      int sparse_index = c_sparse + i;

      for (int j = 0; j < n_struc; j++) {
        double norm_j = struc.descriptor_norms[s](j);
        double norm_j2 = norm_j * norm_j;
        double cut_j = struc.cutoff_values[s](j);

        // Energy kernel.
        double exp_arg =
            (norm_i2 + norm_j2 - 2 * dot_vals(i, j)) / (2 * ls2_new);
        double exp_val = exp(-exp_arg);
        double en_kern = sig2_new * exp_val * cut_i * cut_j;
        double sig_derv = 2 * en_kern / sig_new;
        double ls_derv = 2 * en_kern * exp_arg / ls_new;
        kern_mat(sparse_index, 0) += en_kern;
        sig_mat(sparse_index, 0) += sig_derv;
        ls_mat(sparse_index, 0) += ls_derv;

        // Force kernel.
        int n_neigh = struc.neighbor_counts[s](j);
        int c_neigh = struc.cumulative_neighbor_counts[s](j);
        int atom_index = struc.atom_indices[s](j);

        for (int k = 0; k < n_neigh; k++) {
          int neighbor_index = struc.neighbor_indices[s](c_neigh + k);
          int stress_counter = 0;
          int ind = c_neigh + k;

          for (int comp = 0; comp < 3; comp++) {
            int force_index = 3 * ind + comp;
            double cut_derv = struc.cutoff_dervs[s](force_index);
            double f1 = force_dot(i, force_index);
            double f2 = struc_force_dot(force_index);
            double f3 = (f1 - f2) / ls2_new;

            double force_kern_val =
                sig2_new * exp_val * cut_i * (f3 * cut_j + cut_derv);
            double sig_force_derv = sig_derv * (f3 + cut_derv / cut_j);
            double ls_force_derv =
                ls_derv * (f3 + cut_derv / cut_j) - 2 * en_kern * f3 / ls_new;

            kern_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                force_kern_val;
            kern_mat(sparse_index, 1 + 3 * atom_index + comp) += force_kern_val;

            sig_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                sig_force_derv;
            sig_mat(sparse_index, 1 + 3 * atom_index + comp) += sig_force_derv;

            ls_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                ls_force_derv;
            ls_mat(sparse_index, 1 + 3 * atom_index + comp) += ls_force_derv;

            for (int comp2 = comp; comp2 < 3; comp2++) {
              double coord = struc.neighbor_coordinates[s](ind, comp2);
              kern_mat(sparse_index, 1 + 3 * struc.n_atoms + stress_counter) -=
                  force_kern_val * coord * vol_inv;
              sig_mat(sparse_index, 1 + 3 * struc.n_atoms + stress_counter) -=
                  sig_force_derv * coord * vol_inv;
              ls_mat(sparse_index, 1 + 3 * struc.n_atoms + stress_counter) -=
                  ls_force_derv * coord * vol_inv;
              stress_counter++;
            }
          }
        }
      }
    }
  }

  std::vector<Eigen::MatrixXd> kernel_gradients;
  kernel_gradients.push_back(kern_mat);
  kernel_gradients.push_back(sig_mat);
  kernel_gradients.push_back(ls_mat);
  return kernel_gradients;
}

Eigen::MatrixXd SquaredExponential ::struc_struc(const DescriptorValues &struc1,
                                                 const DescriptorValues &struc2,
                                                 const Eigen::VectorXd &hyps) {

  double sig2 = hyps(0) * hyps(0);
  double ls2 = hyps(1) * hyps(1);

  int n_elements_1 = 1 + 3 * struc1.n_atoms + 6;
  int n_elements_2 = 1 + 3 * struc2.n_atoms + 6;
  Eigen::MatrixXd kernel_matrix =
      Eigen::MatrixXd::Zero(n_elements_1, n_elements_2);

  // Check types.
  int n_types_1 = struc1.n_types;
  int n_types_2 = struc2.n_types;
  assert(n_types_1 == n_types_2);

  // Check descriptor size.
  int n_descriptors_1 = struc1.n_descriptors;
  int n_descriptors_2 = struc2.n_descriptors;
  assert(n_descriptors_1 == n_descriptors_2);

  double vol_inv_1 = 1 / struc1.volume;
  double vol_inv_2 = 1 / struc2.volume;

  std::vector<int> stress_inds{0, 3, 5};
  double empty_thresh = 1e-8;

  for (int s = 0; s < n_types_1; s++) {
    // Compute dot products.
    Eigen::MatrixXd dot_vals =
        struc1.descriptors[s] * struc2.descriptors[s].transpose();
    Eigen::MatrixXd force_dot_1 =
        struc1.descriptor_force_dervs[s] * struc2.descriptors[s].transpose();
    Eigen::MatrixXd force_dot_2 =
        struc2.descriptor_force_dervs[s] * struc1.descriptors[s].transpose();
    Eigen::MatrixXd force_force = struc1.descriptor_force_dervs[s] *
                                  struc2.descriptor_force_dervs[s].transpose();

    Eigen::VectorXd struc_force_dot_1 = struc1.descriptor_force_dots[s];
    Eigen::VectorXd struc_force_dot_2 = struc2.descriptor_force_dots[s];

    // Compute kernels.
    int n_struc1 = struc1.n_clusters_by_type[s];
    int n_struc2 = struc2.n_clusters_by_type[s];

    for (int i = 0; i < n_struc1; i++) {
      double norm_i = struc1.descriptor_norms[s](i);

      // Continue if atom j has no neighbors.
      if (norm_i < empty_thresh)
        continue;

      double norm_i2 = norm_i * norm_i;
      double cut_i = struc1.cutoff_values[s](i);

      for (int j = 0; j < n_struc2; j++) {
        double norm_j = struc2.descriptor_norms[s](j);

        // Continue if atom j has no neighbors.
        if (norm_j < empty_thresh)
          continue;

        double norm_j2 = norm_j * norm_j;
        double cut_j = struc2.cutoff_values[s](j);

        // Energy kernel.
        double exp_arg = (norm_i2 + norm_j2 - 2 * dot_vals(i, j)) / (2 * ls2);
        double exp_val = exp(-exp_arg);
        double en_kern = sig2 * exp_val * cut_i * cut_j;
        kernel_matrix(0, 0) += en_kern;

        int n_neigh_1 = struc1.neighbor_counts[s](i);
        int c_neigh_1 = struc1.cumulative_neighbor_counts[s](i);
        int c_ind_1 = struc1.atom_indices[s](i);

        int n_neigh_2 = struc2.neighbor_counts[s](j);
        int c_neigh_2 = struc2.cumulative_neighbor_counts[s](j);
        int c_ind_2 = struc2.atom_indices[s](j);

        // Energy/force and energy/stress kernels.
        for (int k = 0; k < n_neigh_2; k++) {
          int ind = c_neigh_2 + k;
          int neighbor_index = struc2.neighbor_indices[s](ind);
          int stress_counter = 0;

          for (int comp = 0; comp < 3; comp++) {
            int force_index = 3 * ind + comp;
            double cut_derv_2 = struc2.cutoff_dervs[s](force_index);
            double f1 = force_dot_2(force_index, i);
            double f2 = struc_force_dot_2(force_index);
            double f3 = (f1 - f2) / ls2;
            double force_kern_val =
                sig2 * exp_val * cut_i * (f3 * cut_j + cut_derv_2);

            // Energy/force.
            kernel_matrix(0, 1 + 3 * neighbor_index + comp) -= force_kern_val;
            kernel_matrix(0, 1 + 3 * c_ind_2 + comp) += force_kern_val;

            // Energy/stress.
            for (int comp2 = comp; comp2 < 3; comp2++) {
              double coord = struc2.neighbor_coordinates[s](ind, comp2);
              kernel_matrix(0, 1 + 3 * struc2.n_atoms + stress_counter) -=
                  force_kern_val * coord * vol_inv_2;
              stress_counter++;
            }
          }
        }

        // Force/energy and stress/energy kernels.
        for (int k = 0; k < n_neigh_1; k++) {
          int ind = c_neigh_1 + k;
          int neighbor_index = struc1.neighbor_indices[s](ind);
          int stress_counter = 0;

          for (int comp = 0; comp < 3; comp++) {
            int force_index = 3 * ind + comp;
            double cut_derv_1 = struc1.cutoff_dervs[s](force_index);
            double f1 = force_dot_1(force_index, j);
            double f2 = struc_force_dot_1(force_index);
            double f3 = (f1 - f2) / ls2;
            double force_kern_val =
                sig2 * exp_val * cut_j * (f3 * cut_i + cut_derv_1);

            // Force/energy.
            kernel_matrix(1 + 3 * neighbor_index + comp, 0) -= force_kern_val;
            kernel_matrix(1 + 3 * c_ind_1 + comp, 0) += force_kern_val;

            // Stress/energy.
            for (int comp2 = comp; comp2 < 3; comp2++) {
              double coord = struc1.neighbor_coordinates[s](ind, comp2);
              kernel_matrix(1 + 3 * struc1.n_atoms + stress_counter, 0) -=
                  force_kern_val * coord * vol_inv_1;
              stress_counter++;
            }
          }
        }

        // Force/force, force/stress, stress/force, and stress/stress kernels.
        for (int k = 0; k < n_neigh_1; k++) {
          int ind1 = c_neigh_1 + k;
          int n_ind_1 = struc1.neighbor_indices[s](ind1);

          for (int l = 0; l < n_neigh_2; l++) {
            int ind2 = c_neigh_2 + l;
            int n_ind_2 = struc2.neighbor_indices[s](ind2);

            for (int m = 0; m < 3; m++) {
              int f_ind_1 = 3 * ind1 + m;
              double cut_derv_1 = struc1.cutoff_dervs[s](f_ind_1);
              for (int n = 0; n < 3; n++) {
                int f_ind_2 = 3 * ind2 + n;
                double cut_derv_2 = struc2.cutoff_dervs[s](f_ind_2);

                double v1 =
                    (force_dot_1(f_ind_1, j) - struc_force_dot_1(f_ind_1)) /
                    ls2;
                double v2 =
                    (force_dot_2(f_ind_2, i) - struc_force_dot_2(f_ind_2)) /
                    ls2;
                double v3 = force_force(f_ind_1, f_ind_2) / ls2;

                double v4 = cut_i * cut_derv_2 * v1;
                double v5 = cut_derv_1 * cut_j * v2;
                double v6 = cut_i * cut_j * v1 * v2;
                double v7 = cut_i * cut_j * v3;
                double v8 = cut_derv_1 * cut_derv_2;

                double kern_val = sig2 * exp_val * (v4 + v5 + v6 + v7 + v8);

                // Force/force.
                kernel_matrix(1 + c_ind_1 * 3 + m, 1 + c_ind_2 * 3 + n) +=
                    kern_val;
                kernel_matrix(1 + c_ind_1 * 3 + m, 1 + n_ind_2 * 3 + n) -=
                    kern_val;
                kernel_matrix(1 + n_ind_1 * 3 + m, 1 + c_ind_2 * 3 + n) -=
                    kern_val;
                kernel_matrix(1 + n_ind_1 * 3 + m, 1 + n_ind_2 * 3 + n) +=
                    kern_val;

                // Stress/force.
                int stress_ind_1 = stress_inds[m];
                for (int p = m; p < 3; p++) {
                  double coord = struc1.neighbor_coordinates[s](ind1, p);
                  kernel_matrix(1 + 3 * struc1.n_atoms + stress_ind_1,
                                1 + c_ind_2 * 3 + n) -=
                      kern_val * coord * vol_inv_1;
                  kernel_matrix(1 + 3 * struc1.n_atoms + stress_ind_1,
                                1 + n_ind_2 * 3 + n) +=
                      kern_val * coord * vol_inv_1;
                  stress_ind_1++;
                }

                // Force/stress.
                int stress_ind_2 = stress_inds[n];
                for (int p = n; p < 3; p++) {
                  double coord = struc2.neighbor_coordinates[s](ind2, p);
                  kernel_matrix(1 + c_ind_1 * 3 + m,
                                1 + 3 * struc2.n_atoms + stress_ind_2) -=
                      kern_val * coord * vol_inv_2;
                  kernel_matrix(1 + n_ind_1 * 3 + m,
                                1 + 3 * struc2.n_atoms + stress_ind_2) +=
                      kern_val * coord * vol_inv_2;
                  stress_ind_2++;
                }

                // Stress/stress.
                stress_ind_1 = stress_inds[m];
                for (int p = m; p < 3; p++) {
                  double coord1 = struc1.neighbor_coordinates[s](ind1, p);
                  stress_ind_2 = stress_inds[n];
                  for (int q = n; q < 3; q++) {
                    double coord2 = struc2.neighbor_coordinates[s](ind2, q);
                    kernel_matrix(1 + 3 * struc1.n_atoms + stress_ind_1,
                                  1 + 3 * struc2.n_atoms + stress_ind_2) +=
                        kern_val * coord1 * coord2 * vol_inv_1 * vol_inv_2;
                    stress_ind_2++;
                  }
                  stress_ind_1++;
                }
              }
            }
          }
        }
      }
    }
  }

  return kernel_matrix;
}

Eigen::VectorXd
SquaredExponential ::self_kernel_struc(const DescriptorValues &struc,
                                       const Eigen::VectorXd &hyps) {

  // Note: This can be made slightly faster by ignoring off-diagonal
  // kernel values (see normalized dot product implementation))
  int n_elements = 1 + 3 * struc.n_atoms + 6;
  Eigen::MatrixXd kernel_matrix = struc_struc(struc, struc, hyps);
  Eigen::VectorXd kernel_vector = kernel_matrix.diagonal();

  return kernel_vector;
}

void SquaredExponential ::set_hyperparameters(Eigen::VectorXd hyps) {
  sigma = hyps(0);
  ls = hyps(1);
  sig2 = sigma * sigma;
  ls2 = ls * ls;
  kernel_hyperparameters = hyps;
}

Eigen::MatrixXd
SquaredExponential ::compute_mapping_coefficients(const SparseGP &gp_model,
                                                  int kernel_index) {

  std::cout
      << "Mapping coefficients are not implemented for the squared exponential "
         "kernel."
      << std::endl;

  Eigen::MatrixXd empty_mat;
  return empty_mat;
}

Eigen::MatrixXd
SquaredExponential ::compute_varmap_coefficients(const SparseGP &gp_model,
                                                  int kernel_index) {

  std::cout
      << "Mapping coefficients are not implemented for the squared exponential "
         "kernel."
      << std::endl;

  Eigen::MatrixXd empty_mat;
  return empty_mat;
}

void SquaredExponential ::write_info(std::ofstream &coeff_file) {
  std::cout << "Not implemented." << std::endl;
}

nlohmann::json SquaredExponential ::return_json(){
  nlohmann::json j;
  to_json(j, *this);
  return j;
}
