#include "norm_dot_icm.h"
#include "descriptor.h"
#include "sparse_gp.h"
#include "structure.h"
#include <algorithm>
#undef NDEBUG
#include <assert.h>
#include <cmath>
#include <iostream>

NormalizedDotProduct_ICM ::NormalizedDotProduct_ICM(){};

NormalizedDotProduct_ICM ::NormalizedDotProduct_ICM(
    double sigma, double power, Eigen::MatrixXd icm_coeffs) {

  this->sigma = sigma;
  sig2 = sigma * sigma;
  this->power = power;
  this->icm_coeffs = icm_coeffs;

  // Set kernel hyperparameters.
  n_icm_coeffs = icm_coeffs.rows() * (icm_coeffs.rows() + 1) / 2;
  Eigen::VectorXd hyps(1 + n_icm_coeffs);
  hyps << sigma;
  for (int i = 0; i < n_icm_coeffs; i++) {
    for (int j = i; j < n_icm_coeffs; j++) {
      hyps << icm_coeffs(i, j);
    }
  }
  kernel_hyperparameters = hyps;
}

Eigen::MatrixXd
NormalizedDotProduct_ICM ::envs_envs(const ClusterDescriptor &envs1,
                                     const ClusterDescriptor &envs2,
                                     const Eigen::VectorXd &hyps) {

  // Set square of the signal variance.
  double sig_sq = hyps(0) * hyps(0);

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
  double empty_thresh = 1e-8;

  for (int s1 = 0; s1 < n_types; s1++) {
    for (int s2 = 0; s2 < n_types; s2++) {
      int icm_index = get_icm_index(s1, s2, n_types);
      double icm_val = hyps(1 + icm_index);

      // Compute dot products. (Should be done in parallel with MKL.)
      Eigen::MatrixXd dot_vals =
          envs1.descriptors[s1] * envs2.descriptors[s2].transpose();

      // Compute kernels.
      int n_sparse_1 = envs1.n_clusters_by_type[s1];
      int c_sparse_1 = envs1.cumulative_type_count[s1];
      int n_sparse_2 = envs2.n_clusters_by_type[s2];
      int c_sparse_2 = envs2.cumulative_type_count[s2];

#pragma omp parallel for
      for (int i = 0; i < n_sparse_1; i++) {
        double norm_i = envs1.descriptor_norms[s1](i);

        // Continue if sparse environment i has no neighbors.
        if (norm_i < empty_thresh)
          continue;
        int ind1 = c_sparse_1 + i;

        for (int j = 0; j < n_sparse_2; j++) {
          double norm_j = envs2.descriptor_norms[s2](j);
          double norm_ij = norm_i * norm_j;

          // Continue if atom j has no neighbors.
          if (norm_j < empty_thresh)
            continue;
          int ind2 = c_sparse_2 + j;

          // Energy kernel.
          double norm_dot = dot_vals(i, j) / norm_ij;
          double kern_val = sig_sq * icm_val * pow(norm_dot, power);
          kern_mat(ind1, ind2) += kern_val;
        }
      }
    }
  }
  return kern_mat;
}

std::vector<Eigen::MatrixXd>
NormalizedDotProduct_ICM ::envs_envs_grad(const ClusterDescriptor &envs1,
                                          const ClusterDescriptor &envs2,
                                          const Eigen::VectorXd &hyps) {

  std::vector<Eigen::MatrixXd> grad_mats;
  Eigen::MatrixXd kern = envs_envs(envs1, envs2, hyps);
  Eigen::MatrixXd grad = 2 * kern / hyps(0);
  grad_mats.push_back(kern);
  grad_mats.push_back(grad);

  int n_types = envs1.n_types;
  for (int s1 = 0; s1 < n_types; s1++) {
    int n_sparse_1 = envs1.n_clusters_by_type[s1];
    int c_sparse_1 = envs1.cumulative_type_count[s1];
    for (int s2 = 0; s2 < n_types; s2++) {
      int n_sparse_2 = envs1.n_clusters_by_type[s2];
      int c_sparse_2 = envs1.cumulative_type_count[s2];

      int icm_index = get_icm_index(s1, s2, n_types);
      double icm_val = hyps(1 + icm_index);

      Eigen::MatrixXd icm_grad =
          Eigen::MatrixXd::Zero(kern.rows(), kern.cols());
      icm_grad.block(c_sparse_1, c_sparse_2, n_sparse_1, n_sparse_2) =
          kern.block(c_sparse_1, c_sparse_2, n_sparse_1, n_sparse_2) / icm_val;
      grad_mats.push_back(icm_grad);
    }
  }

  return grad_mats;
}

std::vector<Eigen::MatrixXd>
NormalizedDotProduct_ICM ::envs_struc_grad(const ClusterDescriptor &envs,
                                           const DescriptorValues &struc,
                                           const Eigen::VectorXd &hyps) {

  // Set square of the signal variance.
  double sig_new = hyps(0);
  double sig_sq = sig_new * sig_new;

  // Check types.
  int n_types_1 = envs.n_types;
  int n_types_2 = struc.n_types;
  assert(n_types_1 == n_types_2);

  // Check descriptor size.
  int n_descriptors_1 = envs.n_descriptors;
  int n_descriptors_2 = struc.n_descriptors;
  assert(n_descriptors_1 == n_descriptors_2);

  Eigen::MatrixXd init_mat =
      Eigen::MatrixXd::Zero(envs.n_clusters, 1 + struc.n_atoms * 3 + 6);
  Eigen::MatrixXd kern_mat = init_mat;
  Eigen::MatrixXd sig_mat = init_mat;
  std::vector<Eigen::MatrixXd> icm_mats;
  for (int i = 0; i < n_icm_coeffs; i++) {
    icm_mats.push_back(init_mat);
  }

  int n_types = envs.n_types;
  double vol_inv = 1 / struc.volume;
  double empty_thresh = 1e-8;

  for (int s1 = 0; s1 < n_types; s1++) {
    for (int s2 = 0; s2 < n_types; s2++) {
      int icm_index = get_icm_index(s1, s2, n_types);
      double icm_val = hyps(1 + icm_index);

      // Compute dot products. (Should be done in parallel with MKL.)
      Eigen::MatrixXd dot_vals =
          envs.descriptors[s1] * struc.descriptors[s2].transpose();
      Eigen::MatrixXd force_dot =
          envs.descriptors[s1] * struc.descriptor_force_dervs[s2].transpose();

      Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s2];

      // Compute kernels. Can parallelize over environments.
      int n_sparse = envs.n_clusters_by_type[s1];
      int n_struc = struc.n_clusters_by_type[s2];
      int c_sparse = envs.cumulative_type_count[s1];

#pragma omp parallel for
      for (int i = 0; i < n_sparse; i++) {
        double norm_i = envs.descriptor_norms[s1](i);

        // Continue if sparse environment i has no neighbors.
        if (norm_i < empty_thresh)
          continue;
        int sparse_index = c_sparse + i;

        for (int j = 0; j < n_struc; j++) {
          double norm_j = struc.descriptor_norms[s2](j);
          double norm_ij = norm_i * norm_j;
          double norm_ij3 = norm_ij * norm_j * norm_j;

          // Continue if atom j has no neighbors.
          if (norm_j < empty_thresh)
            continue;

          // Energy kernel.
          double norm_dot = dot_vals(i, j) / norm_ij;
          double dval = power * pow(norm_dot, power - 1);
          kern_mat(sparse_index, 0) += sig_sq * icm_val * pow(norm_dot, power);
          sig_mat(sparse_index, 0) +=
              2 * sig_new * icm_val * pow(norm_dot, power);
          icm_mats[icm_index](sparse_index, 0) += sig_sq * pow(norm_dot, power);

          // Force kernel.
          int n_neigh = struc.neighbor_counts[s2](j);
          int c_neigh = struc.cumulative_neighbor_counts[s2](j);
          int atom_index = struc.atom_indices[s2](j);

          for (int k = 0; k < n_neigh; k++) {
            int neighbor_index = struc.neighbor_indices[s2](c_neigh + k);
            int stress_counter = 0;

            for (int comp = 0; comp < 3; comp++) {
              int ind = c_neigh + k;
              int force_index = 3 * ind + comp;
              double f1 = force_dot(i, force_index) / norm_ij;
              double f2 =
                  dot_vals(i, j) * struc_force_dot(force_index) / norm_ij3;
              double f3 = f1 - f2;
              double force_kern_val = sig_sq * icm_val * dval * f3;
              double sig_force_derv = 2 * sig_new * icm_val * dval * f3;
              double icm_force_derv = sig_sq * dval * f3;

              kern_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                  force_kern_val;
              kern_mat(sparse_index, 1 + 3 * atom_index + comp) +=
                  force_kern_val;

              sig_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                  sig_force_derv;
              sig_mat(sparse_index, 1 + 3 * atom_index + comp) +=
                  sig_force_derv;

              icm_mats[icm_index](sparse_index, 1 + 3 * neighbor_index +
                                                    comp) -= icm_force_derv;
              icm_mats[icm_index](sparse_index, 1 + 3 * neighbor_index +
                                                    comp) -= icm_force_derv;

              for (int comp2 = comp; comp2 < 3; comp2++) {
                double coord = struc.neighbor_coordinates[s2](ind, comp2);
                kern_mat(sparse_index,
                         1 + 3 * struc.n_atoms + stress_counter) -=
                    force_kern_val * coord * vol_inv;

                sig_mat(sparse_index, 1 + 3 * struc.n_atoms + stress_counter) -=
                    sig_force_derv * coord * vol_inv;

                icm_mats[icm_index](sparse_index,
                                    1 + 3 * neighbor_index + comp) -=
                    icm_force_derv * coord * vol_inv;
                stress_counter++;
              }
            }
          }
        }
      }
    }
  }
  std::vector<Eigen::MatrixXd> kernel_gradients;
  kernel_gradients.push_back(kern_mat);
  kernel_gradients.push_back(sig_mat);
  for (int i = 0; i < n_icm_coeffs; i++) {
    kernel_gradients.push_back(icm_mats[i]);
  }
  return kernel_gradients;
}

Eigen::MatrixXd
NormalizedDotProduct_ICM ::envs_struc(const ClusterDescriptor &envs,
                                      const DescriptorValues &struc,
                                      const Eigen::VectorXd &hyps) {

  // Set square of the signal variance.
  double sig_sq = hyps(0) * hyps(0);

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
  double empty_thresh = 1e-8;

  for (int s1 = 0; s1 < n_types; s1++) {
    for (int s2 = 0; s2 < n_types; s2++) {
      int icm_index = get_icm_index(s1, s2, n_types);
      double icm_val = hyps(1 + icm_index);

      // Compute dot products. (Should be done in parallel with MKL.)
      Eigen::MatrixXd dot_vals =
          envs.descriptors[s1] * struc.descriptors[s2].transpose();
      Eigen::MatrixXd force_dot =
          envs.descriptors[s1] * struc.descriptor_force_dervs[s2].transpose();

      Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s2];

      // Compute kernels. Can parallelize over environments.
      int n_sparse = envs.n_clusters_by_type[s1];
      int n_struc = struc.n_clusters_by_type[s2];
      int c_sparse = envs.cumulative_type_count[s1];

#pragma omp parallel for
      for (int i = 0; i < n_sparse; i++) {
        double norm_i = envs.descriptor_norms[s1](i);

        // Continue if sparse environment i has no neighbors.
        if (norm_i < empty_thresh)
          continue;
        int sparse_index = c_sparse + i;

        for (int j = 0; j < n_struc; j++) {
          double norm_j = struc.descriptor_norms[s2](j);
          double norm_ij = norm_i * norm_j;
          double norm_ij3 = norm_ij * norm_j * norm_j;

          // Continue if atom j has no neighbors.
          if (norm_j < empty_thresh)
            continue;

          // Energy kernel.
          double norm_dot = dot_vals(i, j) / norm_ij;
          double dval = power * pow(norm_dot, power - 1);
          kern_mat(sparse_index, 0) += sig_sq * icm_val * pow(norm_dot, power);

          // Force kernel.
          int n_neigh = struc.neighbor_counts[s2](j);
          int c_neigh = struc.cumulative_neighbor_counts[s2](j);
          int atom_index = struc.atom_indices[s2](j);

          for (int k = 0; k < n_neigh; k++) {
            int neighbor_index = struc.neighbor_indices[s2](c_neigh + k);
            int stress_counter = 0;

            for (int comp = 0; comp < 3; comp++) {
              int ind = c_neigh + k;
              int force_index = 3 * ind + comp;
              double f1 = force_dot(i, force_index) / norm_ij;
              double f2 =
                  dot_vals(i, j) * struc_force_dot(force_index) / norm_ij3;
              double f3 = f1 - f2;
              double force_kern_val = sig_sq * icm_val * dval * f3;

              kern_mat(sparse_index, 1 + 3 * neighbor_index + comp) -=
                  force_kern_val;
              kern_mat(sparse_index, 1 + 3 * atom_index + comp) +=
                  force_kern_val;

              for (int comp2 = comp; comp2 < 3; comp2++) {
                double coord = struc.neighbor_coordinates[s2](ind, comp2);
                kern_mat(sparse_index,
                         1 + 3 * struc.n_atoms + stress_counter) -=
                    force_kern_val * coord * vol_inv;
                stress_counter++;
              }
            }
          }
        }
      }
    }
  }

  return kern_mat;
}

Eigen::MatrixXd
NormalizedDotProduct_ICM ::struc_struc(const DescriptorValues &struc1,
                                       const DescriptorValues &struc2,
                                       const Eigen::VectorXd &hyps) {

  // Set square of the signal variance.
  double sig_sq = hyps(0) * hyps(0);

  int n_elements_1 = 1 + 3 * struc1.n_atoms + 6;
  int n_elements_2 = 1 + 3 * struc2.n_atoms + 6;
  Eigen::MatrixXd kernel_matrix =
      Eigen::MatrixXd::Zero(n_elements_1, n_elements_2);

  // Check types.
  int n_types_1 = struc1.n_types;
  int n_types_2 = struc2.n_types;
  bool type_check = (n_types_1 == n_types_2);
  assert(n_types_1 == n_types_2);

  // Check descriptor size.
  int n_descriptors_1 = struc1.n_descriptors;
  int n_descriptors_2 = struc2.n_descriptors;
  assert(n_descriptors_1 == n_descriptors_2);

  double vol_inv_1 = 1 / struc1.volume;
  double vol_inv_2 = 1 / struc2.volume;

  double empty_thresh = 1e-8;
  std::vector<int> stress_inds{0, 3, 5};

  for (int s1 = 0; s1 < n_types_1; s1++) {
    for (int s2 = 0; s2 < n_types_1; s2++) {
      int icm_index = get_icm_index(s1, s2, n_types_1);
      double icm_val = hyps(1 + icm_index);

      // Compute dot products.
      Eigen::MatrixXd dot_vals =
          struc1.descriptors[s1] * struc2.descriptors[s2].transpose();
      Eigen::MatrixXd force_dot_1 = struc1.descriptor_force_dervs[s1] *
                                    struc2.descriptors[s2].transpose();
      Eigen::MatrixXd force_dot_2 = struc2.descriptor_force_dervs[s2] *
                                    struc1.descriptors[s1].transpose();
      Eigen::MatrixXd force_force =
          struc1.descriptor_force_dervs[s1] *
          struc2.descriptor_force_dervs[s2].transpose();

      Eigen::VectorXd struc_force_dot_1 = struc1.descriptor_force_dots[s1];
      Eigen::VectorXd struc_force_dot_2 = struc2.descriptor_force_dots[s2];

      // Compute kernels.
      int n_struc1 = struc1.n_clusters_by_type[s1];
      int n_struc2 = struc2.n_clusters_by_type[s2];

      // TODO: Parallelize.
      for (int i = 0; i < n_struc1; i++) {
        double norm_i = struc1.descriptor_norms[s1](i);

        // Continue if atom i has no neighbors.
        if (norm_i < empty_thresh)
          continue;

        double norm_i2 = norm_i * norm_i;
        double norm_i3 = norm_i2 * norm_i;

        for (int j = 0; j < n_struc2; j++) {
          double norm_j = struc2.descriptor_norms[s2](j);

          // Continue if atom j has no neighbors.
          if (norm_j < empty_thresh)
            continue;

          double norm_j2 = norm_j * norm_j;
          double norm_j3 = norm_j2 * norm_j;
          double norm_ij = norm_i * norm_j;

          // Energy/energy kernel.
          double norm_dot = dot_vals(i, j) / norm_ij;
          double c1 = (power - 1) * power * pow(norm_dot, power - 2);
          double c2 = power * pow(norm_dot, power - 1);
          kernel_matrix(0, 0) += sig_sq * icm_val * pow(norm_dot, power);

          int n_neigh_1 = struc1.neighbor_counts[s1](i);
          int c_neigh_1 = struc1.cumulative_neighbor_counts[s1](i);
          int c_ind_1 = struc1.atom_indices[s1](i);

          int n_neigh_2 = struc2.neighbor_counts[s2](j);
          int c_neigh_2 = struc2.cumulative_neighbor_counts[s2](j);
          int c_ind_2 = struc2.atom_indices[s2](j);

          // Energy/force and energy/stress kernels.
          for (int k = 0; k < n_neigh_2; k++) {
            int ind = c_neigh_2 + k;
            int neighbor_index = struc2.neighbor_indices[s2](ind);
            int stress_counter = 0;

            for (int comp = 0; comp < 3; comp++) {
              int force_index = 3 * ind + comp;
              double f1 = force_dot_2(force_index, i) / norm_ij;
              double f2 = dot_vals(i, j) * struc_force_dot_2(force_index) /
                          (norm_i * norm_j3);
              double f3 = f1 - f2;
              double force_kern_val = sig_sq * icm_val * c2 * f3;

              // Energy/force.
              kernel_matrix(0, 1 + 3 * neighbor_index + comp) -= force_kern_val;
              kernel_matrix(0, 1 + 3 * c_ind_2 + comp) += force_kern_val;

              // Energy/stress.
              for (int comp2 = comp; comp2 < 3; comp2++) {
                double coord = struc2.neighbor_coordinates[s2](ind, comp2);
                kernel_matrix(0, 1 + 3 * struc2.n_atoms + stress_counter) -=
                    force_kern_val * coord * vol_inv_2;
                stress_counter++;
              }
            }
          }

          // Force/energy and stress/energy kernels.
          for (int k = 0; k < n_neigh_1; k++) {
            int ind = c_neigh_1 + k;
            int neighbor_index = struc1.neighbor_indices[s1](ind);
            int stress_counter = 0;

            for (int comp = 0; comp < 3; comp++) {
              int force_index = 3 * ind + comp;
              double f1 = force_dot_1(force_index, j) / norm_ij;
              double f2 = dot_vals(i, j) * struc_force_dot_1(force_index) /
                          (norm_j * norm_i3);
              double f3 = f1 - f2;
              double force_kern_val = sig_sq * icm_val * c2 * f3;

              // Force/energy.
              kernel_matrix(1 + 3 * neighbor_index + comp, 0) -= force_kern_val;
              kernel_matrix(1 + 3 * c_ind_1 + comp, 0) += force_kern_val;

              // Stress/energy.
              for (int comp2 = comp; comp2 < 3; comp2++) {
                double coord = struc1.neighbor_coordinates[s1](ind, comp2);
                kernel_matrix(1 + 3 * struc1.n_atoms + stress_counter, 0) -=
                    force_kern_val * coord * vol_inv_1;
                stress_counter++;
              }
            }
          }

          // Force/force, force/stress, stress/force, and stress/stress kernels.
          for (int k = 0; k < n_neigh_1; k++) {
            int ind1 = c_neigh_1 + k;
            int n_ind_1 = struc1.neighbor_indices[s1](ind1);

            for (int l = 0; l < n_neigh_2; l++) {
              int ind2 = c_neigh_2 + l;
              int n_ind_2 = struc2.neighbor_indices[s2](ind2);

              for (int m = 0; m < 3; m++) {
                int f_ind_1 = 3 * ind1 + m;
                for (int n = 0; n < 3; n++) {
                  int f_ind_2 = 3 * ind2 + n;
                  double v1 = force_dot_1(f_ind_1, j) / norm_ij -
                              norm_dot * struc_force_dot_1(f_ind_1) / norm_i2;
                  double v2 = force_dot_2(f_ind_2, i) / norm_ij -
                              norm_dot * struc_force_dot_2(f_ind_2) / norm_j2;
                  double v3 = force_force(f_ind_1, f_ind_2) / norm_ij;
                  double v4 = struc_force_dot_1(f_ind_1) *
                              force_dot_2(f_ind_2, i) / (norm_i3 * norm_j);
                  double v5 = struc_force_dot_2(f_ind_2) *
                              force_dot_1(f_ind_1, j) / (norm_i * norm_j3);
                  double v6 = struc_force_dot_1(f_ind_1) *
                              struc_force_dot_2(f_ind_2) * norm_dot /
                              (norm_i2 * norm_j2);

                  double kern_val = sig_sq * icm_val *
                                    (c1 * v1 * v2 + c2 * (v3 - v4 - v5 + v6));

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
                    double coord = struc1.neighbor_coordinates[s1](ind1, p);
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
                    double coord = struc2.neighbor_coordinates[s2](ind2, p);
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
                    double coord1 = struc1.neighbor_coordinates[s1](ind1, p);
                    stress_ind_2 = stress_inds[n];
                    for (int q = n; q < 3; q++) {
                      double coord2 = struc2.neighbor_coordinates[s2](ind2, q);
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
  }

  return kernel_matrix;
}

Eigen::VectorXd
NormalizedDotProduct_ICM ::self_kernel_struc(const DescriptorValues &struc,
                                             const Eigen::VectorXd &hyps) {

  //   double sig_sq = hyps(0) * hyps(0);

  //   int n_elements = 1 + 3 * struc.n_atoms + 6;
  //   Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(n_elements);

  //   int n_types = struc.n_types;
  //   double vol_inv = 1 / struc.volume;
  //   double vol_inv_sq = vol_inv * vol_inv;
  //   double empty_thresh = 1e-8;

  //   for (int s = 0; s < n_types; s++) {
  //     // Compute dot products. (Should be done in parallel with MKL.)
  //     Eigen::MatrixXd dot_vals =
  //         struc.descriptors[s] * struc.descriptors[s].transpose();
  //     Eigen::MatrixXd force_dot =
  //         struc.descriptor_force_dervs[s] * struc.descriptors[s].transpose();
  //     Eigen::MatrixXd force_force = struc.descriptor_force_dervs[s] *
  //                                   struc.descriptor_force_dervs[s].transpose();

  //     Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s];

  //     // Compute kernels.
  //     int n_struc = struc.n_clusters_by_type[s];

  //     // TODO: Parallelize.
  //     Eigen::MatrixXd par_mat = Eigen::MatrixXd::Zero(n_struc, n_elements);
  // #pragma omp parallel for
  //     for (int i = 0; i < n_struc; i++) {
  //       double norm_i = struc.descriptor_norms[s](i);

  //       // Continue if atom i has no neighbors.
  //       if (norm_i < empty_thresh)
  //         continue;

  //       double norm_i2 = norm_i * norm_i;
  //       double norm_i3 = norm_i2 * norm_i;

  //       for (int j = i; j < n_struc; j++) {
  //         double norm_j = struc.descriptor_norms[s](j);

  //         // Continue if atom j has no neighbors.
  //         if (norm_j < empty_thresh)
  //           continue;

  //         double mult_fac;
  //         if (i == j)
  //           mult_fac = 1;
  //         else
  //           mult_fac = 2;

  //         double norm_j2 = norm_j * norm_j;
  //         double norm_j3 = norm_j2 * norm_j;
  //         double norm_ij = norm_i * norm_j;

  //         // Energy kernel.
  //         double norm_dot = dot_vals(i, j) / norm_ij;
  //         double c1 = (power - 1) * power * pow(norm_dot, power - 2);
  //         double c2 = power * pow(norm_dot, power - 1);
  //         par_mat(i, 0) += sig_sq * mult_fac * pow(norm_dot, power);

  //         // Force kernel.
  //         int n_neigh_1 = struc.neighbor_counts[s](i);
  //         int c_neigh_1 = struc.cumulative_neighbor_counts[s](i);
  //         int c_ind_1 = struc.atom_indices[s](i);

  //         int n_neigh_2 = struc.neighbor_counts[s](j);
  //         int c_neigh_2 = struc.cumulative_neighbor_counts[s](j);
  //         int c_ind_2 = struc.atom_indices[s](j);

  //         for (int k = 0; k < n_neigh_1; k++) {
  //           int ind1 = c_neigh_1 + k;
  //           int n_ind_1 = struc.neighbor_indices[s](ind1);

  //           for (int l = 0; l < n_neigh_2; l++) {
  //             int ind2 = c_neigh_2 + l;
  //             int n_ind_2 = struc.neighbor_indices[s](ind2);

  //             int stress_counter = 0;
  //             for (int m = 0; m < 3; m++) {
  //               int f_ind_1 = 3 * ind1 + m;
  //               int f_ind_2 = 3 * ind2 + m;
  //               double v1 = force_dot(f_ind_1, j) / norm_ij -
  //                           norm_dot * struc_force_dot(f_ind_1) / norm_i2;
  //               double v2 = force_dot(f_ind_2, i) / norm_ij -
  //                           norm_dot * struc_force_dot(f_ind_2) / norm_j2;
  //               double v3 = force_force(f_ind_1, f_ind_2) / norm_ij;
  //               double v4 = struc_force_dot(f_ind_1) * force_dot(f_ind_2, i)
  //               /
  //                           (norm_i3 * norm_j);
  //               double v5 = struc_force_dot(f_ind_2) * force_dot(f_ind_1, j)
  //               /
  //                           (norm_i * norm_j3);
  //               double v6 = struc_force_dot(f_ind_1) *
  //               struc_force_dot(f_ind_2) *
  //                           norm_dot / (norm_i2 * norm_j2);

  //               double kern_val =
  //                   sig_sq * mult_fac * (c1 * v1 * v2 + c2 * (v3 - v4 - v5 +
  //                   v6));

  //               if (c_ind_1 == c_ind_2)
  //                 par_mat(i, 1 + c_ind_1 * 3 + m) += kern_val;
  //               if (c_ind_1 == n_ind_2)
  //                 par_mat(i, 1 + c_ind_1 * 3 + m) -= kern_val;
  //               if (n_ind_1 == c_ind_2)
  //                 par_mat(i, 1 + n_ind_1 * 3 + m) -= kern_val;
  //               if (n_ind_1 == n_ind_2)
  //                 par_mat(i, 1 + n_ind_1 * 3 + m) += kern_val;

  //               // Stress kernel.
  //               for (int n = m; n < 3; n++) {
  //                 double coord1 = struc.neighbor_coordinates[s](ind1, n);
  //                 double coord2 = struc.neighbor_coordinates[s](ind2, n);
  //                 par_mat(i, 1 + 3 * struc.n_atoms + stress_counter) +=
  //                     kern_val * coord1 * coord2 * vol_inv_sq;
  //                 stress_counter++;
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }

  //     // Reduce kernel values.
  //     for (int i = 0; i < n_struc; i++) {
  //       for (int j = 0; j < n_elements; j++) {
  //         kernel_vector(j) += par_mat(i, j);
  //       }
  //     }
  //   }

  //   return kernel_vector;
}

void NormalizedDotProduct_ICM ::set_hyperparameters(Eigen::VectorXd new_hyps) {
  //   sigma = new_hyps(0);
  //   sig2 = sigma * sigma;
  //   kernel_hyperparameters = new_hyps;
}

Eigen::MatrixXd NormalizedDotProduct_ICM ::compute_mapping_coefficients(
    const SparseGP &gp_model, int kernel_index) {

  //   // Assumes there is at least one sparse environment stored in the sparse
  //   GP.

  //   Eigen::MatrixXd mapping_coeffs;
  //   if (power != 2) {
  //     std::cout
  //         << "Mapping coefficients of the normalized dot product kernel are "
  //            "implemented for power 2 only."
  //         << std::endl;
  //     return mapping_coeffs;
  //   }

  //   // Initialize beta vector.
  //   int p_size = gp_model.sparse_descriptors[kernel_index].n_descriptors;
  //   int beta_size = p_size * (p_size + 1) / 2;
  //   int n_species = gp_model.sparse_descriptors[kernel_index].n_types;
  //   int n_sparse = gp_model.sparse_descriptors[kernel_index].n_clusters;
  //   mapping_coeffs = Eigen::MatrixXd::Zero(n_species, beta_size);

  //   // Get alpha index.
  //   int alpha_ind = 0;
  //   for (int i = 0; i < kernel_index; i++) {
  //     alpha_ind += gp_model.sparse_descriptors[i].n_clusters;
  //   }

  //   // Loop over types.
  //   for (int i = 0; i < n_species; i++) {
  //     int n_types =
  //         gp_model.sparse_descriptors[kernel_index].n_clusters_by_type[i];
  //     int c_types =
  //         gp_model.sparse_descriptors[kernel_index].cumulative_type_count[i];

  //     // Loop over clusters within each type.
  //     for (int j = 0; j < n_types; j++) {
  //       Eigen::VectorXd p_current =
  //           gp_model.sparse_descriptors[kernel_index].descriptors[i].row(j);
  //       double p_norm =
  //           gp_model.sparse_descriptors[kernel_index].descriptor_norms[i](j);
  //       double alpha_val = gp_model.alpha(alpha_ind + c_types + j);
  //       int beta_count = 0;

  //       // First loop over descriptor values.
  //       for (int k = 0; k < p_size; k++) {
  //         double p_ik = p_current(k) / p_norm;

  //         // Second loop over descriptor values.
  //         for (int l = k; l < p_size; l++) {
  //           double p_il = p_current(l) / p_norm;
  //           double beta_val = sig2 * p_ik * p_il * alpha_val;

  //           // Update beta vector.
  //           if (k != l) {
  //             mapping_coeffs(i, beta_count) += 2 * beta_val;
  //           } else {
  //             mapping_coeffs(i, beta_count) += beta_val;
  //           }

  //           beta_count++;
  //         }
  //       }
  //     }
  //   }

  //   return mapping_coeffs;
}

int get_icm_index(int s1, int s2, int n_types) {
  // Get ICM coefficient.
  int s_min = std::min(s1, s2);
  int s_diff = n_types - s_min;
  int icm_index = (n_types * (n_types + 1) / 2) - (s_diff * (s_diff + 1) / 2);
  return icm_index;
}
