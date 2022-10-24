#include "dot_product.h"
#include "descriptor.h"
#include "sparse_gp.h"
#include "structure.h"
#undef NDEBUG
#include <assert.h>
#include <cmath>
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>
#include <stdexcept>

DotProduct ::DotProduct(){};

DotProduct ::DotProduct(double sigma, double power) {

  this->sigma = sigma;
  sig2 = sigma * sigma;
  this->power = power;
  kernel_name = "DotProduct";

  // Set kernel hyperparameters.
  Eigen::VectorXd hyps(1);
  hyps << sigma;
  kernel_hyperparameters = hyps;
}

Eigen::MatrixXd DotProduct ::envs_envs(const ClusterDescriptor &envs1,
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

  for (int s = 0; s < n_types; s++) {
      // Why not do envs1.descriptors[s] / envs1.descriptor_norms[s]
      // and then multiply them to get norm_dot matrix directly??
    // Compute dot products. (Should be done in parallel with MKL.)
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

      // Continue if sparse environment i has no neighbors.
      if (norm_i < empty_thresh)
        continue;
      int ind1 = c_sparse_1 + i;

      for (int j = 0; j < n_sparse_2; j++) {
        double norm_j = envs2.descriptor_norms[s](j);
        //double norm_ij = norm_i * norm_j;

        // Continue if atom j has no neighbors.
        if (norm_j < empty_thresh)
          continue;
        int ind2 = c_sparse_2 + j;

        // Energy kernel.
        double norm_dot = dot_vals(i, j);
        kern_mat(ind1, ind2) += sig_sq * pow(norm_dot, power);
      }
    }
  }
  return kern_mat;
}

std::vector<Eigen::MatrixXd>
DotProduct ::envs_envs_grad(const ClusterDescriptor &envs1,
                                      const ClusterDescriptor &envs2,
                                      const Eigen::VectorXd &hyps) {

  std::vector<Eigen::MatrixXd> grad_mats;
  Eigen::MatrixXd kern = envs_envs(envs1, envs2, hyps);
  Eigen::MatrixXd grad = 2 * kern / hyps(0);
  grad_mats.push_back(kern);
  grad_mats.push_back(grad);
  return grad_mats;
}

std::vector<Eigen::MatrixXd>
DotProduct ::envs_struc_grad(const ClusterDescriptor &envs,
                                       const DescriptorValues &struc,
                                       const Eigen::VectorXd &hyps) {

  std::vector<Eigen::MatrixXd> grad_mats;
  Eigen::MatrixXd kern = envs_struc(envs, struc, hyps);
  Eigen::MatrixXd grad = 2 * kern / hyps(0);
  grad_mats.push_back(kern);
  grad_mats.push_back(grad);
  return grad_mats;
}

Eigen::MatrixXd DotProduct ::envs_struc(const ClusterDescriptor &envs,
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

  for (int s = 0; s < n_types; s++) {
    // Compute dot products. (Should be done in parallel with MKL.)
    Eigen::MatrixXd dot_vals =
        envs.descriptors[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_dot =
        envs.descriptors[s] * struc.descriptor_force_dervs[s].transpose();

    Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s];

    // Compute kernels. Can parallelize over environments.
    int n_sparse = envs.n_clusters_by_type[s];
    int n_struc = struc.n_clusters_by_type[s];
    int c_sparse = envs.cumulative_type_count[s];

#pragma omp parallel for
    for (int i = 0; i < n_sparse; i++) {
      double norm_i = envs.descriptor_norms[s](i);

      // Continue if sparse environment i has no neighbors.
      if (norm_i < empty_thresh)
        continue;
      int sparse_index = c_sparse + i;

      for (int j = 0; j < n_struc; j++) {
        double norm_j = struc.descriptor_norms[s](j);
        //double norm_ij = norm_i * norm_j;
        //double norm_ij3 = norm_ij * norm_j * norm_j;

        // Continue if atom j has no neighbors.
        if (norm_j < empty_thresh)
          continue;

        // Energy kernel.
        double norm_dot = dot_vals(i, j); // / norm_ij;
        double dval = power * pow(norm_dot, power - 1);
        kern_mat(sparse_index, 0) += sig_sq * pow(norm_dot, power);

        // Force kernel.
        int n_neigh = struc.neighbor_counts[s](j);
        int c_neigh = struc.cumulative_neighbor_counts[s](j);
        int atom_index = struc.atom_indices[s](j);

        for (int k = 0; k < n_neigh; k++) {
          int neighbor_index = struc.neighbor_indices[s](c_neigh + k);
          int stress_counter = 0;

          for (int comp = 0; comp < 3; comp++) {
            int ind = c_neigh + k;
            int force_index = 3 * ind + comp;
            double f1 = force_dot(i, force_index); // / norm_ij;
            //double f2 =
            //    dot_vals(i, j) * struc_force_dot(force_index) / norm_ij3;
            double f3 = f1; // - f2;
            double force_kern_val = sig_sq * dval * f3;

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

Eigen::MatrixXd
DotProduct ::struc_struc(const DescriptorValues &struc1,
                                   const DescriptorValues &struc2,
                                   const Eigen::VectorXd &hyps) {
  
  //throw std::logic_error("struc_struc kernel for DotProduct is not implemented");
  
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

    // TODO: Parallelize.
    for (int i = 0; i < n_struc1; i++) {
      double norm_i = struc1.descriptor_norms[s](i);

      // Continue if atom i has no neighbors.
      if (norm_i < empty_thresh)
        continue;

      double norm_i2 = norm_i * norm_i;
      double norm_i3 = norm_i2 * norm_i;

      for (int j = 0; j < n_struc2; j++) {
        double norm_j = struc2.descriptor_norms[s](j);

        // Continue if atom j has no neighbors.
        if (norm_j < empty_thresh)
          continue;

        double norm_j2 = norm_j * norm_j;
        double norm_j3 = norm_j2 * norm_j;
        double norm_ij = norm_i * norm_j;

        // Energy/energy kernel.
        double norm_dot = dot_vals(i, j) / norm_ij;
        double c1 = (power - 1) * power * pow(norm_dot, power - 2);
        if (abs(norm_dot) < empty_thresh && power < 2) {
            throw std::invalid_argument( "Dot product of descriptors is 0, \
                and the negative power function in force-force kernel diverges." );
        }
        double c2 = power * pow(norm_dot, power - 1);
        kernel_matrix(0, 0) += sig_sq * pow(norm_dot, power);

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
            double f1 = force_dot_2(force_index, i) / norm_ij;
            double f2 = dot_vals(i, j) * struc_force_dot_2(force_index) /
                        (norm_i * norm_j3);
            double f3 = f1 - f2;
            double force_kern_val = sig_sq * c2 * f3;

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
            double f1 = force_dot_1(force_index, j) / norm_ij;
            double f2 = dot_vals(i, j) * struc_force_dot_1(force_index) /
                        (norm_j * norm_i3);
            double f3 = f1 - f2;
            double force_kern_val = sig_sq * c2 * f3;

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

                double kern_val =
                    sig_sq * (c1 * v1 * v2 + c2 * (v3 - v4 - v5 + v6));

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
DotProduct ::self_kernel_struc(const DescriptorValues &struc,
                                         const Eigen::VectorXd &hyps) {

  double sig_sq = hyps(0) * hyps(0);

  int n_elements = 1 + 3 * struc.n_atoms + 6;
  Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(n_elements);

  int n_types = struc.n_types;
  double vol_inv = 1 / struc.volume;
  double vol_inv_sq = vol_inv * vol_inv;
  double empty_thresh = 1e-8;

  for (int s = 0; s < n_types; s++) {
    // Compute dot products. (Should be done in parallel with MKL.)
    Eigen::MatrixXd dot_vals =
        struc.descriptors[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_dot =
        struc.descriptor_force_dervs[s] * struc.descriptors[s].transpose();
    Eigen::MatrixXd force_force = struc.descriptor_force_dervs[s] *
                                  struc.descriptor_force_dervs[s].transpose();

    Eigen::VectorXd struc_force_dot = struc.descriptor_force_dots[s];

    // Compute kernels.
    int n_struc = struc.n_clusters_by_type[s];

    // TODO: Parallelize.
    Eigen::MatrixXd par_mat = Eigen::MatrixXd::Zero(n_struc, n_elements);
#pragma omp parallel for
    for (int i = 0; i < n_struc; i++) {
      double norm_i = struc.descriptor_norms[s](i);

      // Continue if atom i has no neighbors.
      if (norm_i < empty_thresh)
        continue;

      double norm_i2 = norm_i * norm_i;
      double norm_i3 = norm_i2 * norm_i;

      for (int j = i; j < n_struc; j++) {
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
        if (abs(norm_dot) < empty_thresh && power < 2) {
            throw std::invalid_argument( "Dot product of descriptors is 0, \
                and the negative power function in force-force kernel diverges." );
        }
        double c2 = power * pow(norm_dot, power - 1);
        par_mat(i, 0) += sig_sq * mult_fac * pow(norm_dot, power);

        // Force kernel.
        int n_neigh_1 = struc.neighbor_counts[s](i);
        int c_neigh_1 = struc.cumulative_neighbor_counts[s](i);
        int c_ind_1 = struc.atom_indices[s](i);

        int n_neigh_2 = struc.neighbor_counts[s](j);
        int c_neigh_2 = struc.cumulative_neighbor_counts[s](j);
        int c_ind_2 = struc.atom_indices[s](j);

        for (int k = 0; k < n_neigh_1; k++) {
          int ind1 = c_neigh_1 + k;
          int n_ind_1 = struc.neighbor_indices[s](ind1);

          for (int l = 0; l < n_neigh_2; l++) {
            int ind2 = c_neigh_2 + l;
            int n_ind_2 = struc.neighbor_indices[s](ind2);

            int stress_counter = 0;
            for (int m = 0; m < 3; m++) {
              int f_ind_1 = 3 * ind1 + m;
              int f_ind_2 = 3 * ind2 + m;
              double v1 = force_dot(f_ind_1, j) / norm_ij -
                          norm_dot * struc_force_dot(f_ind_1) / norm_i2;
              double v2 = force_dot(f_ind_2, i) / norm_ij -
                          norm_dot * struc_force_dot(f_ind_2) / norm_j2;
              double v3 = force_force(f_ind_1, f_ind_2) / norm_ij;
              double v4 = struc_force_dot(f_ind_1) * force_dot(f_ind_2, i) /
                          (norm_i3 * norm_j);
              double v5 = struc_force_dot(f_ind_2) * force_dot(f_ind_1, j) /
                          (norm_i * norm_j3);
              double v6 = struc_force_dot(f_ind_1) * struc_force_dot(f_ind_2) *
                          norm_dot / (norm_i2 * norm_j2);

              double kern_val =
                  sig_sq * mult_fac * (c1 * v1 * v2 + c2 * (v3 - v4 - v5 + v6));

              if (c_ind_1 == c_ind_2)
                par_mat(i, 1 + c_ind_1 * 3 + m) += kern_val;
              if (c_ind_1 == n_ind_2)
                par_mat(i, 1 + c_ind_1 * 3 + m) -= kern_val;
              if (n_ind_1 == c_ind_2)
                par_mat(i, 1 + n_ind_1 * 3 + m) -= kern_val;
              if (n_ind_1 == n_ind_2)
                par_mat(i, 1 + n_ind_1 * 3 + m) += kern_val;

              // Stress kernel.
              for (int n = m; n < 3; n++) {
                double coord1 = struc.neighbor_coordinates[s](ind1, n);
                double coord2 = struc.neighbor_coordinates[s](ind2, n);
                par_mat(i, 1 + 3 * struc.n_atoms + stress_counter) +=
                    kern_val * coord1 * coord2 * vol_inv_sq;
                stress_counter++;
              }
            }
          }
        }
      }
    }

    // Reduce kernel values.
    for (int i = 0; i < n_struc; i++) {
      for (int j = 0; j < n_elements; j++) {
        kernel_vector(j) += par_mat(i, j);
      }
    }
  }

  return kernel_vector;
}

std::vector<Eigen::MatrixXd>
DotProduct ::Kuu_grad(const ClusterDescriptor &envs,
                                const Eigen::MatrixXd &Kuu,
                                const Eigen::VectorXd &new_hyps) {

  std::vector<Eigen::MatrixXd> kernel_gradients;

  // Compute Kuu.
  Eigen::MatrixXd Kuu_new = Kuu * (new_hyps(0) * new_hyps(0) / sig2);
  kernel_gradients.push_back(Kuu_new);

  // Compute sigma gradient.
  Eigen::MatrixXd sigma_gradient = Kuu * (2 * new_hyps(0) / sig2);
  kernel_gradients.push_back(sigma_gradient);

  return kernel_gradients;
}

std::vector<Eigen::MatrixXd>
DotProduct ::Kuf_grad(const ClusterDescriptor &envs,
                                const std::vector<Structure> &strucs,
                                int kernel_index, const Eigen::MatrixXd &Kuf,
                                const Eigen::VectorXd &new_hyps) {

  std::vector<Eigen::MatrixXd> kernel_gradients;

  // Compute Kuf.
  Eigen::MatrixXd Kuf_new = Kuf * (new_hyps(0) * new_hyps(0) / sig2);
  kernel_gradients.push_back(Kuf_new);

  // Compute sigma gradient.
  Eigen::MatrixXd sigma_gradient = Kuf * (2 * new_hyps(0) / sig2);
  kernel_gradients.push_back(sigma_gradient);

  return kernel_gradients;
}

void DotProduct ::set_hyperparameters(Eigen::VectorXd new_hyps) {
  sigma = new_hyps(0);
  sig2 = sigma * sigma;
  kernel_hyperparameters = new_hyps;
}

Eigen::MatrixXd
DotProduct ::compute_mapping_coefficients(const SparseGP &gp_model,
                                                    int kernel_index) {

  // Assumes there is at least one sparse environment stored in the sparse GP.

  Eigen::MatrixXd mapping_coeffs;

  if (power == 1) {
    mapping_coeffs = compute_map_coeff_pow1(gp_model, kernel_index);
  } else if (power == 2) {
    mapping_coeffs = compute_map_coeff_pow2(gp_model, kernel_index);
  } else { 
    std::cout
        << "Mapping coefficients of the dot product kernel are "
           "implemented for power 2 only."
        << std::endl;
  }

  return mapping_coeffs;
}

Eigen::MatrixXd
DotProduct ::compute_map_coeff_pow1(const SparseGP &gp_model,
                                              int kernel_index) {

  // Initialize beta vector.
  int p_size = gp_model.sparse_descriptors[kernel_index].n_descriptors;
  int beta_size = p_size;
  int n_species = gp_model.sparse_descriptors[kernel_index].n_types;
  int n_sparse = gp_model.sparse_descriptors[kernel_index].n_clusters;
  Eigen::MatrixXd mapping_coeffs = Eigen::MatrixXd::Zero(n_species, beta_size);
  double empty_thresh = 1e-8;

  // Get alpha index.
  int alpha_ind = 0;
  for (int i = 0; i < kernel_index; i++) {
    alpha_ind += gp_model.sparse_descriptors[i].n_clusters;
  }

  // Loop over types.
  for (int i = 0; i < n_species; i++) {
    int n_types =
        gp_model.sparse_descriptors[kernel_index].n_clusters_by_type[i];
    int c_types =
        gp_model.sparse_descriptors[kernel_index].cumulative_type_count[i];

    // Loop over clusters within each type.
    for (int j = 0; j < n_types; j++) {
      Eigen::VectorXd p_current =
          gp_model.sparse_descriptors[kernel_index].descriptors[i].row(j);
      double p_norm =
          gp_model.sparse_descriptors[kernel_index].descriptor_norms[i](j);
      
      // Skip empty environments.
      if (p_norm < empty_thresh)
        continue;

      double alpha_val = gp_model.alpha(alpha_ind + c_types + j);
      int beta_count = 0;

      // First loop over descriptor values.
      for (int k = 0; k < p_size; k++) {
        double p_ik = p_current(k); // / p_norm;
        mapping_coeffs(i, beta_count) += sig2 * p_ik * alpha_val;

        beta_count++;
      }
    }
  }

  return mapping_coeffs;
}

Eigen::MatrixXd
DotProduct ::compute_map_coeff_pow2(const SparseGP &gp_model,
                                              int kernel_index) {

  // Initialize beta vector.
  int p_size = gp_model.sparse_descriptors[kernel_index].n_descriptors;
  int beta_size = p_size * (p_size + 1) / 2;
  int n_species = gp_model.sparse_descriptors[kernel_index].n_types;
  int n_sparse = gp_model.sparse_descriptors[kernel_index].n_clusters;
  Eigen::MatrixXd mapping_coeffs = Eigen::MatrixXd::Zero(n_species, beta_size);
  double empty_thresh = 1e-8;

  // Get alpha index.
  int alpha_ind = 0;
  for (int i = 0; i < kernel_index; i++) {
    alpha_ind += gp_model.sparse_descriptors[i].n_clusters;
  }

  // Loop over types.
  for (int i = 0; i < n_species; i++) {
    int n_types =
        gp_model.sparse_descriptors[kernel_index].n_clusters_by_type[i];
    int c_types =
        gp_model.sparse_descriptors[kernel_index].cumulative_type_count[i];

    // Loop over clusters within each type.
    for (int j = 0; j < n_types; j++) {
      Eigen::VectorXd p_current =
          gp_model.sparse_descriptors[kernel_index].descriptors[i].row(j);
      double p_norm =
          gp_model.sparse_descriptors[kernel_index].descriptor_norms[i](j);
      
      // Skip empty environments.
      if (p_norm < empty_thresh)
        continue;

      double alpha_val = gp_model.alpha(alpha_ind + c_types + j);
      int beta_count = 0;

      // First loop over descriptor values.
      for (int k = 0; k < p_size; k++) {
        double p_ik = p_current(k); // / p_norm;

        // Second loop over descriptor values.
        for (int l = k; l < p_size; l++) {
          double p_il = p_current(l); // / p_norm;
          double beta_val = sig2 * p_ik * p_il * alpha_val;

          // Update beta vector.
          if (k != l) {
            mapping_coeffs(i, beta_count) += 2 * beta_val;
          } else {
            mapping_coeffs(i, beta_count) += beta_val;
          }

          beta_count++;
        }
      }
    }
  }

  return mapping_coeffs;
}

Eigen::MatrixXd DotProduct ::compute_varmap_coefficients(
    const SparseGP &gp_model, int kernel_index){

  // Assumes there is at least one sparse environment stored in the sparse GP.

  Eigen::MatrixXd mapping_coeffs;
  if (power != 1){
      std::cout
          << "Mapping coefficients of the normalized dot product kernel are "
             "implemented for power 1 only."
          << std::endl;
      return mapping_coeffs;
  }

  double empty_thresh = 1e-8;

  // Initialize beta vector.
  int p_size = gp_model.sparse_descriptors[kernel_index].n_descriptors;
  int beta_size = p_size * (p_size + 1) / 2;
  int n_species = gp_model.sparse_descriptors[kernel_index].n_types;
  int n_sparse = gp_model.sparse_descriptors[kernel_index].n_clusters;
  //mapping_coeffs = Eigen::MatrixXd::Zero(n_species * n_species, p_size * p_size); // can be reduced by symmetry
  mapping_coeffs = Eigen::MatrixXd::Zero(n_species, p_size * p_size); // can be reduced by symmetry

  // Get alpha index.
  
  int alpha_ind = 0;
  for (int i = 0; i < kernel_index; i++){
      alpha_ind += gp_model.sparse_descriptors[i].n_clusters;
  }

  // Loop over types.
  for (int s = 0; s < n_species; s++){
    int n_types = gp_model.sparse_descriptors[kernel_index].n_clusters_by_type[s];
    int c_types =
      gp_model.sparse_descriptors[kernel_index].cumulative_type_count[s];
    int K_ind = alpha_ind + c_types;

    // Loop over clusters within each type.
    for (int i = 0; i < n_types; i++){
      Eigen::VectorXd pi_current =
        gp_model.sparse_descriptors[kernel_index].descriptors[s].row(i);
      double pi_norm =
        gp_model.sparse_descriptors[kernel_index].descriptor_norms[s](i);

      // Skip empty environments.
      if (pi_norm < empty_thresh)
        continue;

      // TODO: include symmetry of i & j
      // Loop over clusters within each type.
      for (int j = 0; j < n_types; j++){
        Eigen::VectorXd pj_current =
          gp_model.sparse_descriptors[kernel_index].descriptors[s].row(j);
        double pj_norm =
          gp_model.sparse_descriptors[kernel_index].descriptor_norms[s](j);

        // Skip empty environments.
        if (pj_norm < empty_thresh)
          continue;

        double Kuu_inv_ij = gp_model.Kuu_inverse(K_ind + i, K_ind + j);
        double Kuu_inv_ij_normed = Kuu_inv_ij; // / pi_norm / pj_norm;
//        double Sigma_ij = gp_model.Sigma(K_ind + i, K_ind + j);
//        double Sigma_ij_normed = Sigma_ij / pi_norm / pj_norm;
        int beta_count = 0;

        // First loop over descriptor values.
        for (int k = 0; k < p_size; k++) {
          double p_ik = pi_current(k);
    
          // Second loop over descriptor values.
          for (int l = 0; l < p_size; l++){
            double p_jl = pj_current(l);
    
            // Update beta vector.
            double beta_val = sig2 * sig2 * p_ik * p_jl * (- Kuu_inv_ij_normed); // + Sigma_ij_normed); // To match the compute_cluster_uncertainty function
            mapping_coeffs(s, beta_count) += beta_val;

            if (k == l && i == 0 && j == 0) {
              mapping_coeffs(s, beta_count) += sig2; // the self kernel term
            }
    
            beta_count++;
          }
        }
      }
    }
  }

  return mapping_coeffs;
}

void DotProduct ::write_info(std::ofstream &coeff_file) {
  coeff_file << std::fixed << std::setprecision(0);
  coeff_file << power << " DotProduct\n";
}

nlohmann::json DotProduct ::return_json(){
  nlohmann::json j;
  to_json(j, *this);
  return j;
}
