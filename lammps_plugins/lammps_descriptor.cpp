#include "lammps_descriptor.h"
#include "radial.h"
#include "y_grad.h"
#include <cmath>
#include <iostream>

void single_bond_multiple_cutoffs(
    double **x, int *type, int jnum, int n_inner, int i, double xtmp,
    double ytmp, double ztmp, int *jlist,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &single_bond_env_dervs,
    const Eigen::MatrixXd &cutoff_matrix) {

  // Initialize basis vectors and spherical harmonics.
  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gx = std::vector<double>(N, 0);
  std::vector<double> gy = std::vector<double>(N, 0);
  std::vector<double> gz = std::vector<double>(N, 0);

  int n_harmonics = (lmax + 1) * (lmax + 1);
  std::vector<double> h = std::vector<double>(n_harmonics, 0);
  std::vector<double> hx = std::vector<double>(n_harmonics, 0);
  std::vector<double> hy = std::vector<double>(n_harmonics, 0);
  std::vector<double> hz = std::vector<double>(n_harmonics, 0);

  // Prepare LAMMPS variables.
  int central_species = type[i] - 1;
  double delx, dely, delz, rsq, r, bond, bond_x, bond_y, bond_z, g_val, gx_val,
      gy_val, gz_val, h_val;
  int j, s, descriptor_counter;

  // Initialize vectors.
  int n_radial = n_species * N;
  int n_bond = n_radial * n_harmonics;
  single_bond_vals = Eigen::VectorXd::Zero(n_bond);
  single_bond_env_dervs = Eigen::MatrixXd::Zero(n_inner * 3, n_bond);

  // Initialize radial hyperparameters.
  std::vector<double> new_radial_hyps = radial_hyps;

  // Loop over neighbors.
  int n_count = 0;
  for (int jj = 0; jj < jnum; jj++) {
    j = jlist[jj];

    delx = x[j][0] - xtmp;
    dely = x[j][1] - ytmp;
    delz = x[j][2] - ztmp;
    rsq = delx * delx + dely * dely + delz * delz;
    r = sqrt(rsq);

    // Retrieve the cutoff.
    int s = type[j] - 1;
    double cutoff = cutoff_matrix(central_species, s);
    double cutforcesq = cutoff * cutoff;

    if (rsq < cutforcesq) { // minus a small value to prevent numerial error
      // Reset endpoint of the radial basis set.
      new_radial_hyps[1] = cutoff;

      calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, delx,
                       dely, delz, r, cutoff, N, new_radial_hyps, cutoff_hyps);
      get_Y(h, hx, hy, hz, delx, dely, delz, lmax);

      // Store the products and their derivatives.
      descriptor_counter = s * N * n_harmonics;

      for (int radial_counter = 0; radial_counter < N; radial_counter++) {
        // Retrieve radial values.
        g_val = g[radial_counter];
        gx_val = gx[radial_counter];
        gy_val = gy[radial_counter];
        gz_val = gz[radial_counter];

        for (int angular_counter = 0; angular_counter < n_harmonics;
             angular_counter++) {

          h_val = h[angular_counter];
          bond = g_val * h_val;

          // Calculate derivatives with the product rule.
          bond_x = gx_val * h_val + g_val * hx[angular_counter];
          bond_y = gy_val * h_val + g_val * hy[angular_counter];
          bond_z = gz_val * h_val + g_val * hz[angular_counter];

          // Update single bond basis arrays.
          single_bond_vals(descriptor_counter) += bond;

          single_bond_env_dervs(n_count * 3, descriptor_counter) += bond_x;
          single_bond_env_dervs(n_count * 3 + 1, descriptor_counter) += bond_y;
          single_bond_env_dervs(n_count * 3 + 2, descriptor_counter) += bond_z;

          descriptor_counter++;
        }
      }
      n_count++;
    }
  }
}

void B1_descriptor(Eigen::VectorXd &B1_vals, Eigen::MatrixXd &B1_env_dervs,
                   double &norm_squared, Eigen::VectorXd &B1_env_dot,
                   const Eigen::VectorXd &single_bond_vals,
                   const Eigen::MatrixXd &single_bond_env_dervs, int n_species,
                   int N, int lmax) {

  assert(lmax == 0);
  int env_derv_size = single_bond_env_dervs.rows();
  int neigh_size = env_derv_size / 3;
  int n_radial = n_species * N;
  int n_descriptors = n_radial; //(n_radial * (n_radial + 1) / 2) * (lmax + 1);

  // Zero the B1 vectors and matrices.
  B1_vals = Eigen::VectorXd::Zero(n_descriptors);
  B1_env_dervs = Eigen::MatrixXd::Zero(env_derv_size, n_descriptors);
  B1_env_dot = Eigen::VectorXd::Zero(env_derv_size);

  // Compute the descriptor.
  for (int n1 = 0; n1 < n_radial; n1++) {
    // Store B1 value.
    B1_vals(n1) += single_bond_vals(n1);

    // Store environment force derivatives.
    for (int atom_index = 0; atom_index < neigh_size; atom_index++) {
      for (int comp = 0; comp < 3; comp++) {
        B1_env_dervs(atom_index * 3 + comp, n1) +=
            single_bond_env_dervs(atom_index * 3 + comp, n1);
      }
    }
  }
  // Compute descriptor norm and dot products.
  norm_squared = B1_vals.dot(B1_vals);
  B1_env_dot = B1_env_dervs * B1_vals;
}

void B2_descriptor(Eigen::VectorXd &B2_vals, 
                   double &norm_squared,
                   const Eigen::VectorXd &single_bond_vals,
                   int n_species,
                   int N, int lmax) { 

  int n_radial = n_species * N;
  int n_harmonics = (lmax + 1) * (lmax + 1);
  int n_descriptors = (n_radial * (n_radial + 1) / 2) * (lmax + 1);

  int n1_l, n2_l, counter, n1_count, n2_count;

  // Zero the B2 vectors and matrices.
  B2_vals = Eigen::VectorXd::Zero(n_descriptors);

  // Compute the descriptor.
  for (int n1 = n_radial - 1; n1 >= 0; n1--) {
    n1_count = (n1 * (2 * n_radial - n1 + 1)) / 2;

    for (int n2 = n1; n2 < n_radial; n2++) {
      n2_count = n2 - n1;

      for (int l = 0; l < (lmax + 1); l++) {
        counter = l + (n1_count + n2_count) * (lmax + 1);

        for (int m = 0; m < (2 * l + 1); m++) {
          n1_l = n1 * n_harmonics + (l * l + m);
          n2_l = n2 * n_harmonics + (l * l + m);

          // Store B2 value.
          B2_vals(counter) += single_bond_vals(n1_l) * single_bond_vals(n2_l);
        }
        //printf(" | n1 = %d, n2 = %d, l = %d, B2 = %g |\n", n1, n2, l, B2_vals(counter));
      }
    }
  }

  // Compute w(n1, n2, l), where f_ik = w * dB/dr_ik
  norm_squared = B2_vals.dot(B2_vals);
}

void compute_energy_and_u(Eigen::VectorXd &B2_vals, 
                   double &norm_squared,
                   const Eigen::VectorXd &single_bond_vals,
                   int power, int n_species,
                   int N, int lmax, const Eigen::MatrixXd &beta_matrix, 
                   Eigen::VectorXd &u, double *evdwl) {

  int n1_l, n2_l, counter, n1_count, n2_count;
  int n_radial = n_species * N;
  int n_harmonics = (lmax + 1) * (lmax + 1);

  Eigen::VectorXd w;
  if (power == 1) {
    double B2_norm = pow(norm_squared, 0.5);
    *evdwl = B2_vals.dot(beta_matrix.col(0)) / B2_norm;
    w = beta_matrix.col(0) / B2_norm - *evdwl * B2_vals / norm_squared;
  } else if (power == 2) { 
    Eigen::VectorXd beta_p = beta_matrix * B2_vals;
    *evdwl = B2_vals.dot(beta_p) / norm_squared;
    w = 2 * (beta_p - *evdwl * B2_vals) / norm_squared;
  }

  // Compute u(n1, l, m), where f_ik = u * dA/dr_ik
  u = Eigen::VectorXd::Zero(single_bond_vals.size());
  double factor;
  for (int n1 = n_radial - 1; n1 >= 0; n1--) {
    for (int n2 = 0; n2 < n_radial; n2++) {
      if (n1 == n2){
        n1_count = (n1 * (2 * n_radial - n1 + 1)) / 2;
        n2_count = n2 - n1;
        factor = 1.0;
      } else if (n1 < n2) {
        n1_count = (n1 * (2 * n_radial - n1 + 1)) / 2;
        n2_count = n2 - n1;
        factor = 0.5;
      } else {
        n1_count = (n2 * (2 * n_radial - n2 + 1)) / 2;
        n2_count = n1 - n2;
        factor = 0.5;
      }

      for (int l = 0; l < (lmax + 1); l++) {
        counter = l + (n1_count + n2_count) * (lmax + 1);

        for (int m = 0; m < (2 * l + 1); m++) {
          n1_l = n1 * n_harmonics + (l * l + m);
          n2_l = n2 * n_harmonics + (l * l + m);

          u(n1_l) += w(counter) * single_bond_vals(n2_l) * factor;
        }
      }
    }
  }
  u *= 2;
}

/************************************************************************

                                B3

************************************************************************/
void compute_Bk(Eigen::VectorXd &Bk_vals, Eigen::MatrixXd &Bk_force_dervs,
                double &norm_squared, Eigen::VectorXd &Bk_force_dots,
                const Eigen::VectorXcd &single_bond_vals,
                const Eigen::MatrixXcd &single_bond_force_dervs,
                std::vector<std::vector<int>> nu, int nos, int K, int N,
                int lmax, const Eigen::VectorXd &coeffs,
                const Eigen::MatrixXd &beta_matrix, Eigen::VectorXcd &u, 
                double *evdwl) {

  int env_derv_cols = single_bond_force_dervs.cols();
  int env_derv_size = single_bond_force_dervs.rows();
  int n_neighbors = env_derv_size / 3;

  // The value of last counter is the number of descriptors
  std::vector<int> last_index = nu[nu.size()-1];
  int n_d = last_index[last_index.size()-1] + 1; 

  // Initialize arrays.
  Bk_vals = Eigen::VectorXd::Zero(n_d);
  Bk_force_dervs = Eigen::MatrixXd::Zero(env_derv_size, n_d);
  Bk_force_dots = Eigen::VectorXd::Zero(env_derv_size);
  norm_squared = 0.0;

  Eigen::MatrixXcd dA_matr = Eigen::MatrixXd::Zero(n_d, env_derv_cols);
  for (int i = 0; i < nu.size(); i++) {
    std::vector<int> nu_list = nu[i];
    std::vector<int> single_bond_index = std::vector<int>(nu_list.end() - 2 - K, nu_list.end() - 2); // Get n1_l, n2_l, n3_l, etc.
    // Forward
    std::complex<double> A_fwd = 1;
    Eigen::VectorXcd dA = Eigen::VectorXcd::Ones(K);
    for (int t = 0; t < K - 1; t++) {
      A_fwd *= single_bond_vals(single_bond_index[t]);
      dA(t + 1) *= A_fwd;
    }
    // Backward
    std::complex<double> A_bwd = 1;
    for (int t = K - 1; t > 0; t--) {
      A_bwd *= single_bond_vals(single_bond_index[t]);
      dA(t - 1) *= A_bwd;
    }
    std::complex<double> A = A_fwd * single_bond_vals(single_bond_index[K - 1]);

    int counter = nu_list[nu_list.size() - 1];
    int m_index = nu_list[nu_list.size() - 2];
    Bk_vals(counter) += real(coeffs(m_index) * A);

//    // Prepare for partial force calculation
//    for (int t = 0; t < K; t++) {
//      dA_matr(counter, single_bond_index[t]) = coeffs(m_index) * dA(t);
//    }

    // Store force derivatives.
    for (int n = 0; n < n_neighbors; n++) {
      for (int comp = 0; comp < 3; comp++) {
        int ind = n * 3 + comp;
        std::complex<double> dA_dr = 0;
        for (int t = 0; t < K; t++) {
          dA_dr += dA(t) * single_bond_force_dervs(ind, single_bond_index[t]);
        }
        Bk_force_dervs(ind, counter) +=
            real(coeffs(m_index) * dA_dr);
      }
    }
  }

  // Compute descriptor norm and energy.
  Bk_force_dots = Bk_force_dervs * Bk_vals;
  norm_squared = Bk_vals.dot(Bk_vals);
  //Eigen::VectorXd beta_p = beta_matrix * Bk_vals;
  //*evdwl = Bk_vals.dot(beta_p) / norm_squared; 
  //Eigen::VectorXd w = 2 * (beta_p - *evdwl * Bk_vals) / norm_squared; // same size as Bk_vals 

  // Compute u(n1, l, m), where f_ik = u * dA/dr_ik
  //double factor;
  //u = w.transpose() * dA_matr;
   
}

void complex_single_bond_multiple_cutoffs(
    double **x, int *type, int jnum, int n_inner, int i, double xtmp,
    double ytmp, double ztmp, int *jlist,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, Eigen::VectorXcd &single_bond_vals,
    Eigen::MatrixXcd &single_bond_env_dervs,
    const Eigen::MatrixXd &cutoff_matrix) {

  // Initialize basis vectors and spherical harmonics.
  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gx = std::vector<double>(N, 0);
  std::vector<double> gy = std::vector<double>(N, 0);
  std::vector<double> gz = std::vector<double>(N, 0);

  int n_harmonics = (lmax + 1) * (lmax + 1);
  Eigen::VectorXcd h, hx, hy, hz;

  // Prepare LAMMPS variables.
  int central_species = type[i] - 1;
  double delx, dely, delz, rsq, r, g_val, gx_val, gy_val, gz_val;
  std::complex<double> bond, bond_x, bond_y, bond_z, h_val;
  int j, s, descriptor_counter;

  // Initialize vectors.
  int n_radial = n_species * N;
  int n_bond = n_radial * n_harmonics;
  single_bond_vals = Eigen::VectorXcd::Zero(n_bond);
  single_bond_env_dervs = Eigen::MatrixXcd::Zero(n_inner * 3, n_bond);

  // Initialize radial hyperparameters.
  std::vector<double> new_radial_hyps = radial_hyps;

  // Loop over neighbors.
  int n_count = 0;
  for (int jj = 0; jj < jnum; jj++) {
    j = jlist[jj];

    delx = x[j][0] - xtmp;
    dely = x[j][1] - ytmp;
    delz = x[j][2] - ztmp;
    rsq = delx * delx + dely * dely + delz * delz;
    r = sqrt(rsq);

    // Retrieve the cutoff.
    int s = type[j] - 1;
    double cutoff = cutoff_matrix(central_species, s);
    double cutforcesq = cutoff * cutoff;

    if (rsq < cutforcesq) { // minus a small value to prevent numerial error
      // Reset endpoint of the radial basis set.
      new_radial_hyps[1] = cutoff;

      calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, delx,
                       dely, delz, r, cutoff, N, new_radial_hyps, cutoff_hyps);
      get_complex_Y(h, hx, hy, hz, delx, dely, delz, lmax);

      // Store the products and their derivatives.
      descriptor_counter = s * N * n_harmonics;

      for (int radial_counter = 0; radial_counter < N; radial_counter++) {
        // Retrieve radial values.
        g_val = g[radial_counter];
        gx_val = gx[radial_counter];
        gy_val = gy[radial_counter];
        gz_val = gz[radial_counter];

        for (int angular_counter = 0; angular_counter < n_harmonics;
             angular_counter++) {

          h_val = h(angular_counter);
          bond = g_val * h_val;

          // Calculate derivatives with the product rule.
          bond_x = gx_val * h_val + g_val * hx(angular_counter);
          bond_y = gy_val * h_val + g_val * hy(angular_counter);
          bond_z = gz_val * h_val + g_val * hz(angular_counter);

          // Update single bond basis arrays.
          single_bond_vals(descriptor_counter) += bond;

          single_bond_env_dervs(n_count * 3, descriptor_counter) += bond_x;
          single_bond_env_dervs(n_count * 3 + 1, descriptor_counter) += bond_y;
          single_bond_env_dervs(n_count * 3 + 2, descriptor_counter) += bond_z;

          descriptor_counter++;
        }
      }
      n_count++;
    }
  }
}

std::vector<std::vector<int>> B3_indices(int n_radial, int lmax) {
  int n1, n2, n3, l1, l2, l3, m1, m2, m3, n1_l, n2_l, n3_l;
  int n_harmonics = (lmax + 1) * (lmax + 1);
  std::vector<std::vector<int>> index_list;
  int counter = 0;
  for (int n1 = 0; n1 < n_radial; n1++) {
    for (int n2 = n1; n2 < n_radial; n2++) {
      for (int n3 = n2; n3 < n_radial; n3++) {
        for (int l1 = 0; l1 < (lmax + 1); l1++) {
          int ind_1 = pow(lmax + 1, 4) * l1 * l1;
          for (int l2 = 0; l2 < (lmax + 1); l2++) {
            int ind_2 = ind_1 + pow(lmax + 1, 2) * l2 * l2 * (2 * l1 + 1);
            for (int l3 = 0; l3 < (lmax + 1); l3++) {
              if ((abs(l1 - l2) > l3) || (l3 > l1 + l2))
                continue;
              int ind_3 = ind_2 + l3 * l3 * (2 * l2 + 1) * (2 * l1 + 1);
              for (int m1 = 0; m1 < (2 * l1 + 1); m1++) {
                n1_l = n1 * n_harmonics + (l1 * l1 + m1);
                int ind_4 = ind_3 + m1 * (2 * l3 + 1) * (2 * l2 + 1);
                for (int m2 = 0; m2 < (2 * l2 + 1); m2++) {
                  n2_l = n2 * n_harmonics + (l2 * l2 + m2);
                  int ind_5 = ind_4 + m2 * (2 * l3 + 1);
                  for (int m3 = 0; m3 < (2 * l3 + 1); m3++) {
                    if (m1 + m2 + m3 - l1 - l2 - l3 != 0)
                      continue;
                    n3_l = n3 * n_harmonics + (l3 * l3 + m3);

                    int m_index = ind_5 + m3;
                    index_list.push_back({n1, n2, n3, l1, l2, l3, m1, m2, m3, n1_l, n2_l, n3_l, m_index, counter});
                  }
                }
              }
              counter++;
            }
          }
        }
      } 
    }
  }
  return index_list;
}
