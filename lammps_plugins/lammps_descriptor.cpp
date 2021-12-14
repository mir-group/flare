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

void single_bond(
    double **x, int *type, int jnum, int n_inner, int i, double xtmp,
    double ytmp, double ztmp, int *jlist,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    double cutoff, int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &single_bond_env_dervs) {

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
  int itype = type[i];
  double delx, dely, delz, rsq, r, bond, bond_x, bond_y, bond_z, g_val, gx_val,
      gy_val, gz_val, h_val;
  int j, s, descriptor_counter;
  double cutforcesq = cutoff * cutoff;

  // Initialize vectors.
  int n_radial = n_species * N;
  int n_bond = n_radial * n_harmonics;
  single_bond_vals = Eigen::VectorXd::Zero(n_bond);
  single_bond_env_dervs = Eigen::MatrixXd::Zero(n_inner * 3, n_bond);

  // Loop over neighbors.
  int n_count = 0;
  for (int jj = 0; jj < jnum; jj++) {
    j = jlist[jj];

    delx = x[j][0] - xtmp;
    dely = x[j][1] - ytmp;
    delz = x[j][2] - ztmp;
    rsq = delx * delx + dely * dely + delz * delz;
    r = sqrt(rsq);

    if (rsq < cutforcesq) { // minus a small value to prevent numerial error
      s = type[j] - 1;
      calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, delx,
                       dely, delz, r, cutoff, N, radial_hyps, cutoff_hyps);
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
          //printf("i = %d, j = %d, n = %d, lm = %d, idx = %d, bond = %g %g %g %g\n", i, j, radial_counter, angular_counter, descriptor_counter, bond, bond_x, bond_y, bond_z);

          descriptor_counter++;
        }
      }
      n_count++;
    }
  }
  /*
  printf("i = %d, d =", i);
  for(int d = 0; d < n_bond; d++){
    printf(" %g", single_bond_vals(d));
  }
  printf("\n");
  */
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
