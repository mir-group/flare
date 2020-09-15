#include "single_bond.h"
#include "compact_structure.h"
#include "local_environment.h"
#include "radial.h"
#include "y_grad.h"
#include <cmath>
#include <iostream>

void single_bond_update_env(
    Eigen::VectorXd &single_bond_vals, Eigen::MatrixXd &force_dervs,
    Eigen::MatrixXd &stress_dervs,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    double x, double y, double z, double r, int s, int environoment_index,
    int central_index, double rcut, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps) {

  // Calculate radial basis values.
  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gx = std::vector<double>(N, 0);
  std::vector<double> gy = std::vector<double>(N, 0);
  std::vector<double> gz = std::vector<double>(N, 0);

  calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, x, y, z, r,
                   rcut, N, radial_hyps, cutoff_hyps);

  // Calculate spherical harmonics.
  int number_of_harmonics = (lmax + 1) * (lmax + 1);

  std::vector<double> h = std::vector<double>(number_of_harmonics, 0);
  std::vector<double> hx = std::vector<double>(number_of_harmonics, 0);
  std::vector<double> hy = std::vector<double>(number_of_harmonics, 0);
  std::vector<double> hz = std::vector<double>(number_of_harmonics, 0);

  get_Y(h, hx, hy, hz, x, y, z, lmax);

  // Store the products and their derivatives.
  int no_bond_vals = N * number_of_harmonics;
  int descriptor_counter = s * no_bond_vals;
  double bond, bond_x, bond_y, bond_z, g_val, gx_val, gy_val, gz_val, h_val;

  for (int radial_counter = 0; radial_counter < N; radial_counter++) {
    // Retrieve radial values.
    g_val = g[radial_counter];
    gx_val = gx[radial_counter];
    gy_val = gy[radial_counter];
    gz_val = gz[radial_counter];

    for (int angular_counter = 0; angular_counter < number_of_harmonics;
         angular_counter++) {

      h_val = h[angular_counter];
      bond = g_val * h_val;

      // Calculate derivatives with the product rule.
      bond_x = gx_val * h_val + g_val * hx[angular_counter];
      bond_y = gy_val * h_val + g_val * hy[angular_counter];
      bond_z = gz_val * h_val + g_val * hz[angular_counter];

      // Update single bond basis arrays.
      single_bond_vals(descriptor_counter) += bond;

      force_dervs(environoment_index * 3, descriptor_counter) += bond_x;
      force_dervs(environoment_index * 3 + 1, descriptor_counter) += bond_y;
      force_dervs(environoment_index * 3 + 2, descriptor_counter) += bond_z;

      force_dervs(central_index * 3, descriptor_counter) -= bond_x;
      force_dervs(central_index * 3 + 1, descriptor_counter) -= bond_y;
      force_dervs(central_index * 3 + 2, descriptor_counter) -= bond_z;

      stress_dervs(0, descriptor_counter) += bond_x * x;
      stress_dervs(1, descriptor_counter) += bond_x * y;
      stress_dervs(2, descriptor_counter) += bond_x * z;
      stress_dervs(3, descriptor_counter) += bond_y * y;
      stress_dervs(4, descriptor_counter) += bond_y * z;
      stress_dervs(5, descriptor_counter) += bond_z * z;

      descriptor_counter++;
    }
  }
}

void single_bond_sum_env(
    Eigen::VectorXd &single_bond_vals, Eigen::MatrixXd &force_dervs,
    Eigen::MatrixXd &stress_dervs,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    const LocalEnvironment &env, int descriptor_index, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps) {

  int noa = env.many_body_indices[descriptor_index].size();
  int cent_ind = env.central_index;
  double x, y, z, r;
  int s, env_ind, atom_index;
  double rcut = env.many_body_cutoffs[descriptor_index];

  for (int n = 0; n < noa; n++) {
    atom_index = env.many_body_indices[descriptor_index][n];
    x = env.xs[atom_index];
    y = env.ys[atom_index];
    z = env.zs[atom_index];
    r = env.rs[atom_index];
    s = env.environment_species[atom_index];
    env_ind = env.environment_indices[atom_index];

    single_bond_update_env(single_bond_vals, force_dervs, stress_dervs,
                           basis_function, cutoff_function, x, y, z, r, s,
                           env_ind, cent_ind, rcut, N, lmax, radial_hyps,
                           cutoff_hyps);
  }
}

void single_bond_sum_struc(Eigen::MatrixXd &single_bond_vals,
                           Eigen::MatrixXd &force_dervs,
                           const CompactStructure &structure,
                           int descriptor_index) {
  // Initialize single bond arrays.
  int n_atoms = structure.noa;
  int n_neighbors = structure.n_neighbors;
  int nos = structure.descriptor_calculators[descriptor_index]
                ->descriptor_settings[0];
  int N = structure.descriptor_calculators[descriptor_index]
              ->descriptor_settings[1];
  int lmax = structure.descriptor_calculators[descriptor_index]
                 ->descriptor_settings[2];
  int number_of_harmonics = (lmax + 1) * (lmax + 1);
  int no_bond_vals = N * number_of_harmonics;
  int single_bond_size = no_bond_vals * nos;
  single_bond_vals = Eigen::MatrixXd::Zero(n_atoms, single_bond_size);
  force_dervs = Eigen::MatrixXd::Zero(n_neighbors * 3, single_bond_size);

  std::vector<double> radial_hyps =
      structure.descriptor_calculators[descriptor_index]->radial_hyps;
  std::vector<double> cutoff_hyps =
      structure.descriptor_calculators[descriptor_index]->cutoff_hyps;
  // TODO: Make rcut an attribute of the descriptor calculator.
  double rcut = radial_hyps[1];
  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      radial_function =
          structure.descriptor_calculators[descriptor_index]->radial_pointer;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function =
          structure.descriptor_calculators[descriptor_index]->cutoff_pointer;

#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);

    // Initialize radial and spherical harmonic vectors.
    std::vector<double> g = std::vector<double>(N, 0);
    std::vector<double> gx = std::vector<double>(N, 0);
    std::vector<double> gy = std::vector<double>(N, 0);
    std::vector<double> gz = std::vector<double>(N, 0);

    std::vector<double> h = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hx = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hy = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hz = std::vector<double>(number_of_harmonics, 0);

    double x, y, z, r, bond, bond_x, bond_y, bond_z, g_val, gx_val, gy_val,
        gz_val, h_val;
    int s, neigh_index, descriptor_counter;
    for (int j = 0; j < i_neighbors; j++) {
      neigh_index = rel_index + j;
      r = structure.relative_positions(neigh_index, 0);
      if (r > rcut)
        continue; // Skip if outside cutoff.
      x = structure.relative_positions(neigh_index, 1);
      y = structure.relative_positions(neigh_index, 2);
      z = structure.relative_positions(neigh_index, 3);
      s = structure.neighbor_species(neigh_index);

      // Compute radial basis values and spherical harmonics.
      calculate_radial(g, gx, gy, gz, radial_function, cutoff_function, x, y, z,
                       r, rcut, N, radial_hyps, cutoff_hyps);
      get_Y(h, hx, hy, hz, x, y, z, lmax);

      // Store the products and their derivatives.
      descriptor_counter = s * no_bond_vals;

      for (int radial_counter = 0; radial_counter < N; radial_counter++) {
        // Retrieve radial values.
        g_val = g[radial_counter];
        gx_val = gx[radial_counter];
        gy_val = gy[radial_counter];
        gz_val = gz[radial_counter];

        for (int angular_counter = 0; angular_counter < number_of_harmonics;
             angular_counter++) {

          h_val = h[angular_counter];
          bond = g_val * h_val;

          // Calculate derivatives with the product rule.
          bond_x = gx_val * h_val + g_val * hx[angular_counter];
          bond_y = gy_val * h_val + g_val * hy[angular_counter];
          bond_z = gz_val * h_val + g_val * hz[angular_counter];

          // Update single bond arrays.
          single_bond_vals(i, descriptor_counter) += bond;
          force_dervs(neigh_index * 3, descriptor_counter) += bond_x;
          force_dervs(neigh_index * 3 + 1, descriptor_counter) += bond_y;
          force_dervs(neigh_index * 3 + 2, descriptor_counter) += bond_z;

          descriptor_counter++;
        }
      }
    }
  }
}
