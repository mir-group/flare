#include "single_bond.h"
#include "local_environment.h"
#include "radial.h"
#include "y_grad.h"
#include <cmath>
#include <iostream>
using namespace std;

void single_bond_update_env(
    Eigen::VectorXd &single_bond_vals, Eigen::MatrixXd &force_dervs,
    Eigen::MatrixXd &stress_dervs,
    void (*basis_function)(double *, double *, double, int, vector<double>),
    void (*cutoff_function)(double *, double, double, vector<double>), double x,
    double y, double z, double r, int s, int environoment_index,
    int central_index, double rcut, int N, int lmax,
    const vector<double> &radial_hyps, const vector<double> &cutoff_hyps) {

  // Calculate radial basis values.
  double *g = new double[N];
  double *gx = new double[N];
  double *gy = new double[N];
  double *gz = new double[N];

  calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, x, y, z, r,
                   rcut, N, radial_hyps, cutoff_hyps);

  // Calculate spherical harmonics.
  int number_of_harmonics = (lmax + 1) * (lmax + 1);

  vector<double> h = vector<double>(number_of_harmonics, 0);
  vector<double> hx = vector<double>(number_of_harmonics, 0);
  vector<double> hy = vector<double>(number_of_harmonics, 0);
  vector<double> hz = vector<double>(number_of_harmonics, 0);

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

  // Deallocate memory.
  delete[] g;
  delete[] gx;
  delete[] gy;
  delete[] gz;
}

void single_bond_sum_env(
    Eigen::VectorXd &single_bond_vals, Eigen::MatrixXd &force_dervs,
    Eigen::MatrixXd &stress_dervs,
    void (*basis_function)(double *, double *, double, int, vector<double>),
    void (*cutoff_function)(double *, double, double, vector<double>),
    const LocalEnvironment &env, int descriptor_index, int N, int lmax,
    const vector<double> &radial_hyps, const vector<double> &cutoff_hyps) {

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
