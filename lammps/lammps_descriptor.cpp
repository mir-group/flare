#include "lammps_descriptor.h"
#include "radial.h"
#include "y_grad.h"
#include <cmath>
#include <iostream>


void single_bond(double **x, int *type, int jnum, int i, double xtmp,
    double ytmp, double ztmp, int *jlist,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
        int, std::vector<double>)> basis_function,
    std::function<void(std::vector<double> &, double, double,
        std::vector<double>)> cutoff_function,
    double cutoff, int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps,
    Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &single_bond_env_dervs,
    Eigen::MatrixXd &single_bond_cent_dervs){

    // Initialize basis vectors and spherical harmonics.
    std::vector<double> g = std::vector<double> (N, 0);
    std::vector<double> gx = std::vector<double> (N, 0);
    std::vector<double> gy = std::vector<double> (N, 0);
    std::vector<double> gz = std::vector<double> (N, 0);

    int n_harmonics = (lmax + 1) * (lmax + 1);
    std::vector<double> h = std::vector<double>(n_harmonics, 0);
    std::vector<double> hx = std::vector<double>(n_harmonics, 0);
    std::vector<double> hy = std::vector<double>(n_harmonics, 0);
    std::vector<double> hz = std::vector<double>(n_harmonics, 0);

    // Prepare LAMMPS variables.
    int itype = type[i];
    double delx, dely, delz, rsq, r, bond, bond_x, bond_y, bond_z, g_val,
        gx_val, gy_val, gz_val, h_val;
    int j, s, descriptor_counter;
    double cutforcesq = cutoff * cutoff;

    // Initialize vectors.
    int n_radial = n_species * N;
    int n_bond = n_radial * n_harmonics;
    single_bond_vals = Eigen::VectorXd::Zero(n_bond);
    single_bond_env_dervs = Eigen::MatrixXd::Zero(jnum * 3, n_bond);
    single_bond_cent_dervs = Eigen::MatrixXd::Zero(3, n_bond);

    // Loop over neighbors.
    for (int jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;
      r = sqrt(rsq);

      if (rsq < cutforcesq){
        s = type[j] - 1;
        calculate_radial(g, gx, gy, gz, basis_function, cutoff_function,
                         delx, dely, delz, r, cutoff, N, radial_hyps,
                         cutoff_hyps);
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

                single_bond_env_dervs(jj * 3, descriptor_counter) += bond_x;
                single_bond_env_dervs(jj * 3 + 1, descriptor_counter) +=
                    bond_y;
                single_bond_env_dervs(jj * 3 + 2, descriptor_counter) +=
                    bond_z;

                single_bond_cent_dervs(0, descriptor_counter) -= bond_x;
                single_bond_cent_dervs(1, descriptor_counter) -= bond_y;
                single_bond_cent_dervs(2, descriptor_counter) -= bond_z;

                descriptor_counter++;
                }  
            }
        }
    }
}

void B2_descriptor(Eigen::VectorXd &B2_vals, Eigen::MatrixXd &B2_env_dervs,
                   Eigen::MatrixXd &B2_cent_dervs, double &norm_squared,
                   Eigen::VectorXd &B2_env_dot,  Eigen::VectorXd &B2_cent_dot,
                   const Eigen::VectorXd &single_bond_vals,
                   const Eigen::MatrixXd &single_bond_env_dervs,
                   const Eigen::MatrixXd &single_bond_cent_dervs,
                   int n_species, int N, int lmax){

  int env_derv_size = single_bond_env_dervs.rows();
  int neigh_size = env_derv_size / 3;
  int n_radial = n_species * N;
  int n_harmonics = (lmax + 1) * (lmax + 1);
  int n_descriptors = (n_radial * (n_radial + 1) / 2) * (lmax + 1);

  int n1_l, n2_l, counter, n1_count, n2_count;

  // Zero the B2 vectors and matrices.  
  B2_vals = Eigen::VectorXd::Zero(n_descriptors);
  B2_env_dervs = Eigen::MatrixXd::Zero(env_derv_size, n_descriptors);
  B2_cent_dervs = Eigen::MatrixXd::Zero(3, n_descriptors);
  B2_env_dot = Eigen::VectorXd::Zero(env_derv_size);

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

          // Store environment force derivatives.
          for (int atom_index = 0; atom_index < neigh_size; atom_index++) {
            for (int comp = 0; comp < 3; comp++) {
              B2_env_dervs(atom_index * 3 + comp, counter) +=
                  single_bond_vals(n1_l) *
                      single_bond_env_dervs(atom_index * 3 + comp, n2_l) +
                  single_bond_env_dervs(atom_index * 3 + comp, n1_l) *
                      single_bond_vals(n2_l);
            }
          }

          // Store central force derivatives.
          for (int comp = 0; comp < 3; comp++){
              B2_cent_dervs(comp, counter) +=
                single_bond_vals(n1_l) *
                    single_bond_cent_dervs(comp, n2_l) +
                single_bond_cent_dervs(comp, n1_l) *
                    single_bond_vals(n2_l);
          }
        }
      }
    }
  }

  // Compute descriptor norm and dot products.
  norm_squared = B2_vals.dot(B2_vals);
  B2_env_dot = B2_env_dervs * B2_vals;
}
