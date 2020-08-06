#include "lammps_descriptor.h"
#include "radial.h"
#include "y_grad.h"
#include <cmath>
#include <iostream>


void single_bond(double **x, int atom_index, int *type, int inum,
    int *ilist, int *numneigh, int **firstneigh,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
        int, std::vector<double>)> basis_function,
    std::function<void(std::vector<double> &, double, double,
        std::vector<double>)> cutoff_function,
    double cutoff, int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps,
    Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &environment_force_dervs,
    Eigen::MatrixXd &central_force_dervs){

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
    int i = ilist[atom_index];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int itype = type[i];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    double delx, dely, delz, rsq, r, bond, bond_x, bond_y, bond_z, g_val,
        gx_val, gy_val, gz_val, h_val;
    int j, s, descriptor_counter;
    double cutforcesq = cutoff * cutoff;

    // Initialize vectors.
    int n_radial = n_species * N;
    int n_bond = n_radial * n_harmonics;
    single_bond_vals = Eigen::VectorXd::Zero(n_bond);
    environment_force_dervs = Eigen::MatrixXd::Zero(jnum * 3, n_bond);
    central_force_dervs = Eigen::MatrixXd::Zero(3, n_bond);

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

                environment_force_dervs(jj * 3, descriptor_counter) += bond_x;
                environment_force_dervs(jj * 3 + 1, descriptor_counter) +=
                    bond_y;
                environment_force_dervs(jj * 3 + 2, descriptor_counter) +=
                    bond_z;

                central_force_dervs(0, descriptor_counter) -= bond_x;
                central_force_dervs(1, descriptor_counter) -= bond_y;
                central_force_dervs(2, descriptor_counter) -= bond_z;

                descriptor_counter++;
                }  
            }
        }
    }
}
