#include <cmath>
#include "ace.h"

void single_bond_update(
double * single_bond_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, double *),
void (*cutoff_function)(double *, double, double, double *),
double x, double y, double z, double r, double rcut, int N, int lmax,
double * radial_hyps, double * cutoff_hyps){

    // Calculate radial basis values.
    double * g = new double[N];
    double * gx = new double[N];
    double * gy = new double[N];
    double * gz = new double[N];

    calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, 
                     x, y, z, r, rcut, N, radial_hyps, cutoff_hyps);

    // Calculate spherical harmonics.
    int number_of_harmonics = (lmax + 1) * (lmax + 1);

    double * h = new double[number_of_harmonics];
    double * hx = new double[number_of_harmonics];
    double * hy = new double[number_of_harmonics];
    double * hz = new double[number_of_harmonics];

    get_Y(h, hx, hy, hz, x, y, z, lmax);

    // Store the products and their derivatives.
    int single_bond_counter = 0;
    int no_bond_vals = N * number_of_harmonics;
    int y_ind, z_ind;
    double bond, bond_x, bond_y, bond_z, g_val, gx_val, gy_val, gz_val, h_val;

    for (int radial_counter = 0; radial_counter < N; radial_counter ++){
        // Retrieve radial values.
        g_val = g[radial_counter];
        gx_val = gx[radial_counter];
        gy_val = gy[radial_counter];
        gz_val = gz[radial_counter];

        for (int angular_counter = 0; angular_counter < number_of_harmonics;
             angular_counter ++){

            h_val = h[angular_counter];
            bond = g_val * h_val;

            // calculate derivatives with the product rule
            bond_x = gx_val * h_val +
                     g_val * hx[angular_counter];
            bond_y = gy_val * h_val +
                     g_val * hy[angular_counter];
            bond_z = gz_val * h_val +
                     g_val * hz[angular_counter];

            // update single bond basis arrays
            y_ind = single_bond_counter + no_bond_vals;
            z_ind = y_ind + no_bond_vals;

            single_bond_vals[single_bond_counter] += bond;

            environment_dervs[single_bond_counter] += bond_x;
            environment_dervs[y_ind] += bond_y;
            environment_dervs[z_ind] += bond_z;

            central_dervs[single_bond_counter] -= bond_x;
            central_dervs[y_ind] -= bond_y;
            central_dervs[z_ind] -= bond_z;

            single_bond_counter ++;
            }
    }

    // Deallocate memory.
    delete [] g; delete [] gx; delete [] gy; delete [] gz;
    delete [] h; delete [] hx; delete [] hy; delete [] hz;
}

void single_bond_sum(
double * single_bond_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, double *),
void (*cutoff_function)(double *, double, double, double *),
double * xs, double * ys, double * zs, double * rs, int * species,
int noa, double rcut, int N, int lmax,
double * radial_hyps, double * cutoff_hyps){

    int no_basis_vals = N * (lmax + 1) * (lmax + 1);
    int no_derv_vals = 3 * no_basis_vals;

    // Loop over atoms.
    int atom_index, s;
    double x, y, z, r;
    double * bond_ind, * env_ind, * cent_ind;

    for (atom_index = 0; atom_index < noa; atom_index ++){
        x = xs[atom_index];
        y = ys[atom_index];
        z = zs[atom_index];
        r = rs[atom_index];
        s = species[atom_index];

        bond_ind = & single_bond_vals[s * no_basis_vals];
        env_ind = & environment_dervs[no_derv_vals * (s * noa + atom_index)];
        cent_ind = & central_dervs[s * no_derv_vals];

        single_bond_update(bond_ind, env_ind, cent_ind,
                           basis_function, cutoff_function, x, y, z, r, rcut,
                           N, lmax, radial_hyps, cutoff_hyps);
    }
}
