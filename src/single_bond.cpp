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
    double bond, bond_x, bond_y, bond_z;

    for (int radial_counter = 0; radial_counter < N; radial_counter ++){
        for (int angular_counter = 0; angular_counter < number_of_harmonics;
             angular_counter ++){

            bond = g[radial_counter] * h[angular_counter];

            // calculate derivatives with the product rule
            bond_x = gx[radial_counter] * h[angular_counter] +
                     g[radial_counter] * hx[angular_counter];
            bond_y = gy[radial_counter] * h[angular_counter] +
                     g[radial_counter] * hy[angular_counter];
            bond_z = gz[radial_counter] * h[angular_counter] +
                     g[radial_counter] * hz[angular_counter];
            
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
