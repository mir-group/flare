#include <cmath>
#include <iostream>
#include "ace.h"
using namespace std;

void single_bond_update_env(
    vector<double> & single_bond_vals,
    Eigen::MatrixXd & force_dervs, Eigen::MatrixXd & stress_dervs,
    void (*basis_function)(double *, double *, double, int, vector<double>),
    void (*cutoff_function)(double *, double, double, vector<double>),
    double x, double y, double z, double r, int s,
    int environoment_index, int central_index,
    double rcut, int N, int lmax,
    const vector<double> & radial_hyps, const vector<double> & cutoff_hyps){

    // Calculate radial basis values.
    double * g = new double[N];
    double * gx = new double[N];
    double * gy = new double[N];
    double * gz = new double[N];

    calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, 
                     x, y, z, r, rcut, N, radial_hyps, cutoff_hyps);

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

            // Calculate derivatives with the product rule.
            bond_x = gx_val * h_val +
                     g_val * hx[angular_counter];
            bond_y = gy_val * h_val +
                     g_val * hy[angular_counter];
            bond_z = gz_val * h_val +
                     g_val * hz[angular_counter];

            // Update single bond basis arrays.
            single_bond_vals[descriptor_counter] += bond;

            force_dervs(environoment_index * 3, descriptor_counter) +=
                bond_x;
            force_dervs(environoment_index * 3 + 1, descriptor_counter) +=
                bond_y;
            force_dervs(environoment_index * 3 + 2, descriptor_counter) +=
                bond_z;

            force_dervs(central_index * 3, descriptor_counter) -= bond_x;
            force_dervs(central_index * 3 + 1, descriptor_counter) -= bond_y;
            force_dervs(central_index * 3 + 2, descriptor_counter) -= bond_z;

            stress_dervs(0, descriptor_counter) += bond_x * x;
            stress_dervs(1, descriptor_counter) += bond_x * y;
            stress_dervs(2, descriptor_counter) += bond_x * z;
            stress_dervs(3, descriptor_counter) += bond_y * y;
            stress_dervs(4, descriptor_counter) += bond_y * z;
            stress_dervs(5, descriptor_counter) += bond_z * z;

            descriptor_counter ++;
            }
    }

    // Deallocate memory.
    delete [] g; delete [] gx; delete [] gy; delete [] gz;
}

void single_bond_update(
double * single_bond_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, vector<double>),
void (*cutoff_function)(double *, double, double, vector<double>),
double x, double y, double z, double r, double rcut, int N, int lmax,
vector<double> radial_hyps, vector<double> cutoff_hyps){

    // Calculate radial basis values.
    double * g = new double[N];
    double * gx = new double[N];
    double * gy = new double[N];
    double * gz = new double[N];

    calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, 
                     x, y, z, r, rcut, N, radial_hyps, cutoff_hyps);

    // Calculate spherical harmonics.
    int number_of_harmonics = (lmax + 1) * (lmax + 1);

    vector<double> h = vector<double>(number_of_harmonics, 0);
    vector<double> hx = vector<double>(number_of_harmonics, 0);
    vector<double> hy = vector<double>(number_of_harmonics, 0);
    vector<double> hz = vector<double>(number_of_harmonics, 0);

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
}

void single_bond_sum_env(
    vector<double> & single_bond_vals,
    Eigen::MatrixXd & force_dervs, Eigen::MatrixXd & stress_dervs,
    void (*basis_function)(double *, double *, double, int, vector<double>),
    void (*cutoff_function)(double *, double, double, vector<double>),
    const LocalEnvironment & env, double rcut, int N, int lmax,
    const vector<double> & radial_hyps, const vector<double> & cutoff_hyps){

    int noa = env.rs.size();
    int cent_ind = env.central_index;
    double x, y, z, r;
    int s, env_ind;

    for (int atom_index = 0; atom_index < noa; atom_index ++){
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

void single_bond_sum(
double * single_bond_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, vector<double>),
void (*cutoff_function)(double *, double, double, vector<double>),
double * xs, double * ys, double * zs, double * rs, int * species,
int noa, double rcut, int N, int lmax,
vector<double> radial_hyps, vector<double> cutoff_hyps){

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
