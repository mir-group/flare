#include "ace.h"
#include <chrono>
#include <iostream>
#include <cmath>
using namespace std;

int main(){
    // Choose an arbitrary position.
    double x = 2.19;
    double y = 1.23;
    double z = -0.24;
    double r = sqrt(x * x + y * y + z * z);

    // Perturb the position.
    double delta = 1e-8;
    double x_delt = x + delta;
    double y_delt = y + delta;
    double z_delt = z + delta;
    double r_delt = r + delta;
    double r_x = sqrt(x_delt * x_delt + y * y + z * z);
    double r_y = sqrt(x * x + y_delt * y_delt + z * z);
    double r_z = sqrt(x * x + y * y + z_delt * z_delt);

    // Prepare cutoff.
    double rcut = 7;
    double * cutoff_hyps;
    void (*cutoff_function)(double *, double, double, double *) = cos_cutoff;

    // Prepare spherical harmonics.
    int lmax = 10;
    int number_of_harmonics = (lmax + 1) * (lmax + 1);

    // Prepare radial basis set.
    double sigma = 1;
    double first_gauss = 1;
    double final_gauss = 6;
    int N = 10;
    double radial_hyps[3] = {sigma, first_gauss, final_gauss};
    void (*basis_function)(double *, double *, double, int, double *) =
        equispaced_gaussians;

    // Time radial basis.
    double * g = new double[N];
    double * gx = new double[N];
    double * gy = new double[N];
    double * gz = new double[N];

    auto t1 = std::chrono::high_resolution_clock::now();

    int reps = 10000;
    for (int n = 0; n < reps; n++){
    calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, 
                     x, y, z, r, rcut, N, radial_hyps, cutoff_hyps);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto tot_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    auto mean_time = tot_time / reps;
    std::cout << "calculating the radial basis took "
              << mean_time << " microseconds\n";

    delete [] g; delete [] gx; delete [] gy; delete [] gz;

    // Time spherical harmonics.
    double * h = new double[number_of_harmonics];
    double * hx = new double[number_of_harmonics];
    double * hy = new double[number_of_harmonics];
    double * hz = new double[number_of_harmonics];

    t1 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < reps; n++){
    get_Y(h, hx, hy, hz, x, y, z, lmax);
    }
    t2 = std::chrono::high_resolution_clock::now();

    tot_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    mean_time = tot_time / reps;
    std::cout << "calculating the spherical harmonics took "
              << mean_time << " microseconds\n";

    delete [] h; delete [] hx; delete [] hy; delete [] hz;

    // Time single bond update.
    double * single_bond_vals = new double[N * number_of_harmonics]();
    double * environment_dervs = new double[N * number_of_harmonics * 3]();
    double * central_dervs = new double[N * number_of_harmonics * 3]();

    t1 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < reps; n++){
    single_bond_update(single_bond_vals, environment_dervs, central_dervs,
                      basis_function, cutoff_function,
                      x, y, z, r, rcut, N, lmax,
                      radial_hyps, cutoff_hyps);
    }
    t2 = std::chrono::high_resolution_clock::now();

    tot_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    mean_time = tot_time / reps;
    std::cout << "calculating the single bond basis took "
              << mean_time << " microseconds\n";
    
    delete [] single_bond_vals; delete[] environment_dervs;
    delete [] central_dervs; 
}
