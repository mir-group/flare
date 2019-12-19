#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
using namespace std;

// Create test inputs for radial functions.
class SingleBondTest : public ::testing::Test{
    protected:

    // Choose an arbitrary position.
    double delta = 1e-8;

    double x0 = 4.23;
    double y0 = 5.12;
    double z0 = 2.15;

    double x0_pert = x0 + delta;
    double y0_pert = y0 + delta;
    double z0_pert  = z0 + delta;

    double x1 = 3.96;
    double y1 = 2.16;
    double z1 = -1.32;

    double x1_pert = x1 + delta;
    double y1_pert  = y1 + delta;
    double z1_pert = z1 + delta;

    double x = x1 - x0;
    double y = y1 - y0;
    double z = z1 - z0;
    double r = sqrt(x * x + y * y + z * z);

    // Perturb the environment atom.
    double x_delt = x1_pert - x0;
    double y_delt = y1_pert - y0;
    double z_delt = z1_pert - z0;

    double r_x = sqrt(x_delt * x_delt + y * y + z * z);
    double r_y = sqrt(x * x + y_delt * y_delt + z * z);
    double r_z = sqrt(x * x + y * y + z_delt * z_delt);

    // Perturb the central atom.
    double x0_delt = x1 - x0_pert;
    double y0_delt = y1 - y0_pert;
    double z0_delt = z1 - z0_pert;

    double r0_x = sqrt(x0_delt * x0_delt + y * y + z * z);
    double r0_y = sqrt(x * x + y0_delt * y0_delt + z * z);
    double r0_z = sqrt(x * x + y * y + z0_delt * z0_delt);

    // Prepare cutoff.
    double rcut = 7;
    std::vector<double> cutoff_hyps;
    void (*cutoff_function)(double *, double, double, std::vector<double>) =
        cos_cutoff;

    // Prepare spherical harmonics.
    int lmax = 10;
    int number_of_harmonics = (lmax + 1) * (lmax + 1);

    // Prepare radial basis set.
    double sigma = 1;
    double first_gauss = 1;
    double final_gauss = 6;
    int N = 10;
    std::vector<double> radial_hyps = {sigma, first_gauss, final_gauss};
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>) = equispaced_gaussians;

    // Initialize arrays.
    double * single_bond_vals;
    double * environment_dervs;
    double * central_dervs;

    double * single_bond_delt_x;
    double * single_bond_delt_y;
    double * single_bond_delt_z;

    double * env_dervs_delt_x;
    double * env_dervs_delt_y;
    double * env_dervs_delt_z;

    double * cent_dervs_delt_x;
    double * cent_dervs_delt_y;
    double * cent_dervs_delt_z;

    SingleBondTest(){
        single_bond_vals = new double[N * number_of_harmonics]();
        environment_dervs = new double[N * number_of_harmonics * 3]();
        central_dervs = new double[N * number_of_harmonics * 3]();

        single_bond_delt_x = new double[N * number_of_harmonics]();
        single_bond_delt_y = new double[N * number_of_harmonics]();
        single_bond_delt_z = new double[N * number_of_harmonics]();

        env_dervs_delt_x = new double[N * number_of_harmonics * 3]();
        env_dervs_delt_y = new double[N * number_of_harmonics * 3]();
        env_dervs_delt_z = new double[N * number_of_harmonics * 3]();

        cent_dervs_delt_x = new double[N * number_of_harmonics * 3]();
        cent_dervs_delt_y = new double[N * number_of_harmonics * 3]();
        cent_dervs_delt_z = new double[N * number_of_harmonics * 3]();
    }

    ~SingleBondTest(){
        delete [] single_bond_vals; delete[] environment_dervs;
        delete [] central_dervs; delete [] single_bond_delt_x;
        delete [] single_bond_delt_y; delete [] single_bond_delt_z;
        delete [] env_dervs_delt_x; delete [] env_dervs_delt_y;
        delete [] env_dervs_delt_z; delete [] cent_dervs_delt_x;
        delete [] cent_dervs_delt_y;  delete [] cent_dervs_delt_z;
    }
};

TEST_F(SingleBondTest, EnvironmentDervs){

    single_bond_update(single_bond_vals, environment_dervs, central_dervs,
                       basis_function, cutoff_function,
                       x, y, z, r, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);
    
    single_bond_update(single_bond_delt_x, env_dervs_delt_x, cent_dervs_delt_x,
                       basis_function, cutoff_function,
                       x_delt, y, z, r_x, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);
    
    single_bond_update(single_bond_delt_y, env_dervs_delt_y, cent_dervs_delt_y,
                       basis_function, cutoff_function,
                       x, y_delt, z, r_y, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);

    single_bond_update(single_bond_delt_z, env_dervs_delt_z, cent_dervs_delt_z,
                       basis_function, cutoff_function,
                       x, y, z_delt, r_z, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);
    
    double x_finite_diff, y_finite_diff, z_finite_diff, x_diff, y_diff, z_diff;
    int y_ind, z_ind;
    double tolerance = 1e-6;
    for (int n = 0; n < (number_of_harmonics * N); n++){
        // Check x derivative
        x_finite_diff = (single_bond_delt_x[n] - single_bond_vals[n]) / delta;
        x_diff = abs(x_finite_diff - environment_dervs[n]);
        EXPECT_LE(x_diff, tolerance);

        // Check y derivative
        y_ind = n + (N * number_of_harmonics);
        y_finite_diff = (single_bond_delt_y[n] - single_bond_vals[n]) / delta;
        y_diff = abs(y_finite_diff - environment_dervs[y_ind]);
        EXPECT_LE(y_diff, tolerance);

        // Check z derivative
        z_ind = n + (2 * N * number_of_harmonics);
        z_finite_diff = (single_bond_delt_z[n] - single_bond_vals[n]) / delta;
        z_diff = abs(z_finite_diff - environment_dervs[z_ind]);
        EXPECT_LE(z_diff, tolerance);
    }
}

TEST_F(SingleBondTest, CentralDervs){

    single_bond_update(single_bond_vals, environment_dervs, central_dervs,
                       basis_function, cutoff_function,
                       x, y, z, r, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);
    
    single_bond_update(single_bond_delt_x, env_dervs_delt_x, cent_dervs_delt_x,
                       basis_function, cutoff_function,
                       x0_delt, y, z, r0_x, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);
    
    single_bond_update(single_bond_delt_y, env_dervs_delt_y, cent_dervs_delt_y,
                       basis_function, cutoff_function,
                       x, y0_delt, z, r0_y, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);

    single_bond_update(single_bond_delt_z, env_dervs_delt_z, cent_dervs_delt_z,
                       basis_function, cutoff_function,
                       x, y, z0_delt, r0_z, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);
    
    double x_finite_diff, y_finite_diff, z_finite_diff, x_diff, y_diff, z_diff;
    int y_ind, z_ind;
    double tolerance = 1e-6;
    for (int n = 0; n < (number_of_harmonics * N); n++){
        // Check x derivative
        x_finite_diff = (single_bond_delt_x[n] - single_bond_vals[n]) / delta;
        x_diff = abs(x_finite_diff - central_dervs[n]);
        EXPECT_LE(x_diff, tolerance);

        // Check y derivative
        y_ind = n + (N * number_of_harmonics);
        y_finite_diff = (single_bond_delt_y[n] - single_bond_vals[n]) / delta;
        y_diff = abs(y_finite_diff - central_dervs[y_ind]);
        EXPECT_LE(y_diff, tolerance);

        // Check z derivative
        z_ind = n + (2 * N * number_of_harmonics);
        z_finite_diff = (single_bond_delt_z[n] - single_bond_vals[n]) / delta;
        z_diff = abs(z_finite_diff - central_dervs[z_ind]);
        EXPECT_LE(z_diff, tolerance);
    }
}
