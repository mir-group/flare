#include "gtest/gtest.h"
#include "ace.h"
#include <chrono>
#include <iostream>
#include <cmath>
using namespace std;

// Create test inputs for radial functions.
class SingleBondTest : public ::testing::Test{
    protected:

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

    // Initialize arrays.
    double * single_bond_vals;
    double * environment_dervs;
    double * central_dervs;

    SingleBondTest(){
        single_bond_vals = new double[N * number_of_harmonics];
        environment_dervs = new double[N * number_of_harmonics * 3];
        central_dervs = new double[N * number_of_harmonics * 3];
    }

    ~SingleBondTest(){
        delete [] single_bond_vals; delete[] environment_dervs;
        delete [] central_dervs;
    }
};

TEST_F(SingleBondTest, Dummy){

    single_bond_update(single_bond_vals, environment_dervs, central_dervs,
                       basis_function, cutoff_function,
                       x, y, z, r, rcut, N, lmax,
                       radial_hyps, cutoff_hyps);
    
    cout << single_bond_vals[0] << '\n';
    cout << environment_dervs[N * number_of_harmonics - 1] << '\n';

}
