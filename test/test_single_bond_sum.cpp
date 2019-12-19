#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
using namespace std;

class BondSum : public ::testing::Test{
    protected:

    // Choose arbitrary positions.
    double delta = 1e-8;
    double xs [5] = {1.09, -2.03, 1.52, -0.21, 0.62};
    double ys [5] = {-3.15, 0.89, -0.26, -1.81, 2.62};
    double zs [5] = {2.11, -1.03, 0.52, -0.52, 0.87};
    double env_delt [5] = {-3.15, 0.89, -0.26 + delta, -1.81, 2.62};
    double cent_delt [5] =
        {2.11 - delta, -1.03 - delta, 0.52 - delta, -0.52 - delta,
         0.87 - delta};
    int species [5] = {0, 2, 1, 0, 1};

    int spec = 1;
    int atom_index = 2;
    int cart = 1;
 
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
    int nos = 3;
    int noa = 5;

    double * single_bond_vals;
    double * environment_dervs;
    double * central_dervs;

    double * single_bond_vals_2;
    double * environment_dervs_2;
    double * central_dervs_2;

    double * single_bond_vals_3;
    double * environment_dervs_3;
    double * central_dervs_3;

    double rs [5];
    double rs_env_delt [5];
    double rs_cent_delt [5];
    double x, y, z, coord_delt, coord_2;
    BondSum(){
        for (int atom = 0; atom < 5; atom ++){
            x = xs[atom];
            y = ys[atom];
            z = zs[atom];

            coord_delt = env_delt[atom];
            coord_2 = cent_delt[atom];

            rs[atom] = sqrt(x * x + y * y + z * z);
            rs_env_delt[atom] = sqrt(x * x + coord_delt * coord_delt + z * z);
            rs_cent_delt[atom] = sqrt(x * x + y * y + coord_2 * coord_2);
            }

        single_bond_vals = new double[nos * N * number_of_harmonics]();
        environment_dervs = 
            new double[nos * noa * N * number_of_harmonics * 3]();
        central_dervs =
            new double[nos * N * number_of_harmonics * 3]();

        single_bond_vals_2 = new double[nos * N * number_of_harmonics]();
        environment_dervs_2 = 
            new double[nos * noa * N * number_of_harmonics * 3]();
        central_dervs_2 =
            new double[nos * N * number_of_harmonics * 3]();

        single_bond_vals_3 = new double[nos * N * number_of_harmonics]();
        environment_dervs_3 = 
            new double[nos * noa * N * number_of_harmonics * 3]();
        central_dervs_3 =
            new double[nos * N * number_of_harmonics * 3]();
    }

    ~BondSum(){
        delete [] single_bond_vals;
        delete[] environment_dervs;
        delete [] central_dervs;
        delete [] single_bond_vals_2;
        delete[] environment_dervs_2;
        delete [] central_dervs_2;
    }
};

TEST_F(BondSum, TestEnvDerv){

    single_bond_sum(single_bond_vals, environment_dervs, central_dervs,
                    basis_function, cutoff_function, xs, ys, zs, rs,
                    species, noa, rcut, N, lmax,
                    radial_hyps, cutoff_hyps);
                    
    single_bond_sum(single_bond_vals_2, environment_dervs_2, central_dervs_2,
                    basis_function, cutoff_function, xs, env_delt, zs,
                    rs_env_delt, species, noa, rcut, N, lmax,
                    radial_hyps, cutoff_hyps);

    double finite_diff, exact, diff;
    double tolerance = 1e-6;

    int m, n;

    for (int p = 0; p < N * number_of_harmonics; p++){
        m = spec * N * number_of_harmonics + p;
        n = (N * number_of_harmonics) *
            (3 * spec * noa + 3 * atom_index + cart) + p;

        finite_diff = (single_bond_vals_2[m] - single_bond_vals[m]) / delta;
        exact = environment_dervs[n];
        diff = abs(finite_diff - exact);

        EXPECT_LE(diff, tolerance);
    }
}

TEST_F(BondSum, TestCentDerv){
    single_bond_sum(single_bond_vals, environment_dervs, central_dervs,
                    basis_function, cutoff_function, xs, ys, zs, rs,
                    species, noa, rcut, N, lmax,
                    radial_hyps, cutoff_hyps);
                    
    single_bond_sum(single_bond_vals_3, environment_dervs_3, central_dervs_3,
                    basis_function, cutoff_function, xs, ys, cent_delt,
                    rs_cent_delt, species, noa, rcut, N, lmax,
                    radial_hyps, cutoff_hyps);

    double finite_diff, exact, diff;
    double tolerance = 1e-6;

    int m, n;

    for (int spec = 0; spec < nos; spec ++){
        for (int p = 0; p < N * number_of_harmonics; p++){
            m = (N * number_of_harmonics) * spec + p;
            n = (N * number_of_harmonics) * (3 * spec + 2) + p;

            finite_diff = 
                (single_bond_vals_3[m] - single_bond_vals[m]) / delta;
            exact = central_dervs[n];
            diff = abs(finite_diff - exact);

            EXPECT_LE(diff, tolerance);
        }
    }
}
