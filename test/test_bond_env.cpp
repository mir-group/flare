#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class BondEnv : public ::testing::Test{
    protected:

    double delta = 1e-8;
    int noa = 5;
    int nos = 5;

    Eigen::MatrixXd cell {3, 3};
    std::vector<int> species {0, 2, 1, 3, 4};
    Eigen::MatrixXd positions_1 {noa, 3}, positions_2 {noa, 3},
                    positions_3 {noa, 3};

    Structure struc1, struc2, struc3;
    LocalEnvironment env1, env2, env3;

    double rcut = 3;
 
    // Prepare cutoff.
    std::vector<double> cutoff_hyps;
    void (*cutoff_function)(double *, double, double, std::vector<double>) =
        cos_cutoff;

    // Prepare spherical harmonics.
    int lmax = 10;
    int number_of_harmonics = (lmax + 1) * (lmax + 1);

    // Prepare radial basis set.
    double sigma = 1;
    double first_gauss = 0;
    double final_gauss = 3;
    int N = 10;
    std::vector<double> radial_hyps = {sigma, first_gauss, final_gauss};
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>) = equispaced_gaussians;

    // Initialize matrices.
    int no_descriptors = nos * N * number_of_harmonics;
    std::vector<double> single_bond_vals, single_bond_vals_2,
        single_bond_vals_3;

    Eigen::MatrixXd force_dervs, force_dervs_2, force_dervs_3,
        stress_dervs, stress_dervs_2, stress_dervs_3;

    BondEnv(){
        // Create arbitrary structure.
        cell << 1.3, 0.5, 0.8,
               -1.2, 1, 0.73,
               -0.8, 0.1, 0.9;

        positions_1 << 1.2, 0.7, 2.3,
                     3.1, 2.5, 8.9,
                     -1.8, -5.8, 3.0,
                     0.2, 1.1, 2.1,
                     3.2, 1.1, 3.3;

        struc1 = Structure(cell, species, positions_1);

            // Perturb the x coordinate of the first atom.
        positions_2 = positions_1;
        positions_2(0, 0) += delta;
        struc2 =  Structure(cell, species, positions_2);

        // Perturb the z coordinate of the fifth atom.
        positions_3 = positions_1;
        positions_3(4, 2) += delta;
        struc3 = Structure(cell, species, positions_3);

        // Create local environments.
        env1 = LocalEnvironment(struc1, 0, rcut);
        env2 = LocalEnvironment(struc2, 0, rcut);
        env3 = LocalEnvironment(struc3, 0, rcut);

        // Initialize matrices.
        single_bond_vals = std::vector<double> (no_descriptors, 0);
        single_bond_vals_2 = std::vector<double> (no_descriptors, 0);
        single_bond_vals_3 = std::vector<double> (no_descriptors, 0);

        force_dervs = force_dervs_2 = force_dervs_3 =
            Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
        stress_dervs = stress_dervs_2 = stress_dervs_3 = 
            Eigen::MatrixXd::Zero(6, no_descriptors);

    }

};

TEST_F(BondEnv, ForceTest){
    int test = 3000;

    single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
        basis_function, cutoff_function, env1, rcut, N, lmax,
        radial_hyps, cutoff_hyps);

    single_bond_sum_env(single_bond_vals_2, force_dervs_2, stress_dervs_2,
        basis_function, cutoff_function, env2, rcut, N, lmax,
        radial_hyps, cutoff_hyps);

    single_bond_sum_env(single_bond_vals_3, force_dervs_3, stress_dervs_3,
        basis_function, cutoff_function, env3, rcut, N, lmax,
        radial_hyps, cutoff_hyps);

    double finite_diff, exact, diff;
    double tolerance = 1e-6;
    
    for (int n = 0; n < single_bond_vals.size(); n ++){
        // Check central derivative.
        finite_diff = 
            (single_bond_vals_2[n] - single_bond_vals[n]) / delta;
        exact = force_dervs(0, n);
        diff = abs(finite_diff - exact);
        EXPECT_LE(diff, tolerance);

        // Check environment derivative.
        finite_diff = 
            (single_bond_vals_3[n] - single_bond_vals[n]) / delta;
        exact = force_dervs(14, n);
        diff = abs(finite_diff - exact);
        EXPECT_LE(diff, tolerance);

    }
}

// TODO: Test stress calculation.
