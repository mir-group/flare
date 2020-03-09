#include "gtest/gtest.h"
#include "single_bond.h"
#include "local_environment.h"
#include "structure.h"
#include "cutoffs.h"
#include "radial.h"
#include "descriptor.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class BondEnv : public ::testing::Test{
    protected:

    double delta = 1e-8;
    int noa = 5;
    int nos = 5;

    Eigen::MatrixXd cell {3, 3}, cell_2 {3, 3};
    std::vector<int> species {0, 2, 1, 3, 4};
    Eigen::MatrixXd positions_1 {noa, 3}, positions_2 {noa, 3};

    Structure struc1, struc2;
    LocalEnvironment env1, env2;

    double rcut = 3;
    std::vector<double> many_body_cutoffs {rcut};
 
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
    Eigen::VectorXd single_bond_vals, single_bond_vals_2;

    Eigen::MatrixXd force_dervs, force_dervs_2, stress_dervs, stress_dervs_2;

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
        env1 = LocalEnvironment(struc1, 0, rcut);
        env1.many_body_cutoffs = many_body_cutoffs;
        env1.compute_indices();

        single_bond_vals =Eigen::VectorXd::Zero(no_descriptors);
        force_dervs = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
        stress_dervs = Eigen::MatrixXd::Zero(6, no_descriptors);
    }

};

TEST_F(BondEnv, CentTest){
    single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
        basis_function, cutoff_function, env1, 0, N, lmax,
        radial_hyps, cutoff_hyps);

    // Perturb the coordinates of the central atom.
    for (int m = 0; m < 3; m ++){
        positions_2 = positions_1;
        positions_2(0, m) += delta;
        struc2 =  Structure(cell, species, positions_2);
        env2 = LocalEnvironment(struc2, 0, rcut);
        env2.many_body_cutoffs = many_body_cutoffs;
        env2.compute_indices();

        // Initialize matrices.
        single_bond_vals_2 = Eigen::VectorXd::Zero(no_descriptors);
        force_dervs_2 = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
        stress_dervs_2 = Eigen::MatrixXd::Zero(6, no_descriptors);

        single_bond_sum_env(single_bond_vals_2, force_dervs_2, stress_dervs_2,
            basis_function, cutoff_function, env2, 0, N, lmax,
            radial_hyps, cutoff_hyps);

        double finite_diff, exact, diff;
        double tolerance = 1e-6;

        // Check derivatives.
        for (int n = 0; n < single_bond_vals.rows(); n ++){
            finite_diff = 
                (single_bond_vals_2[n] - single_bond_vals[n]) / delta;
            exact = force_dervs(m, n);
            diff = abs(finite_diff - exact);
            EXPECT_LE(diff, tolerance);
        }
    }
}

TEST_F(BondEnv, EnvTest){
    single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
        basis_function, cutoff_function, env1, 0, N, lmax,
        radial_hyps, cutoff_hyps);

    double finite_diff, exact, diff;
    double tolerance = 1e-5;

    // Perturb the coordinates of the environment atoms.
    for (int p = 1; p < noa; p ++){
        for (int m = 0; m < 3; m ++){
            positions_2 = positions_1;
            positions_2(p, m) += delta;
            struc2 =  Structure(cell, species, positions_2);
            env2 = LocalEnvironment(struc2, 0, rcut);
            env2.many_body_cutoffs = many_body_cutoffs;
            env2.compute_indices();

            // Initialize matrices.
            single_bond_vals_2 = Eigen::VectorXd::Zero(no_descriptors);
            force_dervs_2 = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
            stress_dervs_2 = Eigen::MatrixXd::Zero(6, no_descriptors);

            single_bond_sum_env(single_bond_vals_2, force_dervs_2,
                stress_dervs_2, basis_function, cutoff_function, env2, 0,
                N, lmax, radial_hyps, cutoff_hyps);

            // Check derivatives.
            for (int n = 0; n < single_bond_vals.rows(); n ++){
                finite_diff = 
                    (single_bond_vals_2[n] - single_bond_vals[n]) / delta;
                exact = force_dervs(p * 3 + m, n);
                diff = abs(finite_diff - exact);
                EXPECT_LE(diff, tolerance);
            }
        }
    }
}

TEST_F(BondEnv, StressTest){
    int stress_ind = 0;
    double finite_diff, exact, diff;
    double tolerance = 1e-5;

    single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
        basis_function, cutoff_function, env1, 0, N, lmax,
        radial_hyps, cutoff_hyps);

    // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
    for (int m = 0; m < 3; m ++){
        for (int n = m; n < 3; n ++){
            cell_2 = cell;
            positions_2 = positions_1;

            // Perform strain.
            cell_2(0, m) += cell(0, n) * delta;
            cell_2(1, m) += cell(1, n) * delta;
            cell_2(2, m) += cell(2, n) * delta;
            for (int k = 0; k < noa; k ++){
                positions_2(k, m) += positions_1(k, n) * delta;
            }

            struc2 = Structure(cell_2, species, positions_2);
            env2 = LocalEnvironment(struc2, 0, rcut);
            env2.many_body_cutoffs = many_body_cutoffs;
            env2.compute_indices();

            single_bond_vals_2 = Eigen::VectorXd::Zero(no_descriptors);
            force_dervs_2 = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
            stress_dervs_2 = Eigen::MatrixXd::Zero(6, no_descriptors);

            // Calculate descriptors.
            single_bond_sum_env(single_bond_vals_2, force_dervs_2,
                stress_dervs_2, basis_function, cutoff_function, env2, 0,
                N, lmax, radial_hyps, cutoff_hyps);

            // Check stress derivatives.
            for (int p = 0; p < single_bond_vals.rows(); p ++){
                finite_diff = 
                    (single_bond_vals_2[p] - single_bond_vals[p]) / delta;
                exact = stress_dervs(stress_ind, p);
                diff = abs(finite_diff - exact);
                EXPECT_LE(diff, tolerance);
            }

            stress_ind ++;
        }
    }
}
