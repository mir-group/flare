#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class DescriptorTest : public ::testing::Test{
    protected:

    double delta = 1e-8;
    int noa = 5;
    int nos = 5;

    Eigen::MatrixXd cell {3, 3}, cell_2 {3, 3}, Rx {3, 3}, Ry {3, 3},
        Rz {3, 3}, R {3, 3};
    std::vector<int> species {0, 2, 1, 3, 4};
    Eigen::MatrixXd positions_1 {noa, 3}, positions_2 {noa, 3};

    Structure struc1, struc2;
    LocalEnvironment env1, env2;
    DescriptorCalculator desc1, desc2;

    double rcut = 3;
    
    double xrot = 1.28;
    double yrot = -3.21;
    double zrot = 0.42;
 
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
    std::vector<double> radial_hyps = {first_gauss, final_gauss};
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>) = equispaced_gaussians;

    // Initialize matrices.
    int no_descriptors = nos * N * number_of_harmonics;
    std::vector<double> single_bond_vals, single_bond_vals_2;

    Eigen::MatrixXd force_dervs, force_dervs_2, stress_dervs, stress_dervs_2;

    // Set up descriptor calculator.
    std::string radial_string = "chebyshev";
    std::string cutoff_string = "quadratic";

    DescriptorTest(){
        // Define rotation matrices.
        Rx << 1, 0, 0,
              0, cos(xrot), -sin(xrot),
              0, sin(xrot), cos(xrot);
        Ry << cos(yrot), 0, sin(yrot),
              0, 1, 0,
              -sin(yrot), 0, cos(yrot);
        Rz << cos(zrot), -sin(zrot), 0,
              sin(zrot), cos(zrot), 0,
              0, 0, 1;
        R = Rx * Ry * Rz;

        // Create arbitrary structure.
        cell << 1.3, 0.5, 0.8,
               -1.2, 1, 0.73,
               -0.8, 0.1, 0.9;
        cell_2 = cell * R.transpose();

        positions_1 << 1.2, 0.7, 2.3,
                     3.1, 2.5, 8.9,
                     -1.8, -5.8, 3.0,
                     0.2, 1.1, 2.1,
                     3.2, 1.1, 3.3;
        positions_2 = positions_1 * R.transpose();

        struc1 = Structure(cell, species, positions_1);
        struc2 = Structure(cell_2, species, positions_2);

        env1 = LocalEnvironment(struc1, 0, rcut);
        env2 = LocalEnvironment(struc2, 0, rcut);

        single_bond_vals = std::vector<double> (no_descriptors, 0);
        force_dervs = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
        stress_dervs = Eigen::MatrixXd::Zero(6, no_descriptors);

        // Create descriptor calculator.
        desc1 = DescriptorCalculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, nos, N, lmax);
        desc2 = DescriptorCalculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, nos, N, lmax);
    }

};

TEST_F(DescriptorTest, RotationTest){
    desc1.compute_B1(env1);
    desc2.compute_B1(env2);

    // Check that B1 is rotationally invariant.
    int no_desc = desc1.descriptor_vals.size();
    double d1, d2, diff;
    double tol = 1e-10;
    for (int n = 0; n < no_desc; n ++){
        d1 = desc1.descriptor_vals[n];
        d2 = desc2.descriptor_vals[n];
        diff = d1 - d2;
        EXPECT_LE(abs(diff), tol);
    }
}
