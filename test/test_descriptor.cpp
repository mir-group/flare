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

    Eigen::MatrixXd cell {3, 3}, cell_2 {3, 3};
    std::vector<int> species {0, 2, 1, 3, 4};
    Eigen::MatrixXd positions_1 {noa, 3}, positions_2 {noa, 3};

    Structure struc1, struc2;
    LocalEnvironment env1, env2;
    DescriptorCalculator desc;

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

        single_bond_vals = std::vector<double> (no_descriptors, 0);
        force_dervs = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
        stress_dervs = Eigen::MatrixXd::Zero(6, no_descriptors);

        // Create descriptor calculator.
        desc = DescriptorCalculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, nos, N, lmax);
    }

};

TEST_F(DescriptorTest, DummyTest){
    desc.compute_B1(env1);

    std::cout << desc.nos << std::endl;
    std::cout << desc.N << std::endl;
    std::cout << desc.lmax << std::endl;
    std::cout << desc.single_bond_vals[242] << std::endl;
    std::cout << desc.descriptor_vals[2] << std::endl;
}
