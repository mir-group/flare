#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class BondEnv : public ::testing::Test{
    protected:

    double delta = 1e-8;
    Eigen::MatrixXd cell {3, 3};
    std::vector<int> species {0, 1, 2, 3, 4};
    Eigen::MatrixXd positions {5, 3};

    Structure struc1, struc2, struc3;
    LocalEnvironment env1, env2, env3;

    int nos = 5;
    double rcut = 3;
    int atom = 0;
 
    // Prepare cutoff.
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

    // Initialize matrices.
    std::vector<double> single_bond_vals;
    

    BondEnv(){
        cell << 1.3, 0.5, 0.8,
               -1.2, 1, 0.73,
               -0.8, 0.1, 0.9;

        positions << 1.2, 0.7, 2.3,
                     3.1, 2.5, 8.9,
                     -1.8, -5.8, 3.0,
                     0.2, 1.1, 2.1,
                     3.2, 1.1, 3.3;
        
        struc1 = Structure(cell, species, positions);

        // Structure struc1 = Structure(cell, species, positions);
        
    }

};

TEST_F(BondEnv, BondEnv){
    EXPECT_EQ(1, 1);

    std::cout << struc1.cell(0, 0) << std::endl;
}