#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

TEST(EnvironmentTest, EnvironmentTest){
    Eigen::MatrixXd cell(3, 3);
    Eigen::MatrixXd positions(2, 3);

    // Create arbitrary structure.
    double alat = 10;
    double root = sqrt(2)/2;
    cell << alat, 0, 0,
            root * alat, root * alat, 0,
            0, 0, alat;
    std::vector<int> species {0, 1};
    positions << 0, 0, 0,
                 2 * alat * root, alat * root - 0.0001, 0;
    Structure test_struc = 
        Structure(cell, species, positions);
    
    // Create local environment.
    int atom = 0;
    double cutoff = alat * 3;
    LocalEnvironment test_env =
        LocalEnvironment(test_struc, atom, cutoff);

    EXPECT_EQ(ceil(cutoff / test_struc.max_cutoff), test_env.sweep);
}
