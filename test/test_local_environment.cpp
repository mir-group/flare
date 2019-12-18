#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <Eigen/Dense>

TEST(EnvironmentTest, EnvironmentTest){
    Eigen::MatrixXd cell(3, 3);
    Eigen::MatrixXd positions(5, 3);

    // Create arbitrary structure.
    cell << 1.3, 0.5, 0.8,
            -1.2, 1, 0.73,
            -0.8, 0.1, 0.9;

    std::vector<int> species {1, 2, 3, 4, 5};

    positions << 1.2, 0.7, 2.3,
                 3.1, 2.5, 8.9,
                 -1.8, -5.8, 3.0,
                 0.2, 1.1, 2.1,
                 3.2, 1.1, 3.3;

    Structure test_struc = 
        Structure(cell, species, positions);
    
    LocalEnvironment test_env =
        LocalEnvironment(test_struc, 1, 4.2, 2);

    std::cout << test_env.environment_species[200];
}
