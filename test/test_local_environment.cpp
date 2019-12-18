#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

TEST(EnvironmentTest, EnvironmentTest){
    Eigen::MatrixXd cell(3, 3);
    Eigen::MatrixXd positions(2, 3);

    // Create arbitrary structure.
    double alat = 1;
    double root = sqrt(2)/2;
    cell << alat, 0, 0,
            root * alat, root * alat, 0,
            0, 0, alat;

    std::vector<int> species {0, 1};

    double cutoff = alat * root;

    positions << 0, 0, 0,
                 2 * alat * root, alat * root - 0.0001, 0;

    Structure test_struc = 
        Structure(cell, species, positions);

    double max_cutoff = test_struc.get_max_cutoff();
    std::cout << max_cutoff << std::endl;
    
    int atom = 0;
    int sweep = 5;
    LocalEnvironment test_env =
        LocalEnvironment(test_struc, atom, cutoff, sweep);

    std::cout << test_env.environment_species.size() << std::endl;
    // std::cout << test_env.xs[0] << std::endl;
    // std::cout << test_env.ys[0] << std::endl;
    // std::cout << test_env.zs[0] << std::endl;
    // std::cout << test_env.xs[1] << std::endl;
    // std::cout << test_env.ys[1] << std::endl;
    // std::cout << test_env.zs[1] << std::endl;
}
