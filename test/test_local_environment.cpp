#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

TEST(EnvironmentTest, SweepTest){
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
    
    // Create local environment.
    int atom = 0;
    double cutoff = 3;
    LocalEnvironment test_env =
        LocalEnvironment(test_struc, atom, cutoff);

    EXPECT_EQ(ceil(cutoff / test_struc.max_cutoff), test_env.sweep);

    // Check that the number of atoms in the local environment is correct.
    std::vector<int> env_ind, env_spec, unique_ind;
    std::vector<double> rs, xs, ys, zs;
    int sweep_val = test_env.sweep + 3;
    LocalEnvironment :: compute_environment(
        test_struc, atom, cutoff, sweep_val, env_ind, env_spec, unique_ind,
        rs, xs, ys, zs);
    int expanded_count = rs.size();
    EXPECT_EQ(test_env.rs.size(), expanded_count);

    EXPECT_EQ(test_env.neighbor_list.size(), 5);
}
