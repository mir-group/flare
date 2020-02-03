#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

class EnvironmentTest : public :: testing :: Test{
    public:
        Eigen::MatrixXd cell{3, 3};
        std::vector<int> species {0, 1, 2, 3, 4};
        Eigen::MatrixXd positions{5, 3};
        DescriptorCalculator desc1;
        StructureDataset test_struc;

        std::string radial_string = "chebyshev";
        std::string cutoff_string = "cosine";
        std::vector<double> radial_hyps {0, 5};
        std::vector<double> cutoff_hyps;
        std::vector<int> descriptor_settings {5, 10, 10};
        double cutoff = 3;

    EnvironmentTest(){
        cell << 1.3, 0.5, 0.8,
               -1.2, 1, 0.73,
               -0.8, 0.1, 0.9;
    
        positions << 1.2, 0.7, 2.3,
                     3.1, 2.5, 8.9,
                    -1.8, -5.8, 3.0,
                     0.2, 1.1, 2.1,
                     3.2, 1.1, 3.3;

        desc1 = DescriptorCalculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings);
        test_struc = StructureDataset(cell, species, positions, desc1, cutoff);
    }
};

TEST_F(EnvironmentTest, SweepTest){
       Eigen::MatrixXd cell(3, 3);
       Eigen::MatrixXd positions(5, 3);
    
    // Create local environment.
    int atom = 0;
    LocalEnvironment test_env =
        LocalEnvironment(test_struc, atom, cutoff);

    EXPECT_EQ(ceil(cutoff / test_struc.max_cutoff), test_env.sweep);

    // Check that the number of atoms in the local environment is correct.
    std::vector<int> env_ind, env_spec, unique_ind;
    std::vector<double> rs, xs, ys, zs;
    int sweep_val = test_env.sweep + 3;
    LocalEnvironment :: compute_environment(
        test_struc, test_env.noa, atom, cutoff, sweep_val, env_ind, env_spec,
        unique_ind, rs, xs, ys, zs);
    int expanded_count = rs.size();
    EXPECT_EQ(test_env.rs.size(), expanded_count);

    EXPECT_EQ(test_env.neighbor_list.size(), 5);
}
