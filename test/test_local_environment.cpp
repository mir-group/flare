#include "gtest/gtest.h"
#include "local_environment.h"
#include "structure.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#define THRESHOLD 1e-8

class EnvironmentTest : public :: testing :: Test{
    public:
        Eigen::MatrixXd cell{3, 3};
        std::vector<int> species {0, 1, 2, 3, 4};
        Eigen::MatrixXd positions{5, 3};
        B2_Calculator desc1;
        StructureDataset test_struc;
        int atom;
        LocalEnvironment test_env;

        std::string radial_string = "chebyshev";
        std::string cutoff_string = "cosine";
        std::vector<double> radial_hyps {0, 5};
        std::vector<double> cutoff_hyps;
        std::vector<int> descriptor_settings {5, 5, 5};
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

        desc1 = B2_Calculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings);
        test_struc = StructureDataset(cell, species, positions, desc1, cutoff);

        atom = 0;
        test_env = LocalEnvironment(test_struc, atom, cutoff, &desc1);
    }
};

TEST_F(EnvironmentTest, SweepTest){
       Eigen::MatrixXd cell(3, 3);
       Eigen::MatrixXd positions(5, 3);

    EXPECT_EQ(ceil(cutoff / test_struc.max_cutoff), test_env.sweep);

    // Check that the number of atoms in the local environment is correct.
    std::vector<int> env_ind, env_spec, unique_ind;
    std::vector<double> rs, xs, ys, zs, xrel, yrel, zrel;
    int sweep_val = test_env.sweep + 3;
    LocalEnvironment :: compute_environment(
        test_struc, test_env.noa, atom, cutoff, sweep_val, env_ind, env_spec,
        unique_ind, rs, xs, ys, zs, xrel, yrel, zrel);
    int expanded_count = rs.size();
    EXPECT_EQ(test_env.rs.size(), expanded_count);
    EXPECT_EQ(test_env.neighbor_list.size(), 5);

    // Check that the relative coordinates are computed correctly.
    for (int i = 0; i < test_env.rs.size(); i ++){
        EXPECT_EQ(test_env.xs[i] / test_env.rs[i], test_env.xrel[i]);
        EXPECT_EQ(test_env.ys[i] / test_env.rs[i], test_env.yrel[i]);
        EXPECT_EQ(test_env.zs[i] / test_env.rs[i], test_env.zrel[i]);
    }
}

TEST_F(EnvironmentTest, DotTest){
    // Calculate the descriptor norm the old fashioned way.
    double norm_val = 0;
    double val_curr;
    int no_desc = test_env.descriptor_vals.rows();

    for (int i = 0; i < no_desc; i++){
        val_curr = test_env.descriptor_vals(i);
        norm_val += val_curr * val_curr;
    }
    norm_val = sqrt(norm_val);
    EXPECT_NEAR(norm_val, test_env.descriptor_norm, THRESHOLD);
}

// TEST_F(EnvironmentTest, NestedTest){
//     NestedEnvironment nest =  NestedEnvironment(test_struc, 0, cutoff, 3, 2, 1);
//     std::cout << nest.three_body_indices.size() << std::endl;
//     std::cout << nest.cross_bond_dists.size() << std::endl;
// }
