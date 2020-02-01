#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

TEST(StructureTest, TestWrapped){
    // Check that the wrapped coordinates are equivalent to Cartesian coordinates up to lattice translations.

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
    
    // Take positions minus wrapped positions.
    Eigen::MatrixXd wrap_diff =
        test_struc.positions - test_struc.wrapped_positions;
    
    Eigen::MatrixXd wrap_rel =
        (wrap_diff * test_struc.cell_transpose) *
        test_struc.cell_dot_inverse;

    // Check that it maps to the origin under lattice translations.
    Eigen::MatrixXd check_lat =
        wrap_rel.array().round() - wrap_rel.array();

    for (int i = 0; i < 5; i ++){
        for (int j = 0; j < 3; j ++){
            EXPECT_LE(abs(check_lat(i,j)), 1e-10);
        }
    }
}

TEST(StructureTest, StructureDescriptor){
    Eigen::MatrixXd cell(3, 3);
    Eigen::MatrixXd positions(5, 3);

    // Create arbitrary structure.
    cell << 10, 0, 0,
            0, 10, 0,
            0, 0, 10;

    std::vector<int> species {0, 0, 0, 1, 1};

    positions << 1.2, 0.7, 2.3,
                 3.1, 2.5, 8.9,
                 -1.8, -5.8, 3.0,
                 0.2, 1.1, 2.1,
                 3.2, 1.1, 3.3;

    std::string radial_string = "chebyshev";
    std::string cutoff_string = "quadratic";
    std::vector<double> radial_hyps = {0, 5};
    std::vector<double> cutoff_hyps;
    std::vector<int> descriptor_settings {2, 10, 10};

    DescriptorCalculator desc1 = 
        DescriptorCalculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings);
    double cutoff = 5;

    StructureDescriptor test_struc = 
        StructureDescriptor(cell, species, positions, desc1, cutoff);
    
    // Check that structure descriptors match environment descriptors.
    LocalEnvironment env;
    for (int i = 0; i < test_struc.species.size(); i ++){
        env = LocalEnvironment(test_struc, i, cutoff);
        desc1.compute_B2(env);
        for (int j = 0; j < desc1.descriptor_vals.size(); j ++){
            EXPECT_EQ(desc1.descriptor_vals[j],
                      test_struc.descriptor_vals[i][j]);
            for (int k = 0; k < test_struc.species.size(); k ++){
                EXPECT_EQ(desc1.descriptor_force_dervs(k, j),
                          test_struc.descriptor_force_dervs[i](k, j));
            }
        }
    }
}
