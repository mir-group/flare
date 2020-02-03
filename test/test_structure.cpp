#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class StructureTest : public ::testing::Test{
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

    StructureTest(){
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

TEST_F(StructureTest, TestWrapped){
    // Check that the wrapped coordinates are equivalent to Cartesian coordinates up to lattice translations.

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

TEST_F(StructureTest, StructureDescriptor){
    // Check that structure descriptors match environment descriptors.
    LocalEnvironment env;
    for (int i = 0; i < test_struc.species.size(); i ++){
        env = LocalEnvironment(test_struc, i, cutoff);
        desc1.compute_B2(env);
        for (int j = 0; j < desc1.descriptor_vals.size(); j ++){
            EXPECT_EQ(desc1.descriptor_vals(j),
                      test_struc.descriptor_vals[i](j));
            for (int k = 0; k < test_struc.species.size(); k ++){
                EXPECT_EQ(desc1.descriptor_force_dervs(k, j),
                          test_struc.descriptor_force_dervs[i](k, j));
            }
        }
    }
}

TEST_F(StructureTest, StructureDataset){
    // Check that label vectors are empty by default.
    EXPECT_EQ(test_struc.energy.size(), 0);
    EXPECT_EQ(test_struc.force_components.size(), 0);
    EXPECT_EQ(test_struc.stress_components.size(), 0);

    // Check that EFS labels are set correctly.
    std::vector<double> energy {2.0};
    std::vector<double> force_components {1, 2, 3};
    std::vector<double> stress_components {4, 5, 6};

    test_struc = StructureDataset(cell, species, positions, desc1, cutoff,
                                  energy, force_components,
                                  stress_components);
    
    for (int i = 0; i < energy.size(); i ++){
        EXPECT_EQ(energy[i], test_struc.energy[i]);
    }

    for (int i = 0; i < force_components.size(); i ++){
        EXPECT_EQ(force_components[i], test_struc.force_components[i]);
    }

    for (int i = 0; i < stress_components.size(); i ++){
        EXPECT_EQ(stress_components[i], test_struc.stress_components[i]);
    }
}
