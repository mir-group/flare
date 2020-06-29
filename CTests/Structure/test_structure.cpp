#include "gtest/gtest.h"
#include "Structure/structure.h"
#include "Environment/local_environment.h"
#include "Descriptor/descriptor.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class StructureTest : public ::testing::Test{
    public:
        Eigen::MatrixXd cell{3, 3};
        std::vector<int> species {0, 1, 2, 3, 4};
        Eigen::MatrixXd positions{5, 3};
        B2_Calculator desc1;
        std::vector<DescriptorCalculator *> descriptor_calculators;
        StructureDescriptor test_struc;

        std::string radial_string = "chebyshev";
        std::string cutoff_string = "cosine";
        std::vector<double> radial_hyps {0, 5};
        std::vector<double> cutoff_hyps;
        std::vector<int> descriptor_settings {5, 5, 5};
        int descriptor_index = 0;
        double cutoff = 3;
        std::vector<double> many_body_cutoffs {cutoff};

    StructureTest(){
        cell << 4.0, 0.5, 0.8,
               -1.2, 3.9, 0.73,
               -0.8, 0.1, 4.1;
    
        positions << 1.2, 0.7, 2.3,
                     3.1, 2.5, 8.9,
                    -1.8, -5.8, 3.0,
                     0.2, 1.1, 2.1,
                     3.2, 1.1, 3.3;

        desc1 = B2_Calculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings, descriptor_index);
        descriptor_calculators.push_back(&desc1);
        test_struc = StructureDescriptor(cell, species, positions, cutoff, 
            many_body_cutoffs, descriptor_calculators);
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
        env = LocalEnvironment(test_struc, i, cutoff, many_body_cutoffs,
            descriptor_calculators);
        desc1.compute(env);

        for (int j = 0; j < desc1.descriptor_vals.size(); j ++){
            EXPECT_EQ(desc1.descriptor_vals(j),
                      test_struc.local_environments[i]
                        .descriptor_vals[0](j));
            for (int k = 0; k < test_struc.species.size(); k ++){
                EXPECT_EQ(desc1.descriptor_force_dervs(k, j),
                          test_struc.local_environments[i]
                            .descriptor_force_dervs[0](k, j));
            }
        }
    }
}
