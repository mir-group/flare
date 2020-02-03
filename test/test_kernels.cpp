#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#define THRESHOLD 1e-8

class KernelTest : public ::testing::Test{
    public:
        // structure
        Eigen::MatrixXd cell{3, 3};
        std::vector<int> species {0, 0, 0, 0, 0};
        Eigen::MatrixXd positions{5, 3};
        StructureDescriptor test_struc;

        // descriptor
        std::string radial_string = "chebyshev";
        std::string cutoff_string = "cosine";
        std::vector<double> radial_hyps {0, 5};
        std::vector<double> cutoff_hyps;
        std::vector<int> descriptor_settings {1, 10, 10};
        double cutoff = 5;
        DescriptorCalculator desc1;

        // kernel
        double signal_variance = 1;
        double power = 2;
        DotProductKernel kernel;

    KernelTest(){
        cell << 10, 0, 0,
                0, 10, 0,
                0, 0, 10;

        positions << 1.2, 0.7, 2.3,
                     3.1, 2.5, 8.9,
                    -1.8, -5.8, 3.0,
                     0.2, 1.1, 2.1,
                     3.2, 1.1, 3.3;

        desc1 = DescriptorCalculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings);
        test_struc = StructureDescriptor(cell, species, positions, desc1,
                                         cutoff);

        kernel = DotProductKernel(signal_variance, power);
    }
};

TEST_F(KernelTest, NormTest){
    LocalEnvironmentDescriptor env1 = test_struc.environment_descriptors[0];
    LocalEnvironmentDescriptor env2 = test_struc.environment_descriptors[1];
    double kern_val = kernel.env_env(env1, env1);
    EXPECT_NEAR(kern_val, 1, THRESHOLD);
}

TEST_F(KernelTest, StrucTest){
    LocalEnvironmentDescriptor env1 = test_struc.environment_descriptors[0];
    Eigen::VectorXd kern_vec = kernel.env_struc(env1, test_struc);

    std::cout << kern_vec << std::endl;
}
