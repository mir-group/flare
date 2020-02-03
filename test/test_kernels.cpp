#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#define THRESHOLD 1e-8

class KernelTest : public ::testing::Test{
    public:
        // structure
        Eigen::MatrixXd cell{3, 3}, cell_2{3, 3};
        std::vector<int> species {0, 1, 0, 1, 0};
        Eigen::MatrixXd positions{5, 3}, positions_2{5, 3}, positions_3{5, 3};
        StructureDescriptor test_struc, test_struc_2, test_struc_3;

        // environment
        LocalEnvironmentDescriptor test_env;

        // descriptor
        std::string radial_string = "chebyshev";
        std::string cutoff_string = "cosine";
        std::vector<double> radial_hyps {0, 5};
        std::vector<double> cutoff_hyps;
        std::vector<int> descriptor_settings {2, 10, 10};
        double cutoff = 5;
        DescriptorCalculator desc1;

        // kernel
        double signal_variance = 1;
        double power = 2;
        DotProductKernel kernel;
        Eigen::VectorXd kern_vec;

    KernelTest(){
        cell << 10, 0, 0,
                0, 10, 0,
                0, 0, 10;

        positions << -0.68402216, 0.54343671, -0.52961224,
                      0.33045915, 0.40010388, 0.59849816,
                     -0.92832825, 0.06239221, 0.51601996,
                      0.75120489, -0.39606988, -0.34000017,
                     -0.8242705, -0.73860995, 0.92679555;
        
        positions_2 << 0.69955637, -0.41619112, -0.51725003,
                       0.43189622, 0.88548458, -0.74495343,
                       0.31395126, -0.32179606, -0.35013419,
                       0.08793497, -0.70567732, -0.3811633,
                       0.35585787, -0.87190223, 0.06770428;

        desc1 = DescriptorCalculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings);
        test_struc = StructureDescriptor(cell, species, positions, desc1,
                                         cutoff);

        test_struc_2 = StructureDescriptor(cell, species, positions_2, desc1,
                                           cutoff);
        test_env = test_struc_2.environment_descriptors[0];

        kernel = DotProductKernel(signal_variance, power);

        kern_vec = kernel.env_struc(test_env, test_struc);
    }
};

TEST_F(KernelTest, NormTest){
    LocalEnvironmentDescriptor env1 = test_struc.environment_descriptors[0];
    LocalEnvironmentDescriptor env2 = test_struc.environment_descriptors[1];
    double kern_val = kernel.env_env(env1, env1);
    EXPECT_NEAR(kern_val, 1, THRESHOLD);
}

TEST_F(KernelTest, ForceTest){
    // Perturb the coordinates of the environment atoms.
    double thresh = 1e-6;
    int noa = 5;
    double delta = 1e-8;
    Eigen::VectorXd kern_pert;
    double fin_val, exact_val, abs_diff;
    for (int p = 0; p < noa; p ++){
        for (int m = 0; m < 3; m ++){
            positions_3 = positions;
            positions_3(p, m) += delta;
            test_struc_3 =  StructureDescriptor(cell, species, positions_3,
                                                desc1, cutoff);
            kern_pert = kernel.env_struc(test_env, test_struc_3);
            fin_val = -(kern_pert(0) - kern_vec(0)) / delta;
            exact_val = kern_vec(1 + 3 * p + m);
            abs_diff = abs(fin_val - exact_val); 

            EXPECT_NEAR(abs_diff, 0, thresh);
        }
    }
}

TEST_F(KernelTest, StressTest){
    double thresh = 1e-6;
    int noa = 5;
    double delta = 1e-8;
    Eigen::VectorXd kern_pert;
    double fin_val, exact_val, abs_diff;
    int stress_count = 0;
    // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
    for (int m = 0; m < 3; m ++){
        for (int n = m; n < 3; n ++){
            cell_2 = cell;
            positions_3 = positions;

            // Perform strain.
            cell_2(0, m) += cell(0, n) * delta;
            cell_2(1, m) += cell(1, n) * delta;
            cell_2(2, m) += cell(2, n) * delta;
            for (int k = 0; k < noa; k ++){
                positions_3(k, m) += positions(k, n) * delta;
            }

            test_struc_3 = StructureDescriptor(cell_2, species, positions_3,
                                               desc1, cutoff);

            kern_pert = kernel.env_struc(test_env, test_struc_3);
            fin_val = -(kern_pert(0) - kern_vec(0)) / delta;
            exact_val = kern_vec(1 + 3 * noa + stress_count)
                * test_struc.volume;
            abs_diff = abs(fin_val - exact_val); 

            EXPECT_NEAR(abs_diff, 0, thresh);

            stress_count ++;
        }
    }
}
