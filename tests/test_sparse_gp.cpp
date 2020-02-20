#include "gtest/gtest.h"
#include "sparse_gp.h"
#include "descriptor.h"

class SparseTest : public ::testing::Test{
    public:
        // structure
        Eigen::MatrixXd cell{3, 3}, cell_2{3, 3};
        std::vector<int> species {0, 1, 0, 1, 0};
        Eigen::MatrixXd positions{5, 3}, positions_2{5, 3}, positions_3{5, 3};
        StructureDescriptor test_struc, test_struc_2, test_struc_3;

        // descriptor
        std::string radial_string = "chebyshev";
        std::string cutoff_string = "cosine";
        std::vector<double> radial_hyps {0, 5};
        std::vector<double> cutoff_hyps;
        std::vector<int> descriptor_settings {2, 5, 5};
        double cutoff = 5;
        std::vector<double> nested_cutoffs {5, 5};
        B2_Calculator desc1;

        // kernel
        double signal_variance = 2;
        double length_scale = 1;
        double power = 2;
        std::vector<double> kernel_hyperparameters {signal_variance, power};
        DotProductKernel kernel;
        TwoBodyKernel two_body_kernel;
        ThreeBodyKernel three_body_kernel;
        DotProductKernel many_body_kernel;
        std::vector<Kernel *> kernels;

    SparseTest(){
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

        desc1 = B2_Calculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings);

        test_struc = StructureDescriptor(cell, species, positions, cutoff,
                                         nested_cutoffs, &desc1);

        two_body_kernel = TwoBodyKernel(length_scale, cutoff_string,
            cutoff_hyps);
        three_body_kernel = ThreeBodyKernel(length_scale, cutoff_string,
            cutoff_hyps);
        many_body_kernel = DotProductKernel(kernel_hyperparameters);

        kernels = 
            std::vector<Kernel *> {&two_body_kernel, &three_body_kernel,
                &many_body_kernel};
    }
};

TEST_F(SparseTest, AddSparseTest){
    SparseGP sparse_gp = SparseGP(kernels);
    LocalEnvironment env1 = test_struc.environment_descriptors[0];
    LocalEnvironment env2 = test_struc.environment_descriptors[1];
    LocalEnvironment env3 = test_struc.environment_descriptors[2];
    sparse_gp.add_sparse_environment(env1);
    sparse_gp.add_sparse_environment(env2);
    sparse_gp.add_sparse_environment(env3);

    test_struc.energy = std::vector<double> {10};
    test_struc.forces = std::vector<double> (15, 10);
    test_struc.stresses = std::vector<double> {1, 2, 3, 4, 5, 6};

    sparse_gp.add_training_structure(test_struc);
    // sparse_gp.add_training_structure(test_struc);
    // std::cout << sparse_gp.Kuf << std::endl;
    // std::cout << sparse_gp.training_structures.size() << std::endl;
}
