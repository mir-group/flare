#include "gtest/gtest.h"
#include "sparse_gp.h"
#include "descriptor.h"

class SparseTest : public ::testing::Test{
    public:
        // structure
        int n_atoms = 2;
        Eigen::MatrixXd cell{3, 3}, cell_2{3, 3};
        std::vector<int> species {0, 1};
        Eigen::MatrixXd positions{n_atoms, 3}, positions_2{n_atoms, 3},
            positions_3{n_atoms, 3};
        StructureDescriptor test_struc, test_struc_2, test_struc_3;
    
        // labels
        Eigen::VectorXd energy{1}, forces{n_atoms * 3}, stresses{6};

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
                      0.33045915, 0.40010388, 0.59849816;
        
        positions_2 << 0.69955637, -0.41619112, -0.51725003,
                       0.43189622, 0.88548458, -0.74495343;
        
        energy = Eigen::VectorXd::Random(1);
        forces = Eigen::VectorXd::Random(n_atoms * 3);
        stresses = Eigen::VectorXd::Random(6);

        desc1 = B2_Calculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings);

        test_struc = StructureDescriptor(cell, species, positions, cutoff,
                                         nested_cutoffs, &desc1);
        test_struc.energy = energy;
        test_struc.forces = forces;
        test_struc.stresses = stresses;

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

TEST_F(SparseTest, UpdateK){
    SparseGP sparse_gp = SparseGP(kernels);
    LocalEnvironment env1 = test_struc.local_environments[0];
    LocalEnvironment env2 = test_struc.local_environments[1];
    sparse_gp.add_sparse_environment(env1);
    sparse_gp.add_sparse_environment(env2);

    sparse_gp.add_training_structure(test_struc);

    test_struc.stresses = Eigen::VectorXd {};
    sparse_gp.add_training_structure(test_struc);

    EXPECT_EQ(sparse_gp.Kuu.rows(), 2);
    EXPECT_EQ(sparse_gp.Kuf.rows(), 2);
    EXPECT_EQ(sparse_gp.Kuf.cols(), 2 + 6 * n_atoms + 6);
    EXPECT_EQ(sparse_gp.y.size(), 2 + 6 * n_atoms + 6);
    EXPECT_EQ(test_struc.forces, sparse_gp.y.segment(1, 3 * n_atoms));

    double kern_val = 0;
    double kern_curr;
    for (int i = 0; i < kernels.size(); i ++){
        kern_curr =  kernels[i] -> env_env(env1, env1);
        kern_val += kern_curr;
    }
    EXPECT_EQ(kern_val, sparse_gp.Kuu(0,0));
}
