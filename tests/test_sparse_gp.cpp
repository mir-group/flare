#include "gtest/gtest.h"
#include "sparse_gp.h"
#include "descriptor.h"
#include "omp.h"
#include <chrono>

class SparseTest : public ::testing::Test{
    public:
        // structure
        int n_atoms = 4;
        Eigen::MatrixXd cell{3, 3}, cell_2{3, 3};
        std::vector<int> species {0, 1, 0, 1};
        Eigen::MatrixXd positions{n_atoms, 3};
        StructureDescriptor test_struc;
    
        // labels
        Eigen::VectorXd energy{1}, forces{n_atoms * 3}, stresses{6};

        // descriptor
        std::string radial_string = "chebyshev";
        std::string cutoff_string = "cosine";
        std::vector<double> radial_hyps {0, 5};
        std::vector<double> cutoff_hyps;
        std::vector<int> descriptor_settings {2, 3, 3};
        double cutoff = 5;
        std::vector<double> nested_cutoffs {5, 5};
        std::vector<double> many_body_cutoffs {5};
        B2_Calculator desc1;
        std::vector<DescriptorCalculator *> calcs;

        // kernel
        double signal_variance = 1;
        double length_scale = 1;
        double power = 2;
        int descriptor_index = 0;
        DotProductKernel kernel;
        TwoBodyKernel two_body_kernel;
        ThreeBodyKernel three_body_kernel;
        DotProductKernel many_body_kernel;
        std::vector<Kernel *> kernels;

    SparseTest(){
        cell << 100, 0, 0,
                0, 100, 0,
                0, 0, 100;

        positions = Eigen::MatrixXd::Random(n_atoms, 3);
        energy = Eigen::VectorXd::Random(1);
        forces = Eigen::VectorXd::Random(n_atoms * 3);
        stresses = Eigen::VectorXd::Random(6);

        desc1 = B2_Calculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings, 0);
        calcs.push_back(&desc1);

        test_struc = StructureDescriptor(cell, species, positions, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         calcs);
        test_struc.energy = energy;
        test_struc.forces = forces;
        test_struc.stresses = stresses;

        two_body_kernel = TwoBodyKernel(signal_variance, length_scale,
            cutoff_string, cutoff_hyps);
        three_body_kernel = ThreeBodyKernel(signal_variance, length_scale,
            cutoff_string, cutoff_hyps);
        many_body_kernel = DotProductKernel(signal_variance, power,
            descriptor_index);

        kernels = 
            std::vector<Kernel *> {&many_body_kernel};

        // kernels = 
        //     std::vector<Kernel *> {&two_body_kernel, &three_body_kernel};
    }
};

TEST_F(SparseTest, UpdateK){
    double sigma_e = 1;
    double sigma_f = 2;
    double sigma_s = 3;

    SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);
    LocalEnvironment env1 = test_struc.local_environments[0];
    LocalEnvironment env2 = test_struc.local_environments[1];
    sparse_gp.add_sparse_environment(env1);
    sparse_gp.add_sparse_environment(env2);

    sparse_gp.add_training_structure(test_struc);

    test_struc.stresses = Eigen::VectorXd {};
    sparse_gp.add_training_structure(test_struc);

    EXPECT_EQ(sparse_gp.Kuu.rows(), 2);
    EXPECT_EQ(sparse_gp.Kuf_struc.rows(), 2);
    EXPECT_EQ(sparse_gp.Kuf_struc.cols(), 2 + 6 * n_atoms + 6);
    EXPECT_EQ(sparse_gp.y_struc.size(), 2 + 6 * n_atoms + 6);
    EXPECT_EQ(test_struc.forces, sparse_gp.y_struc.segment(1, 3 * n_atoms));

    double kern_val = 0;
    double kern_curr;
    for (int i = 0; i < kernels.size(); i ++){
        kern_curr =  kernels[i] -> env_env(env1, env1);
        kern_val += kern_curr;
    }
    EXPECT_EQ(kern_val, sparse_gp.Kuu(0,0));

    // Add a training environment to the training set.
    env1.compute_neighbor_descriptors();
    Eigen::VectorXd force{3};
    force << 1, 1, 1;
    env1.force = force;
    sparse_gp.add_training_environment(env1);

    sparse_gp.update_alpha();
    EXPECT_EQ(sparse_gp.Kuf.rows(), 2);
    EXPECT_EQ(sparse_gp.Kuf.cols(),
        sparse_gp.Kuf_struc.cols() + sparse_gp.Kuf_env.cols());
    EXPECT_EQ(sparse_gp.y.size(),
        sparse_gp.y_struc.size() + sparse_gp.y_env.size());
    EXPECT_EQ(sparse_gp.noise_matrix.rows(),
        sparse_gp.noise_env.size() + sparse_gp.noise_struc.size());    
}

TEST_F(SparseTest, TrainingEnvironments){
    double sigma_e = 1;
    double sigma_f = 2;
    double sigma_s = 3;

    SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);
    SparseGP sparse_gp_2 = SparseGP(kernels, sigma_e, sigma_f, sigma_s);
    LocalEnvironment env1 = test_struc.local_environments[0];
    LocalEnvironment env2 = test_struc.local_environments[1];

    // Add sparse environments.
    std::vector<LocalEnvironment> sparse_envs {env1, env2};
    sparse_gp.add_sparse_environments(sparse_envs);

    sparse_gp_2.add_sparse_environment(env1);
    sparse_gp_2.add_sparse_environment(env2);

    // Add a training environment to the training set.
    env1.compute_neighbor_descriptors();
    env2.compute_neighbor_descriptors();
    Eigen::VectorXd force{3};
    force << 1, 1, 1;
    env1.force = force;
    env2.force = force;
    std::vector<LocalEnvironment> envs {env1, env2};

    sparse_gp.add_training_environments(envs);

    sparse_gp_2.add_training_environment(env1);
    sparse_gp_2.add_training_environment(env2);

    sparse_gp.update_alpha();
    sparse_gp_2.update_alpha();

    // Check that Kufs match.
    for (int i = 0; i < sparse_gp.Kuf.rows(); i++){
        for (int j = 0; j < sparse_gp.Kuf.cols(); j++){
            EXPECT_EQ(sparse_gp.Kuf(i, j), sparse_gp_2.Kuf(i, j));
        }
    }

    // Check that Kuus match.
    for (int i = 0; i < sparse_gp.Kuu.rows(); i++){
        for (int j = 0; j < sparse_gp.Kuu.cols(); j++){
            EXPECT_EQ(sparse_gp.Kuu(i, j), sparse_gp_2.Kuu(i, j));
        }
    }

    // Check that ys and noises match.
    for (int i = 0; i < sparse_gp.y.size(); i++){
            EXPECT_EQ(sparse_gp.y(i), sparse_gp_2.y(i));
            EXPECT_EQ(sparse_gp.noise(i), sparse_gp_2.noise(i));
    }
}

TEST_F(SparseTest, Predict){

    double sigma_e = 0.1;
    double sigma_f = 0.01;
    double sigma_s = 1;

    SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);
    LocalEnvironment env1 = test_struc.local_environments[0];
    LocalEnvironment env2 = test_struc.local_environments[1];
    sparse_gp.add_sparse_environment(env1);
    sparse_gp.add_sparse_environment(env2);

    test_struc.stresses = Eigen::VectorXd {};

    sparse_gp.add_training_structure(test_struc);
    sparse_gp.update_alpha();
    Eigen::VectorXd pred_vals, pred_env;

    // Predict on structure.
    pred_vals = sparse_gp.predict(test_struc);

    // Predict on local environment
    env1.compute_descriptors_and_gradients();
    env1.compute_neighbor_descriptors();
    pred_env = sparse_gp.predict_force(env1);

    // Check that the prediction methods agree.
    EXPECT_EQ(pred_vals(1), pred_env(0));

}


TEST_F(SparseTest, TestBeta){
    double sigma_e = 1;
    double sigma_f = 2;
    double sigma_s = 3;

    SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

    LocalEnvironment env1 = test_struc.local_environments[0];
    LocalEnvironment env2 = test_struc.local_environments[1];
    LocalEnvironment env3 = test_struc.local_environments[2];
    LocalEnvironment env4 = test_struc.local_environments[3];
    sparse_gp.add_sparse_environment(env1);
    sparse_gp.add_sparse_environment(env2);
    sparse_gp.add_sparse_environment(env3);
    sparse_gp.add_sparse_environment(env4);
    sparse_gp.add_training_structure(test_struc);
    // sparse_gp.update_alpha();
    sparse_gp.update_alpha_LLT();

    // Predict local energy with alpha.
    double loc_en = sparse_gp.predict_local_energy(env1);
    std::cout << loc_en << std::endl;

    // Predict local energy with beta.
    sparse_gp.compute_beta(0, 0);
    env1.compute_descriptor_squared();
    std::cout << (sparse_gp.beta.row(0)).dot(env1.descriptor_squared[0]) << std::endl;

    // Write beta to file.
    std::string beta_file = "beta.txt";
    std::string contributor = 
        "Jonathan Vandermause, jonathan_vandermause@g.harvard.edu";
    int descriptor_index = 0;
    sparse_gp.write_beta(beta_file, contributor, descriptor_index);

    // // Check that memory profile works
    // sparse_gp.memory_profile();
    // std::cout << sparse_gp.model_size << std::endl;
    // std::cout << sparse_gp.sparse_size << std::endl;
    // std::cout << sparse_gp.training_size << std::endl;
    // std::cout << sizeof(sparse_gp.sparse_environments[0]) << std::endl;
    // std::cout << sparse_gp.Kuu_size << std::endl;

    // std::cout << sparse_gp.Kuu.size() << std::endl;
    // std::cout << sizeof(sparse_gp.Kuu(0,0)) << std::endl;

}

TEST(CountThreads, CountThreads){
    int test = omp_get_max_threads();
    std::cout << test << std::endl;
    EXPECT_EQ(1, 1);
}

// TEST_F(SparseTest, ThreeBodyGrid){
//     double sigma_e = 0.1;
//     double sigma_f = 0.01;
//     double sigma_s = 1;

//     double min_dist = 0.1;
//     double max_dist = 3;
//     double cutoff = 3;
//     int n_species = 5;
//     int n_dist = 3;
//     int n_angle = 3;

//     SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);
//     sparse_gp.three_body_grid(min_dist, max_dist, cutoff, n_species,
//         n_dist, n_angle);

//     // std::cout << sparse_gp.sparse_environments.size() << std::endl;
//     // std::cout << sparse_gp.sparse_environments[1].xs.size() << std::endl;
// }
