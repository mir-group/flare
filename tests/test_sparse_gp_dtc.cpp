#include "sparse_gp_dtc.h"
#include "test_structure.h"
#include <chrono>

TEST_F(StructureTest, SparseTest) {
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel);
  SparseGP_DTC sparse_gp = SparseGP_DTC(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
    // test_struc.forces = forces;
  //   test_struc.stresses = stresses;

  Eigen::VectorXd forces_2 = Eigen::VectorXd::Random(n_atoms * 3);
  test_struc_2.forces = forces_2;

  sparse_gp.add_training_structure(test_struc);
  sparse_gp.add_sparse_environments(test_struc);

  std::cout << "Kuu1" << std::endl;
  std::cout << sparse_gp.Kuu << std::endl;
//   sparse_gp.add_training_structure(test_struc_2);
  sparse_gp.add_sparse_environments(test_struc_2);

  std::cout << "Kuu2" << std::endl;
  std::cout << sparse_gp.Kuu << std::endl;

  EXPECT_EQ(sparse_gp.Sigma.rows(), 0);
  EXPECT_EQ(sparse_gp.Kuu_inverse.rows(), 0);

  sparse_gp.update_matrices_QR();
  EXPECT_EQ(sparse_gp.sparse_descriptors[0].n_clusters, sparse_gp.Sigma.rows());
  EXPECT_EQ(sparse_gp.sparse_descriptors[0].n_clusters,
            sparse_gp.Kuu_inverse.rows());

  sparse_gp.predict_on_structure(test_struc);

  // Check that the variances on all quantities are positive.
  int mean_size = test_struc.mean_efs.size();
  for (int i = 0; i < mean_size; i++) {
    EXPECT_GE(test_struc.variance_efs[i], 0);
  }

  // Compute the marginal likelihood.
  sparse_gp.compute_likelihood();

  EXPECT_EQ(sparse_gp.data_fit + sparse_gp.complexity_penalty +
                sparse_gp.constant_term,
            sparse_gp.log_marginal_likelihood);
  double like1 = sparse_gp.log_marginal_likelihood;

    // Check the likelihood function.
    Eigen::VectorXd hyps = sparse_gp.hyperparameters;
    double like2 = sparse_gp.compute_likelihood_gradient(hyps);

    EXPECT_NEAR(like1, like2, 1e-8);
}

TEST_F(StructureTest, SqExpKuf) {
  // Test K_uf grad method of squared exponential kernel.

  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel);
  SparseGP_DTC sparse_gp = SparseGP_DTC(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
//   test_struc.forces = forces;
  test_struc.stresses = stresses;
  test_struc_2.energy = energy;

  sparse_gp.add_training_structure(test_struc);
  sparse_gp.add_sparse_environments(test_struc);
  sparse_gp.add_training_structure(test_struc_2);

  int kernel_index = 0;
  Eigen::VectorXd new_hyps(2);
  new_hyps << 2.0, 0.9;
  std::cout << new_hyps << std::endl;

  std::vector<Eigen::MatrixXd> Kuf_grad = kernel.Kuf_grad(
      sparse_gp.sparse_descriptors[0], sparse_gp.training_structures,
      kernel_index, sparse_gp.Kuf, new_hyps);

//   std::cout << sparse_gp.Kuf << std::endl;
//   std::cout << Kuf_grad[0] << std::endl;
}

TEST_F(StructureTest, LikeGrad) {
  // Check that the DTC likelihood gradient is correctly computed.
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel);
  SparseGP_DTC sparse_gp = SparseGP_DTC(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  test_struc.forces = forces;
  test_struc.stresses = stresses;

  sparse_gp.add_training_structure(test_struc);
  sparse_gp.add_sparse_environments(test_struc);

  EXPECT_EQ(sparse_gp.Sigma.rows(), 0);
  EXPECT_EQ(sparse_gp.Kuu_inverse.rows(), 0);

  sparse_gp.update_matrices_QR();

  // Check the likelihood function.
  Eigen::VectorXd hyps = sparse_gp.hyperparameters;
  sparse_gp.compute_likelihood_gradient(hyps);
  Eigen::VectorXd like_grad = sparse_gp.likelihood_gradient;

  int n_hyps = hyps.size();
  Eigen::VectorXd hyps_up, hyps_down;
  double pert = 1e-4, like_up, like_down, fin_diff;

  for (int i = 0; i < n_hyps; i++) {
    hyps_up = hyps;
    hyps_down = hyps;
    hyps_up(i) += pert;
    hyps_down(i) -= pert;

    like_up = sparse_gp.compute_likelihood_gradient(hyps_up);
    like_down = sparse_gp.compute_likelihood_gradient(hyps_down);

    fin_diff = (like_up - like_down) / (2 * pert);

    EXPECT_NEAR(like_grad(i), fin_diff, 1e-7);
  }
}

TEST_F(StructureTest, Set_Hyps) {
  // Check the reset hyperparameters method.

  int power = 2;
  double sig1 = 1.5, sig2 = 2.0, sig_e_1 = 1.0, sig_e_2 = 2.0, sig_f_1 = 1.5,
    sig_f_2 = 2.5, sig_s_1 = 3.0, sig_s_2 = 3.5;

  NormalizedDotProduct kernel_1 = NormalizedDotProduct(sig1, power);
  NormalizedDotProduct kernel_2 = NormalizedDotProduct(sig2, power);

  std::vector<Kernel *> kernels_1{&kernel_1};
  std::vector<Kernel *> kernels_2{&kernel_2};

  SparseGP_DTC sparse_gp_1 = SparseGP_DTC(kernels_1, sig_e_1, sig_f_1, sig_s_1);
  SparseGP_DTC sparse_gp_2 = SparseGP_DTC(kernels_2, sig_e_2, sig_f_2, sig_s_2);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  test_struc.forces = forces;
  test_struc.stresses = stresses;

  // Add sparse environments and training structures.
  sparse_gp_1.add_training_structure(test_struc);
  sparse_gp_1.add_sparse_environments(test_struc);
  sparse_gp_2.add_training_structure(test_struc);
  sparse_gp_2.add_sparse_environments(test_struc);

  sparse_gp_1.update_matrices_QR();
  sparse_gp_2.update_matrices_QR();

  // Compute likelihoods.
  sparse_gp_1.compute_likelihood();
  sparse_gp_2.compute_likelihood();

  EXPECT_NE(sparse_gp_1.log_marginal_likelihood,
            sparse_gp_2.log_marginal_likelihood);

  // Reset the hyperparameters of the second GP.
  Eigen::VectorXd new_hyps(4);
  new_hyps << sig1, sig_e_1, sig_f_1, sig_s_1;
  sparse_gp_2.set_hyperparameters(new_hyps);

  sparse_gp_2.compute_likelihood();

  EXPECT_EQ(sparse_gp_1.log_marginal_likelihood,
            sparse_gp_2.log_marginal_likelihood);
}

TEST_F(StructureTest, AddOrder) {
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel);
  SparseGP_DTC sparse_gp_1 = SparseGP_DTC(kernels, sigma_e, sigma_f, sigma_s);
  SparseGP_DTC sparse_gp_2 = SparseGP_DTC(kernels, sigma_e, sigma_f, sigma_s);

  // Add structure first.
  sparse_gp_1.add_training_structure(test_struc);
  sparse_gp_1.add_sparse_environments(test_struc);
  sparse_gp_1.update_matrices_QR();

  // Add environments first.
  sparse_gp_2.add_sparse_environments(test_struc);
  sparse_gp_2.add_training_structure(test_struc);
  sparse_gp_2.update_matrices_QR();

  // Check that matrices match.
  for (int i = 0; i < sparse_gp_1.Kuf.rows(); i++) {
    for (int j = 0; j < sparse_gp_1.Kuf.cols(); j++) {
      EXPECT_EQ(sparse_gp_1.Kuf(i, j), sparse_gp_2.Kuf(i, j));
    }
  }

  for (int i = 0; i < sparse_gp_1.Kuu.rows(); i++) {
    for (int j = 0; j < sparse_gp_1.Kuu.cols(); j++) {
      EXPECT_EQ(sparse_gp_1.Kuu(i, j), sparse_gp_2.Kuu(i, j));
    }
  }
}
