#include "compact_gp.h"
#include "test_compact_structure.h"
#include <chrono>

TEST_F(CompactStructureTest, SparseTest) {
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<CompactKernel *> kernels;
  kernels.push_back(&kernel);
  CompactGP sparse_gp = CompactGP(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  //   test_struc.forces = forces;
  //   test_struc.stresses = stresses;

  sparse_gp.add_training_structure(test_struc);
  sparse_gp.add_sparse_environments(test_struc);

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

TEST_F(CompactStructureTest, SqExpKuf) {
  // Test K_uf grad method of squared exponential kernel.

  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<CompactKernel *> kernels;
  kernels.push_back(&kernel);
  CompactGP sparse_gp = CompactGP(kernels, sigma_e, sigma_f, sigma_s);

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
