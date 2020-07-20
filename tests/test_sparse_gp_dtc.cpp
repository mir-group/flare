#include "sparse_gp_dtc.h"
#include "test_sparse_gp.h"

TEST_F(SparseTest, DTC_Prediction){
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  SparseGP_DTC sparse_gp = SparseGP_DTC(kernels, sigma_e, sigma_f, sigma_s);

  LocalEnvironment env1 = test_struc.local_environments[0];
  LocalEnvironment env2 = test_struc.local_environments[1];
  std::vector<LocalEnvironment> sparse_envs {env1, env2};
  sparse_gp.add_sparse_environments(sparse_envs);
  sparse_gp.add_training_structure(test_struc);

  EXPECT_EQ(sparse_gp.Sigma.rows(), 0);
  EXPECT_EQ(sparse_gp.Kuu_inverse.rows(), 0);

  sparse_gp.update_matrices();

  EXPECT_EQ(sparse_gp.sparse_environments.size(), sparse_gp.Sigma.rows());
  EXPECT_EQ(sparse_gp.sparse_environments.size(), sparse_gp.Kuu_inverse.rows());

  Eigen::VectorXd mean_vector, variance_vector;
  std::vector<Eigen::VectorXd> mean_contributions;
  sparse_gp.predict_DTC(test_struc, mean_vector, variance_vector,
                        mean_contributions);

  // Check that mean contributions sum to the total.
  int mean_size = mean_vector.rows();
  EXPECT_EQ(mean_size, 1 + test_struc.noa * 3 + 6);
  Eigen::VectorXd mean_sum = Eigen::VectorXd::Zero(mean_size);
  for (int i = 0; i < sparse_gp.kernels.size(); i++){
      mean_sum += mean_contributions[i];
  }

  double threshold = 1e-8;
  for (int i = 0; i < mean_size; i++){
      EXPECT_NEAR(mean_sum[i], mean_vector[i], threshold);
  }

  // Check that Kuu and Kuf kernels sum to the total.
  int n_sparse = sparse_gp.Kuu.rows();

  Eigen::MatrixXd Kuu_sum =
    Eigen::MatrixXd::Zero(n_sparse, n_sparse);
  Eigen::MatrixXd Kuf_sum =
    Eigen::MatrixXd::Zero(sparse_gp.Kuf_struc.rows(),
                          sparse_gp.Kuf_struc.cols());
  for (int i = 0; i < kernels.size(); i++){
      Kuu_sum += sparse_gp.Kuu_kernels[i];
      Kuf_sum.block(0, 0, n_sparse, sparse_gp.n_energy_labels) +=
        sparse_gp.Kuf_struc_energy[i];
      Kuf_sum.block(0, sparse_gp.n_energy_labels, n_sparse,
                    sparse_gp.n_force_labels) +=
        sparse_gp.Kuf_struc_force[i];
      Kuf_sum.block(0, sparse_gp.n_energy_labels + sparse_gp.n_force_labels,
                    n_sparse, sparse_gp.n_stress_labels) +=
        sparse_gp.Kuf_struc_stress[i];
  }

  for (int i = 0; i < sparse_gp.Kuu.rows(); i++){
      for (int j = 0; j < sparse_gp.Kuu.cols(); j++){
          EXPECT_NEAR(Kuu_sum(i, j), sparse_gp.Kuu(i, j), threshold);
      }
  }

  for (int i = 0; i < sparse_gp.Kuf_struc.rows(); i++){
      for (int j = 0; j < sparse_gp.Kuf_struc.cols(); j++){
          EXPECT_NEAR(Kuf_sum(i, j), sparse_gp.Kuf_struc(i, j), threshold);
      }
  }

  // Check that the variances on all quantities are positive.
  for (int i = 0; i < mean_size; i++){
      EXPECT_GE(variance_vector[i], 0);
  }

  // Compute the marginal likelihood.
  sparse_gp.compute_DTC_likelihood();
  EXPECT_EQ(sparse_gp.data_fit + sparse_gp.complexity_penalty +
            sparse_gp.constant_term, sparse_gp.log_marginal_likelihood);

}
