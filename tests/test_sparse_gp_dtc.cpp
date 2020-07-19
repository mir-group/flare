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

  std::cout << "kuf kernels" << std::endl;
  std::cout << sparse_gp.Kuf_struc_energy[0] << std::endl;
  std::cout << sparse_gp.Kuf_struc_force[0] << std::endl;
  std::cout << sparse_gp.Kuf_struc_stress[0] << std::endl;

//   EXPECT_EQ(sparse_gp.Sigma.rows(), 0);
//   EXPECT_EQ(sparse_gp.Kuu_inverse.rows(), 0);

//   sparse_gp.update_matrices();

//   EXPECT_EQ(sparse_gp.sparse_environments.size(), sparse_gp.Sigma.rows());
//   EXPECT_EQ(sparse_gp.sparse_environments.size(), sparse_gp.Kuu_inverse.rows());

//   Eigen::VectorXd mean_vector, variance_vector;
//   std::vector<Eigen::VectorXd> mean_contributions;
//   sparse_gp.predict_DTC(test_struc, mean_vector, variance_vector,
//                         mean_contributions);

//   // Check that mean contributions sum to the total.
//   int mean_size = mean_vector.rows();
//   EXPECT_EQ(mean_size, 1 + test_struc.noa * 3 + 6);
//   Eigen::VectorXd mean_sum = Eigen::VectorXd::Zero(mean_size);
//   for (int i = 0; i < sparse_gp.kernels.size(); i++){
//       mean_sum += mean_contributions[i];
//   }

//   double threshold = 1e-8;
//   for (int i = 0; i < mean_size; i++){
//       EXPECT_NEAR(mean_sum[i], mean_vector[i], threshold);
//   }

//   // Check that Kuu kernels sum to the total.
//   Eigen::MatrixXd Kuu_sum =
//     Eigen::MatrixXd::Zero(sparse_gp.Kuu.rows(),
//                           sparse_gp.Kuu.cols());
//   for (int i = 0; i < kernels.size(); i++){
//       Kuu_sum += sparse_gp.Kuu_kernels[i];
//   }

//   for (int i = 0; i < sparse_gp.Kuu.rows(); i++){
//       for (int j = 0; j < sparse_gp.Kuu.cols(); j++){
//           EXPECT_NEAR(Kuu_sum(i, j), sparse_gp.Kuu(i, j), threshold);
//       }
//   }

//   // TODO: Check that Kuf kernels sum to the total.

//   // Check that the variances on all quantities are positive.
//   for (int i = 0; i < mean_size; i++){
//       EXPECT_GE(variance_vector[i], 0);
//   }

//   std::cout << mean_vector << std::endl;

//   // Compute the marginal likelihood.
//   sparse_gp.compute_DTC_likelihood();
//   EXPECT_EQ(sparse_gp.data_fit + sparse_gp.complexity_penalty +
//             sparse_gp.constant_term, sparse_gp.log_marginal_likelihood);

}

// TEST(MatTest, MatTest){
//     Eigen::MatrixXd test1 = Eigen::MatrixXd::Zero(2, 2);
//     test1(1, 1) = 3.14;
//     std::cout << test1 << std::endl;

//     Eigen::MatrixXd test2 = Eigen::MatrixXd::Zero(2, 1);

//     // std::cout << test1.cols() << std::endl;

//     test1.block(0, 1, 2, 1) = test2;
//     std::cout << test1 << std::endl;
// }

TEST(VecTest, VecTest){
    Eigen::VectorXd test1 = Eigen::VectorXd::Constant(2, 3);
    std::cout << test1 << std::endl;
}
