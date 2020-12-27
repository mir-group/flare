#include "sparse_gp.h"
#include "test_structure.h"
#include "omp.h"
#include <thread>
#include <chrono>
#include <numeric> // Iota

// TEST(TestPar, TestPar){
//   std::cout << omp_get_max_threads() << std::endl;
//   #pragma omp parallel for
//   for (int atom = 0; atom < 4; atom++) {
//     std::cout << omp_get_thread_num() << std::endl;
//     std::this_thread::sleep_for(std::chrono::milliseconds(2000));
//   }
// }

TEST_F(StructureTest, SortTest_norm){
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel_norm);
  SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  test_struc.forces = forces;
  test_struc.stresses = stresses;

  sparse_gp.add_training_structure(test_struc);
  sparse_gp.add_all_environments(test_struc);
  sparse_gp.update_matrices_QR();

  // Compute variances.
  std::vector<Eigen::VectorXd> variances =
    sparse_gp.compute_cluster_uncertainties(test_struc_2);

  // Sort clusters.
  std::vector<std::vector<int>> clusters =
    sparse_gp.sort_clusters_by_uncertainty(test_struc_2);

  EXPECT_EQ(variances.size(), clusters.size());

  for (int i = 0; i < variances.size(); i++){
    for (int j = 0; j < variances[i].size() - 1; j++){
      int ind = clusters[i][j];
      int ind2 = clusters[i][j+1];
      EXPECT_GE(variances[i][ind], variances[i][ind2]);
    }
  }
}

TEST_F(StructureTest, SparseTest_norm) {
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel_norm);
  SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  test_struc.forces = forces;
  //   test_struc.stresses = stresses;

  Eigen::VectorXd forces_2 = Eigen::VectorXd::Random(n_atoms * 3);
  test_struc_2.forces = forces_2;

  sparse_gp.add_training_structure(test_struc);
  sparse_gp.add_all_environments(test_struc);
  sparse_gp.add_all_environments(test_struc_2);

  EXPECT_EQ(sparse_gp.Sigma.rows(), 0);
  EXPECT_EQ(sparse_gp.Kuu_inverse.rows(), 0);

  sparse_gp.update_matrices_QR();
  EXPECT_EQ(sparse_gp.sparse_descriptors[0].n_clusters, sparse_gp.Sigma.rows());
  EXPECT_EQ(sparse_gp.sparse_descriptors[0].n_clusters,
            sparse_gp.Kuu_inverse.rows());

  sparse_gp.predict_DTC(test_struc);
  std::vector<Eigen::VectorXd> cluster_variances =
    sparse_gp.compute_cluster_uncertainties(test_struc_2);

  // Check that the variances on all quantities are positive.
  int mean_size = test_struc.mean_efs.size();
  for (int i = 0; i < mean_size; i++) {
    EXPECT_GE(test_struc.variance_efs[i], 0);
  }

  for (int i = 0; i < cluster_variances.size(); i++){
      for (int j = 0; j < cluster_variances[i].size(); j++){
          EXPECT_GE(cluster_variances[i](j), 0);
      }
  }

  // Compute the marginal likelihood.
  sparse_gp.compute_likelihood();

  EXPECT_EQ(sparse_gp.data_fit + sparse_gp.complexity_penalty +
                sparse_gp.constant_term,
            sparse_gp.log_marginal_likelihood);
  double like1 = sparse_gp.log_marginal_likelihood;

  sparse_gp.compute_likelihood_stable();
  double like2 = sparse_gp.log_marginal_likelihood;

  // Check the likelihood function.
  Eigen::VectorXd hyps = sparse_gp.hyperparameters;
  double like3 = sparse_gp.compute_likelihood_gradient(hyps);

  EXPECT_NEAR(like1, like2, 1e-8);
  EXPECT_NEAR(like1, like3, 1e-8);
}

TEST_F(StructureTest, TestAdd_norm){
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel_norm);
  SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  test_struc.forces = forces;
  test_struc.stresses = stresses;

  std::vector<int> envs(1);
  envs[0] = 2;

  sparse_gp.add_training_structure(test_struc);
  sparse_gp.add_random_environments(test_struc, envs);

  sparse_gp.update_matrices_QR();
  EXPECT_EQ(sparse_gp.sparse_descriptors[0].n_clusters, sparse_gp.Sigma.rows());
  EXPECT_EQ(sparse_gp.sparse_descriptors[0].n_clusters,
            sparse_gp.Kuu_inverse.rows());
}

TEST_F(StructureTest, LikeGrad_norm) {
  // Check that the DTC likelihood gradient is correctly computed.
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel_3_norm);
  SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  test_struc.forces = forces;
  test_struc.stresses = stresses;

  sparse_gp.add_training_structure(test_struc);
  //sparse_gp.add_all_environments(test_struc);
  //std::vector<int> envs(1);
  //envs[0] = 2;

  //sparse_gp.add_training_structure(test_struc);
  //sparse_gp.add_random_environments(test_struc, envs);

  std::vector<int> atoms_to_add;
  atoms_to_add.push_back(0);
  atoms_to_add.push_back(1);
  atoms_to_add.push_back(2);
  sparse_gp.add_specific_environments(test_struc, atoms_to_add); // for debug

  EXPECT_EQ(sparse_gp.Sigma.rows(), 0);
  EXPECT_EQ(sparse_gp.Kuu_inverse.rows(), 0);

  sparse_gp.update_matrices_QR();

  // Test mapping coefficients.
  sparse_gp.write_mapping_coefficients("beta.txt", "Jon V", 0);

  std::cout
      << "Done mapping mean coeffs"
      << std::endl;

  //sparse_gp.write_varmap_coefficients("beta_var.txt", "YX", 0);

  // Test mapped variance
  //sparse_gp.Sigma = Eigen::MatrixXd::Identity(sparse_gp.Sigma.rows(), sparse_gp.Sigma.cols());
  //sparse_gp.Kuu_inverse = Eigen::MatrixXd::Zero(sparse_gp.Kuu.rows(), sparse_gp.Kuu.cols());

  // compute the gp predicted variance
  sparse_gp.predict_DTC(test_struc);
  double gp_var_en = test_struc.variance_efs[0];

  std::cout
      << "predicted dtc var"
      << std::endl;

  // Compute mapping coefficients.
  Eigen::MatrixXd mapped_coeffs =
    kernels[0]->compute_varmap_coefficients(sparse_gp, 0);

  std::cout
      << "computed varmap coeffs"
      << std::endl;

  std::vector<Eigen::MatrixXd> beta_matrices;
  // Set number of descriptors.
  int n_radial = N * n_species; // N and n_species defined in test_structure.h
  int n_descriptors = (n_radial * (n_radial + 1) / 2) * (L + 1);

  std::cout
      << "computing beta matrix"
      << std::endl;

  // Fill in the beta matrix.
  // TODO: Remove factor of 2 from beta.
  Eigen::MatrixXd beta_matrix; // = Eigen::MatrixXd::Zero(n_species * n_descriptors, n_species * n_descriptors);
  int beta_count;
  double beta_val;

  for (int k = 0; k < n_species; k++) {
    for (int l = 0; l < n_species; l++) {
      int kl = k * n_species + l;

      beta_count = 0;
      beta_matrix = Eigen::MatrixXd::Zero(n_descriptors, n_descriptors);

      for (int i = 0; i < n_descriptors; i++) {
        for (int j = 0; j < n_descriptors; j++) {
          beta_matrix(i, j) = mapped_coeffs(kl, beta_count);
          beta_count++;
        }
      }
      beta_matrices.push_back(beta_matrix);
    }
  }

  std::cout
      << "got beta matrices"
      << std::endl;

  // get the (local energy) descriptors of test struc
  Eigen::MatrixXd energy_desc = Eigen::MatrixXd::Zero(n_species, n_descriptors);
  Eigen::MatrixXd mapped_var = Eigen::MatrixXd::Zero(1, 1);
  for (int s = 0; s < n_species; s++) { // TODO: need to check the n_species here
    int n_struc_s = test_struc.descriptors[0].n_clusters_by_type[s];
    for (int j = 0; j < n_struc_s; j++) {
      double norm_j = test_struc.descriptors[0].descriptor_norms[s](j);

      for (int k = 0; k < n_descriptors; k++) {
        energy_desc(s, k) += test_struc.descriptors[0].descriptors[s](j, k) / norm_j;
      }
//      for (int t = 0; t < n_species; t++) { // TODO: need to check the n_species here
//        int n_struc_t = test_struc.descriptors[0].n_clusters_by_type[t];
//        for (int i = 0; i < n_struc_t; i++) {
//          double norm_i = test_struc.descriptors[0].descriptor_norms[t](i);
//
//          mapped_var += test_struc.descriptors[0].descriptors[s].row(j) * 
//              beta_matrices[s * n_species + t] * 
//              test_struc.descriptors[0].descriptors[t].row(i).transpose() / 
//              norm_j / norm_i;
//        }
//      }
    }
  }

  for (int s = 0; s < n_species; s++) { // TODO: need to check the n_species here
    for (int t = 0; t < n_species; t++) { // TODO: need to check the n_species here
      mapped_var += energy_desc.row(s) * beta_matrices[s * n_species + t] * energy_desc.row(t).transpose();
    }
  }

//  // get the (local energy) descriptors of test struc
//  Eigen::MatrixXd mapped_var = Eigen::MatrixXd::Zero(1, 1);
//  for (int s = 0; s < n_species; s++) { // TODO: need to check the n_species here
//    int n_struc = test_struc.descriptors[0].n_clusters_by_type[s];
//    Eigen::MatrixXd desc_type = Eigen::MatrixXd::Zero(1, n_descriptors);
//    for (int j = 0; j < n_struc; j++) {
//      double norm_j = test_struc.descriptors[0].descriptor_norms[s](j);
//      for (int k = 0; k < n_descriptors; k++) {
//        desc_type(0, k) += test_struc.descriptors[0].descriptors[s](j, k) / norm_j;
//      }
//    }
//    mapped_var += desc_type * beta_matrices[s] * desc_type.transpose();
//  }

  std::cout
      << "computed mapped var"
      << std::endl;

  EXPECT_NEAR(gp_var_en, mapped_var(0, 0), 1e-7);

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

TEST_F(StructureTest, Set_Hyps_norm) {
  // Check the reset hyperparameters method.

  int power = 2;
  double sig1 = 1.5, sig2 = 2.0, sig_e_1 = 1.0, sig_e_2 = 2.0, sig_f_1 = 1.5,
    sig_f_2 = 2.5, sig_s_1 = 3.0, sig_s_2 = 3.5;

  SquaredExponential kernel_1 = SquaredExponential(sig1, power);
  SquaredExponential kernel_2 = SquaredExponential(sig2, power);

  std::vector<Kernel *> kernels_1{&kernel_1};
  std::vector<Kernel *> kernels_2{&kernel_2};

  SparseGP sparse_gp_1 = SparseGP(kernels_1, sig_e_1, sig_f_1, sig_s_1);
  SparseGP sparse_gp_2 = SparseGP(kernels_2, sig_e_2, sig_f_2, sig_s_2);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  test_struc.forces = forces;
  test_struc.stresses = stresses;

  Eigen::VectorXd energy_2 = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces_2 = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses_2 = Eigen::VectorXd::Random(6);
  test_struc_2.energy = energy_2;
  test_struc_2.forces = forces_2;
  test_struc_2.stresses = stresses_2;

  // Add sparse environments and training structures.
  sparse_gp_1.add_training_structure(test_struc);
  sparse_gp_1.add_training_structure(test_struc_2);
  sparse_gp_1.add_all_environments(test_struc);
  sparse_gp_1.add_all_environments(test_struc_2);

  sparse_gp_2.add_training_structure(test_struc);
  sparse_gp_2.add_training_structure(test_struc_2);
  sparse_gp_2.add_all_environments(test_struc);
  sparse_gp_2.add_all_environments(test_struc_2);

  sparse_gp_1.update_matrices_QR();
  sparse_gp_2.update_matrices_QR();

  // Compute likelihoods.
  sparse_gp_1.compute_likelihood();
  sparse_gp_2.compute_likelihood();

  EXPECT_NE(sparse_gp_1.log_marginal_likelihood,
            sparse_gp_2.log_marginal_likelihood);

  // Reset the hyperparameters of the second GP.
  Eigen::VectorXd new_hyps(5);
  new_hyps << sig1, power, sig_e_1, sig_f_1, sig_s_1;
  sparse_gp_2.set_hyperparameters(new_hyps);

  sparse_gp_2.compute_likelihood();

  EXPECT_NEAR(sparse_gp_1.log_marginal_likelihood,
            sparse_gp_2.log_marginal_likelihood, 1e-8);
}

TEST_F(StructureTest, AddOrder_norm) {
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel_3);
  SparseGP sparse_gp_1 = SparseGP(kernels, sigma_e, sigma_f, sigma_s);
  SparseGP sparse_gp_2 = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
//   test_struc.energy = energy;
    test_struc.forces = forces;
  //   test_struc.stresses = stresses;

  // Add structure first.
  sparse_gp_1.add_training_structure(test_struc);
  sparse_gp_1.add_all_environments(test_struc);
  sparse_gp_1.update_matrices_QR();

  // Add environments first.
  sparse_gp_2.add_all_environments(test_struc);
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
