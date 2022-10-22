#include "test_structure.h"
#include <list>

// Test different kernel types with different descriptor types
template <typename T>
class KernelTest : public StructureTest {
public:
  using List = std::list<T>;
  T kernel;
  double hyp0 = 1.0; // sigma
  double hyp1 = 2.0; // ls or power

  KernelTest() {}
//    if (typeid(T).name() == std::string("SquaredExponential")) {
//      kernel = T(this->sigma, this->ls);
//    } else if (typeid(T).name() == std::string("NormalizedDotProduct")) {
//      kernel = T(this->sigma, this->power);
//    }

};

using KernelTypes = ::testing::Types<SquaredExponential, NormalizedDotProduct, DotProduct>;
TYPED_TEST_SUITE(KernelTest, KernelTypes);

TYPED_TEST(KernelTest, TimeSelfKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);
  auto start = std::chrono::steady_clock::now();
  Eigen::VectorXd self_kern =
      kernel.self_kernel_struc(this->struc_desc, kernel.kernel_hyperparameters);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Self kernel: " << elapsed_seconds.count() << "s\n";
}

TYPED_TEST(KernelTest, TestEnvsEnvs) {
  TypeParam kernel(this->hyp0, this->hyp1);
  Eigen::MatrixXd kernel_matrix =
      kernel.struc_struc(this->struc_desc, this->struc_desc, kernel.kernel_hyperparameters);

  ClusterDescriptor envs;
  envs.add_all_clusters(this->struc_desc);
  Eigen::MatrixXd kern_mat =
      kernel.envs_envs(envs, envs, kernel.kernel_hyperparameters);

  double kern_sum = 0;
  for (int i = 0; i < envs.n_clusters; i++) {
    for (int j = 0; j < envs.n_clusters; j++) {
      kern_sum += kern_mat(i, j);
    }
  }

  EXPECT_NEAR(kern_sum, kernel_matrix(0, 0), 1e-8);
}

TYPED_TEST(KernelTest, TestEnvsStruc) {
  TypeParam kernel(this->hyp0, this->hyp1);
  Eigen::MatrixXd kernel_matrix =
      kernel.struc_struc(this->struc_desc, this->struc_desc, kernel.kernel_hyperparameters);

  ClusterDescriptor envs;
  envs.add_all_clusters(this->struc_desc);
  Eigen::MatrixXd kern_mat =
      kernel.envs_struc(envs, this->struc_desc, kernel.kernel_hyperparameters);

  Eigen::VectorXd kern_sum = Eigen::VectorXd::Zero(kern_mat.cols());
  for (int i = 0; i < kern_mat.cols(); i++) {
    for (int j = 0; j < kern_mat.rows(); j++) {
      kern_sum(i) += kern_mat(j, i);
    }
  }

  for (int i = 0; i < kern_sum.size(); i++) {
    EXPECT_NEAR(kern_sum(i), kernel_matrix.row(0)(i), 1e-8);
  }
}

TYPED_TEST(KernelTest, SqExpGrad) {
  // Test envs_envs_grad and envs_struc_grad methods of squared exponential
  // kernel.

  TypeParam kernel(this->hyp0, this->hyp1);
  std::string kernel_name(typeid(kernel).name());
  if (kernel_name.find(std::string("SquaredExponential")) == std::string::npos) {
    GTEST_SKIP();
  }

  // Define cluster.
  ClusterDescriptor envs;
  envs.add_all_clusters(this->struc_desc);

  // Define hyperparameters.
  double delta = 1e-4;
  double thresh = 1e-6;
  Eigen::VectorXd hyps = kernel.kernel_hyperparameters;
  Eigen::VectorXd sig_hyps_up = hyps;
  Eigen::VectorXd sig_hyps_down = hyps;
  sig_hyps_up(0) += delta;
  sig_hyps_down(0) -= delta;
  Eigen::VectorXd ls_hyps_up = hyps;
  Eigen::VectorXd ls_hyps_down = hyps;
  ls_hyps_up(1) += delta;
  ls_hyps_down(1) -= delta;

  // Check env/env gradient.
  std::vector<Eigen::MatrixXd> env_grad_1 =
      kernel.envs_envs_grad(envs, envs, kernel.kernel_hyperparameters);
  std::vector<Eigen::MatrixXd> env_grad_2 =
      kernel.envs_envs_grad(envs, envs, ls_hyps_up);
  std::vector<Eigen::MatrixXd> env_grad_3 =
      kernel.envs_envs_grad(envs, envs, ls_hyps_down);

  double exact_val = env_grad_1[2](0, 1);
  double fin_val = (env_grad_2[0](0, 1) - env_grad_3[0](0, 1)) / (2 * delta);
  EXPECT_NEAR(exact_val, fin_val, thresh);

  // Check env/struc gradient.
  std::vector<Eigen::MatrixXd> struc_grad_1 =
      kernel.envs_struc_grad(envs, this->struc_desc, kernel.kernel_hyperparameters);

  for (int g = 0; g < 2; g++) {
    Eigen::VectorXd hyps_up = hyps;
    Eigen::VectorXd hyps_down = hyps;
    hyps_up(g) += delta;
    hyps_down(g) -= delta;
    std::vector<Eigen::MatrixXd> struc_grad_2 =
        kernel.envs_struc_grad(envs, this->struc_desc, hyps_up);
    std::vector<Eigen::MatrixXd> struc_grad_3 =
        kernel.envs_struc_grad(envs, this->struc_desc, hyps_down);

    for (int i = 0; i < envs.n_clusters; i++) {
      for (int j = 0; j < this->test_struc.noa * 3 + 7; j++) {
        exact_val = struc_grad_1[g + 1](i, j);
        fin_val = (struc_grad_2[0](i, j) - struc_grad_3[0](i, j)) / (2 * delta);
        EXPECT_NEAR(exact_val, fin_val, thresh);
      }
    }
  }
}

TYPED_TEST(KernelTest, EnergyForceKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);

  // TODO: Systematically test all implemented descriptors and kernels.

  //   ThreeBody three_body_desc =
  //     ThreeBody(cutoff, n_species, cutoff_string, cutoff_hyps);
  //   dc[0] = &three_body_desc;

  //   ThreeBodyWide three_body_desc =
  //     ThreeBodyWide(cutoff, n_species, cutoff_string, cutoff_hyps);
  //   dc[0] = &three_body_desc;

  //   FourBody four_body_desc =
  //     FourBody(cutoff, n_species, cutoff_string, cutoff_hyps);
  //   dc[0] = &four_body_desc;

  //   TwoBody two_body_desc =
  //     TwoBody(cutoff, n_species, cutoff_string, cutoff_hyps);
  //   dc[0] = &two_body_desc;

  Eigen::MatrixXd cell = this->cell;
  std::vector<int> species = this->species;
  Eigen::MatrixXd positions = this->positions;

  Eigen::MatrixXd cell_2 = this->cell_2;
  std::vector<int> species_2 = this->species_2;
  Eigen::MatrixXd positions_2 = this->positions_2;

  double cutoff = this->cutoff;
  std::vector<Descriptor *> dc = this->dc;

  Structure test_struc(cell, species, positions, cutoff, dc);
  Structure test_struc_2(cell_2, species_2, positions_2, cutoff, dc);
  DescriptorValues struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0], kernel.kernel_hyperparameters);

  double delta = 1e-4;
  double thresh = 1e-4;

  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  Structure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;

  // Check energy/force kernel.
  for (int p = 0; p < test_struc_2.noa; p++) {
    for (int m = 0; m < 3; m++) {
      positions_3 = positions_4 = positions_2;
      positions_3(p, m) += delta;
      positions_4(p, m) -= delta;

      test_struc_3 =
          Structure(cell_2, species_2, positions_3, cutoff, dc);
      test_struc_4 =
          Structure(cell_2, species_2, positions_4, cutoff, dc);

      kern_pert = kernel.struc_struc(struc_desc, test_struc_3.descriptors[0],
                                     kernel.kernel_hyperparameters);
      kern_pert_2 = kernel.struc_struc(struc_desc, test_struc_4.descriptors[0],
                                       kernel.kernel_hyperparameters);
      fin_val = -(kern_pert(0, 0) - kern_pert_2(0, 0)) / (2 * delta);
      exact_val = kernel_matrix(0, 1 + 3 * p + m);

      EXPECT_NEAR(fin_val, exact_val, thresh);
    }
  }
}

TYPED_TEST(KernelTest, ForceEnergyKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);

  Eigen::MatrixXd cell = this->cell;
  std::vector<int> species = this->species;
  Eigen::MatrixXd positions = this->positions;

  Eigen::MatrixXd cell_2 = this->cell_2;
  std::vector<int> species_2 = this->species_2;
  Eigen::MatrixXd positions_2 = this->positions_2;

  double cutoff = this->cutoff;
  std::vector<Descriptor *> dc = this->dc;

  Structure test_struc(cell, species, positions, cutoff, dc);
  Structure test_struc_2(cell_2, species_2, positions_2, cutoff, dc);
  DescriptorValues struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0], kernel.kernel_hyperparameters);

  double delta = 1e-4;
  double thresh = 1e-4;

  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  Structure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;

  // Check force/energy kernel.
  for (int p = 0; p < test_struc.noa; p++) {
    for (int m = 0; m < 3; m++) {
      positions_3 = positions_4 = positions;
      positions_3(p, m) += delta;
      positions_4(p, m) -= delta;

      test_struc_3 = Structure(cell, species, positions_3, cutoff, dc);
      test_struc_4 = Structure(cell, species, positions_4, cutoff, dc);

      kern_pert = kernel.struc_struc(test_struc_2.descriptors[0],
                                     test_struc_3.descriptors[0],
                                     kernel.kernel_hyperparameters);
      kern_pert_2 = kernel.struc_struc(test_struc_2.descriptors[0],
                                       test_struc_4.descriptors[0],
                                       kernel.kernel_hyperparameters);
      fin_val = -(kern_pert(0, 0) - kern_pert_2(0, 0)) / (2 * delta);
      exact_val = kernel_matrix(1 + 3 * p + m, 0);

      EXPECT_NEAR(fin_val, exact_val, thresh);
    }
  }
}

TYPED_TEST(KernelTest, EnergyStressKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);

  Eigen::MatrixXd cell = this->cell;
  std::vector<int> species = this->species;
  Eigen::MatrixXd positions = this->positions;

  Eigen::MatrixXd cell_2 = this->cell_2;
  std::vector<int> species_2 = this->species_2;
  Eigen::MatrixXd positions_2 = this->positions_2;

  double cutoff = this->cutoff;
  std::vector<Descriptor *> dc = this->dc;

  Structure test_struc(cell, species, positions, cutoff, dc);
  Structure test_struc_2(cell_2, species_2, positions_2, cutoff, dc);
  DescriptorValues struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0], kernel.kernel_hyperparameters);

  double delta = 1e-4;
  double thresh = 1e-4;

  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  Structure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;

  // Check energy/stress.
  int stress_ind_1 = 0;
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_3 = cell_4 = cell_2;
      positions_3 = positions_4 = positions_2;

      // Perform strain.
      cell_3(0, m) += cell_2(0, n) * delta;
      cell_3(1, m) += cell_2(1, n) * delta;
      cell_3(2, m) += cell_2(2, n) * delta;

      cell_4(0, m) -= cell_2(0, n) * delta;
      cell_4(1, m) -= cell_2(1, n) * delta;
      cell_4(2, m) -= cell_2(2, n) * delta;

      for (int k = 0; k < test_struc.noa; k++) {
        positions_3(k, m) += positions_2(k, n) * delta;
        positions_4(k, m) -= positions_2(k, n) * delta;
      }

      test_struc_3 =
          Structure(cell_3, species_2, positions_3, cutoff, dc);
      test_struc_4 =
          Structure(cell_4, species_2, positions_4, cutoff, dc);

      kern_pert = kernel.struc_struc(struc_desc, test_struc_3.descriptors[0],
                                     kernel.kernel_hyperparameters);
      kern_pert_2 = kernel.struc_struc(struc_desc, test_struc_4.descriptors[0],
                                       kernel.kernel_hyperparameters);
      fin_val = -(kern_pert(0, 0) - kern_pert_2(0, 0)) / (2 * delta);
      exact_val = kernel_matrix(0, 1 + 3 * test_struc_2.noa + stress_ind_1) *
                  test_struc_2.volume;

      EXPECT_NEAR(fin_val, exact_val, thresh);

      stress_ind_1++;
    }
  }
}

TYPED_TEST(KernelTest, StressEnergyKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);

  Eigen::MatrixXd cell = this->cell;
  std::vector<int> species = this->species;
  Eigen::MatrixXd positions = this->positions;

  Eigen::MatrixXd cell_2 = this->cell_2;
  std::vector<int> species_2 = this->species_2;
  Eigen::MatrixXd positions_2 = this->positions_2;

  double cutoff = this->cutoff;
  std::vector<Descriptor *> dc = this->dc;

  Structure test_struc(cell, species, positions, cutoff, dc);
  Structure test_struc_2(cell_2, species_2, positions_2, cutoff, dc);
  DescriptorValues struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0], kernel.kernel_hyperparameters);

  double delta = 1e-4;
  double thresh = 1e-4;

  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  Structure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;

  // Check stress/energy.
  int stress_ind_1 = 0;
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_3 = cell_4 = cell;
      positions_3 = positions_4 = positions;

      // Perform strain.
      cell_3(0, m) += cell(0, n) * delta;
      cell_3(1, m) += cell(1, n) * delta;
      cell_3(2, m) += cell(2, n) * delta;

      cell_4(0, m) -= cell(0, n) * delta;
      cell_4(1, m) -= cell(1, n) * delta;
      cell_4(2, m) -= cell(2, n) * delta;

      for (int k = 0; k < test_struc.noa; k++) {
        positions_3(k, m) += positions(k, n) * delta;
        positions_4(k, m) -= positions(k, n) * delta;
      }

      test_struc_3 = Structure(cell_3, species, positions_3, cutoff, dc);
      test_struc_4 = Structure(cell_4, species, positions_4, cutoff, dc);

      kern_pert = kernel.struc_struc(test_struc_2.descriptors[0],
                                     test_struc_3.descriptors[0],
                                     kernel.kernel_hyperparameters);
      kern_pert_2 = kernel.struc_struc(test_struc_2.descriptors[0],
                                       test_struc_4.descriptors[0],
                                       kernel.kernel_hyperparameters);
      fin_val = -(kern_pert(0, 0) - kern_pert_2(0, 0)) / (2 * delta);
      exact_val = kernel_matrix(1 + 3 * test_struc.noa + stress_ind_1, 0) *
                  test_struc.volume;

      EXPECT_NEAR(fin_val, exact_val, thresh);

      stress_ind_1++;
    }
  }
}

TYPED_TEST(KernelTest, ForceForceKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);

  Eigen::MatrixXd cell = this->cell;
  std::vector<int> species = this->species;
  Eigen::MatrixXd positions = this->positions;

  Eigen::MatrixXd cell_2 = this->cell_2;
  std::vector<int> species_2 = this->species_2;
  Eigen::MatrixXd positions_2 = this->positions_2;

  double cutoff = this->cutoff;
  std::vector<Descriptor *> dc = this->dc;

  Structure test_struc(cell, species, positions, cutoff, dc);
  Structure test_struc_2(cell_2, species_2, positions_2, cutoff, dc);
  DescriptorValues struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0], kernel.kernel_hyperparameters);

  double delta = 1e-4;
  double thresh = 1e-4;

  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  Structure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;

  // Check force/force kernel.
  for (int m = 0; m < test_struc.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions;
      positions_4 = positions;
      positions_3(m, n) += delta;
      positions_4(m, n) -= delta;

      test_struc_3 = Structure(cell, species, positions_3, cutoff, dc);
      test_struc_4 = Structure(cell, species, positions_4, cutoff, dc);

      for (int p = 0; p < test_struc_2.noa; p++) {
        for (int q = 0; q < 3; q++) {
          positions_5 = positions_2;
          positions_6 = positions_2;
          positions_5(p, q) += delta;
          positions_6(p, q) -= delta;

          test_struc_5 =
              Structure(cell_2, species_2, positions_5, cutoff, dc);
          test_struc_6 =
              Structure(cell_2, species_2, positions_6, cutoff, dc);

          kern_pert = kernel.struc_struc(test_struc_3.descriptors[0],
                                         test_struc_5.descriptors[0],
                                         kernel.kernel_hyperparameters);
          kern_pert_2 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_6.descriptors[0],
                                           kernel.kernel_hyperparameters);
          kern_pert_3 = kernel.struc_struc(test_struc_3.descriptors[0],
                                           test_struc_6.descriptors[0],
                                           kernel.kernel_hyperparameters);
          kern_pert_4 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_5.descriptors[0],
                                           kernel.kernel_hyperparameters);

          fin_val = (kern_pert(0, 0) + kern_pert_2(0, 0) - kern_pert_3(0, 0) -
                     kern_pert_4(0, 0)) /
                    (4 * delta * delta);
          exact_val = kernel_matrix(1 + 3 * m + n, 1 + 3 * p + q);

          EXPECT_NEAR(fin_val, exact_val, thresh);
        }
      }
    }
  }
}

TYPED_TEST(KernelTest, ForceStressKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);

  Eigen::MatrixXd cell = this->cell;
  std::vector<int> species = this->species;
  Eigen::MatrixXd positions = this->positions;

  Eigen::MatrixXd cell_2 = this->cell_2;
  std::vector<int> species_2 = this->species_2;
  Eigen::MatrixXd positions_2 = this->positions_2;

  double cutoff = this->cutoff;
  std::vector<Descriptor *> dc = this->dc;

  Structure test_struc(cell, species, positions, cutoff, dc);
  Structure test_struc_2(cell_2, species_2, positions_2, cutoff, dc);
  DescriptorValues struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0], kernel.kernel_hyperparameters);

  double delta = 1e-4;
  double thresh = 1e-4;

  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  Structure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;

  // Check force/stress kernel.
  for (int m = 0; m < test_struc.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions;
      positions_4 = positions;
      positions_3(m, n) += delta;
      positions_4(m, n) -= delta;

      test_struc_3 = Structure(cell, species, positions_3, cutoff, dc);
      test_struc_4 = Structure(cell, species, positions_4, cutoff, dc);

      int stress_count = 0;
      for (int p = 0; p < 3; p++) {
        for (int q = p; q < 3; q++) {
          cell_3 = cell_4 = cell_2;
          positions_5 = positions_6 = positions_2;

          // Perform strain.
          cell_3(0, p) += cell_2(0, q) * delta;
          cell_3(1, p) += cell_2(1, q) * delta;
          cell_3(2, p) += cell_2(2, q) * delta;

          cell_4(0, p) -= cell_2(0, q) * delta;
          cell_4(1, p) -= cell_2(1, q) * delta;
          cell_4(2, p) -= cell_2(2, q) * delta;

          for (int k = 0; k < test_struc_2.noa; k++) {
            positions_5(k, p) += positions_2(k, q) * delta;
            positions_6(k, p) -= positions_2(k, q) * delta;
          }

          test_struc_5 =
              Structure(cell_3, species_2, positions_5, cutoff, dc);
          test_struc_6 =
              Structure(cell_4, species_2, positions_6, cutoff, dc);

          kern_pert = kernel.struc_struc(test_struc_3.descriptors[0],
                                         test_struc_5.descriptors[0],
                                         kernel.kernel_hyperparameters);
          kern_pert_2 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_6.descriptors[0],
                                           kernel.kernel_hyperparameters);
          kern_pert_3 = kernel.struc_struc(test_struc_3.descriptors[0],
                                           test_struc_6.descriptors[0],
                                           kernel.kernel_hyperparameters);
          kern_pert_4 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_5.descriptors[0],
                                           kernel.kernel_hyperparameters);

          fin_val = (kern_pert(0, 0) + kern_pert_2(0, 0) - kern_pert_3(0, 0) -
                     kern_pert_4(0, 0)) /
                    (4 * delta * delta);
          exact_val = kernel_matrix(1 + 3 * m + n,
                                    1 + 3 * test_struc_2.noa + stress_count) *
                      test_struc_2.volume;

          EXPECT_NEAR(fin_val, exact_val, thresh);

          stress_count++;
        }
      }
    }
  }
}

TYPED_TEST(KernelTest, StressForceKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);

  Eigen::MatrixXd cell = this->cell;
  std::vector<int> species = this->species;
  Eigen::MatrixXd positions = this->positions;

  Eigen::MatrixXd cell_2 = this->cell_2;
  std::vector<int> species_2 = this->species_2;
  Eigen::MatrixXd positions_2 = this->positions_2;

  double cutoff = this->cutoff;
  std::vector<Descriptor *> dc = this->dc;

  Structure test_struc(cell, species, positions, cutoff, dc);
  Structure test_struc_2(cell_2, species_2, positions_2, cutoff, dc);
  DescriptorValues struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0], kernel.kernel_hyperparameters);

  double delta = 1e-4;
  double thresh = 1e-4;

  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  Structure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;

  // Check stress/force kernel.
  for (int m = 0; m < test_struc_2.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions_2;
      positions_4 = positions_2;
      positions_3(m, n) += delta;
      positions_4(m, n) -= delta;

      test_struc_3 =
          Structure(cell_2, species_2, positions_3, cutoff, dc);
      test_struc_4 =
          Structure(cell_2, species_2, positions_4, cutoff, dc);

      int stress_count = 0;
      for (int p = 0; p < 3; p++) {
        for (int q = p; q < 3; q++) {
          cell_3 = cell_4 = cell;
          positions_5 = positions_6 = positions;

          // Perform strain.
          cell_3(0, p) += cell(0, q) * delta;
          cell_3(1, p) += cell(1, q) * delta;
          cell_3(2, p) += cell(2, q) * delta;

          cell_4(0, p) -= cell(0, q) * delta;
          cell_4(1, p) -= cell(1, q) * delta;
          cell_4(2, p) -= cell(2, q) * delta;

          for (int k = 0; k < test_struc_2.noa; k++) {
            positions_5(k, p) += positions(k, q) * delta;
            positions_6(k, p) -= positions(k, q) * delta;
          }

          test_struc_5 =
              Structure(cell_3, species, positions_5, cutoff, dc);
          test_struc_6 =
              Structure(cell_4, species, positions_6, cutoff, dc);

          kern_pert = kernel.struc_struc(test_struc_3.descriptors[0],
                                         test_struc_5.descriptors[0],
                                         kernel.kernel_hyperparameters);
          kern_pert_2 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_6.descriptors[0],
                                           kernel.kernel_hyperparameters);
          kern_pert_3 = kernel.struc_struc(test_struc_3.descriptors[0],
                                           test_struc_6.descriptors[0],
                                           kernel.kernel_hyperparameters);
          kern_pert_4 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_5.descriptors[0],
                                           kernel.kernel_hyperparameters);

          fin_val = (kern_pert(0, 0) + kern_pert_2(0, 0) - kern_pert_3(0, 0) -
                     kern_pert_4(0, 0)) /
                    (4 * delta * delta);
          exact_val = kernel_matrix(1 + 3 * test_struc.noa + stress_count,
                                    1 + 3 * m + n) *
                      test_struc.volume;

          EXPECT_NEAR(fin_val, exact_val, thresh);

          stress_count++;
        }
      }
    }
  }
}

TYPED_TEST(KernelTest, StressStressKernel) {
  TypeParam kernel(this->hyp0, this->hyp1);

  Eigen::MatrixXd cell = this->cell;
  std::vector<int> species = this->species;
  Eigen::MatrixXd positions = this->positions;

  Eigen::MatrixXd cell_2 = this->cell_2;
  std::vector<int> species_2 = this->species_2;
  Eigen::MatrixXd positions_2 = this->positions_2;

  double cutoff = this->cutoff;
  std::vector<Descriptor *> dc = this->dc;

  Structure test_struc(cell, species, positions, cutoff, dc);
  Structure test_struc_2(cell_2, species_2, positions_2, cutoff, dc);
  DescriptorValues struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0], kernel.kernel_hyperparameters);

  double delta = 1e-4;
  double thresh = 1e-4;

  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  Structure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;

  // Check stress/stress kernel.
  int stress_ind_1 = 0;
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_3 = cell_4 = cell;
      positions_3 = positions_4 = positions;

      // Perform strain.
      cell_3(0, m) += cell(0, n) * delta;
      cell_3(1, m) += cell(1, n) * delta;
      cell_3(2, m) += cell(2, n) * delta;

      cell_4(0, m) -= cell(0, n) * delta;
      cell_4(1, m) -= cell(1, n) * delta;
      cell_4(2, m) -= cell(2, n) * delta;

      for (int k = 0; k < test_struc.noa; k++) {
        positions_3(k, m) += positions(k, n) * delta;
        positions_4(k, m) -= positions(k, n) * delta;
      }

      test_struc_3 = Structure(cell_3, species, positions_3, cutoff, dc);
      test_struc_4 = Structure(cell_4, species, positions_4, cutoff, dc);

      int stress_ind_2 = 0;
      for (int p = 0; p < 3; p++) {
        for (int q = p; q < 3; q++) {
          cell_5 = cell_6 = cell_2;
          positions_5 = positions_6 = positions_2;

          // Perform strain.
          cell_5(0, p) += cell_2(0, q) * delta;
          cell_5(1, p) += cell_2(1, q) * delta;
          cell_5(2, p) += cell_2(2, q) * delta;

          cell_6(0, p) -= cell_2(0, q) * delta;
          cell_6(1, p) -= cell_2(1, q) * delta;
          cell_6(2, p) -= cell_2(2, q) * delta;

          for (int k = 0; k < test_struc_2.noa; k++) {
            positions_5(k, p) += positions_2(k, q) * delta;
            positions_6(k, p) -= positions_2(k, q) * delta;
          }

          test_struc_5 =
              Structure(cell_5, species_2, positions_5, cutoff, dc);
          test_struc_6 =
              Structure(cell_6, species_2, positions_6, cutoff, dc);

          kern_pert = kernel.struc_struc(test_struc_3.descriptors[0],
                                         test_struc_5.descriptors[0],
                                         kernel.kernel_hyperparameters);
          kern_pert_2 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_6.descriptors[0],
                                           kernel.kernel_hyperparameters);
          kern_pert_3 = kernel.struc_struc(test_struc_3.descriptors[0],
                                           test_struc_6.descriptors[0],
                                           kernel.kernel_hyperparameters);
          kern_pert_4 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_5.descriptors[0],
                                           kernel.kernel_hyperparameters);

          fin_val = (kern_pert(0, 0) + kern_pert_2(0, 0) - kern_pert_3(0, 0) -
                     kern_pert_4(0, 0)) /
                    (4 * delta * delta);

          exact_val = kernel_matrix(1 + 3 * test_struc.noa + stress_ind_1,
                                    1 + 3 * test_struc_2.noa + stress_ind_2) *
                      test_struc.volume * test_struc_2.volume;

          EXPECT_NEAR(fin_val, exact_val, thresh);

          stress_ind_2++;
        }
      }
      stress_ind_1++;
    }
  }
}
