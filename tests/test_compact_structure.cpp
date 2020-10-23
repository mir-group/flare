#include "compact_environments.h"
#include "normalized_dot_product.h"
#include "compact_structure.h"
#include "compact_structures.h"
#include "descriptor.h"
#include "three_body.h"
#include "two_body.h"
#include "dot_product_kernel.h"
#include "squared_exponential.h"
#include "local_environment.h"
#include "b2.h"
#include "structure.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdlib.h>

class CompactStructureTest : public ::testing::Test {
public:
  int n_atoms = 5;
  int n_species = 3;
  Eigen::MatrixXd cell, cell_2;
  std::vector<int> species, species_2;
  Eigen::MatrixXd positions, positions_2;
  B2_Calculator desc1;
  B2 ps;
  std::vector<DescriptorCalculator *> descriptor_calculators;
  std::vector<CompactDescriptor *> dc;
  CompactStructure test_struc, test_struc_2;
  StructureDescriptor struc2;
  DescriptorValues struc_desc;

  double cell_size = 10;
  double cutoff = cell_size / 2;
  int N = 3;
  int L = 3;
  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  std::vector<double> radial_hyps{0, cutoff};
  std::vector<double> cutoff_hyps;
  std::vector<int> descriptor_settings{5, N, L};
  int descriptor_index = 0;
  std::vector<double> many_body_cutoffs{cutoff};

  double sigma = 2.0;
  double ls = 0.9;
  int power = 2;
  NormalizedDotProduct kernel_3;
  DotProductKernel kernel_2;
  SquaredExponential kernel;

  CompactStructureTest() {
    // Make positions.
    cell = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    cell_2 = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    positions = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;
    positions_2 = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;

    // Make random species.
    for (int i = 0; i < n_atoms; i++) {
      species.push_back(rand() % n_species);
      species_2.push_back(rand() % n_species);
    }

    desc1 = B2_Calculator(radial_string, cutoff_string, radial_hyps,
                          cutoff_hyps, descriptor_settings, descriptor_index);
    ps = B2(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
            descriptor_settings);

    descriptor_calculators.push_back(&desc1);
    dc.push_back(&ps);

    test_struc = CompactStructure(cell, species, positions, cutoff, dc);
    test_struc_2 = CompactStructure(cell_2, species_2, positions_2, cutoff, dc);
    struc2 = StructureDescriptor(cell, species, positions, cutoff,
                                 many_body_cutoffs, descriptor_calculators);
    
    struc_desc = test_struc.descriptors[0];

    kernel = SquaredExponential(sigma, ls);
    kernel_2 = DotProductKernel(sigma, power, 0);
    kernel_3 = NormalizedDotProduct(sigma, power);
  }
};

TEST_F(CompactStructureTest, TestDescriptor) {
  auto start = std::chrono::steady_clock::now();
  CompactStructure struc1 =
      CompactStructure(cell, species, positions, cutoff, dc);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Compact structure construction: " << elapsed_seconds.count()
            << "s\n";

  start = std::chrono::steady_clock::now();
  StructureDescriptor struc2 =
      StructureDescriptor(cell, species, positions, cutoff, many_body_cutoffs,
                          descriptor_calculators);
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "Structure descriptor construction: " << elapsed_seconds.count()
            << "s\n";
}

TEST_F(CompactStructureTest, TestEnvironments){
    ClusterDescriptor envs;
    envs.add_cluster(struc_desc);
}

TEST_F(CompactStructureTest, TimeSelfKernel) {
  auto start = std::chrono::steady_clock::now();
  Eigen::VectorXd self_kern =
      kernel.self_kernel_struc(struc_desc);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Compact self kernel: " << elapsed_seconds.count() << "s\n";

  start = std::chrono::steady_clock::now();
  Eigen::VectorXd prev_self = kernel_2.self_kernel_struc(struc2);
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "Non-compact self kernel: " << elapsed_seconds.count() << "s\n";

//   for (int i = 0; i < self_kern.size(); i++) {
//     EXPECT_NEAR(self_kern(i), prev_self(i), 1e-8);
//   }
}

TEST_F(CompactStructureTest, TestEnvsEnvs){
  Eigen::MatrixXd kernel_matrix =
    kernel.struc_struc(struc_desc, struc_desc);

  ClusterDescriptor envs;
  envs.add_cluster(struc_desc);
  Eigen::MatrixXd kern_mat = kernel.envs_envs(envs, envs);

  double kern_sum = 0;
  for (int i = 0; i < envs.n_clusters; i++){
    for (int j = 0; j < envs.n_clusters; j++){
      kern_sum += kern_mat(i, j);
    }
  }

  EXPECT_NEAR(kern_sum, kernel_matrix(0, 0), 1e-8);
}

TEST_F(CompactStructureTest, TestEnvsStruc){
  Eigen::MatrixXd kernel_matrix =
    kernel.struc_struc(struc_desc, struc_desc);

  ClusterDescriptor envs;
  envs.add_cluster(struc_desc);
  Eigen::MatrixXd kern_mat = kernel.envs_struc(envs, struc_desc);

  Eigen::VectorXd kern_sum = Eigen::VectorXd::Zero(kern_mat.cols());
  for (int i = 0; i < kern_mat.cols(); i++){
      for (int j = 0; j < kern_mat.rows(); j++){
        kern_sum(i) += kern_mat(j, i);
      }
  }
  
  for (int i = 0; i < kern_sum.size(); i++){
      EXPECT_NEAR(kern_sum(i), kernel_matrix.row(0)(i), 1e-8);
  }
}

TEST_F(CompactStructureTest, StrucStrucFull) {

//   ThreeBody three_body_desc =
//     ThreeBody(cutoff, n_species, cutoff_string, cutoff_hyps);
//   dc[0] = &three_body_desc;

  TwoBody two_body_desc =
    TwoBody(cutoff, n_species, cutoff_string, cutoff_hyps);
  dc[0] = &two_body_desc;

  test_struc = CompactStructure(cell, species, positions, cutoff, dc);
  test_struc_2 = CompactStructure(cell_2, species_2, positions_2, cutoff, dc);
  struc_desc = test_struc.descriptors[0];

  // Compute full kernel matrix.
  Eigen::MatrixXd kernel_matrix = kernel.struc_struc(
      struc_desc, test_struc_2.descriptors[0]);

  double delta = 1e-5;
  double thresh = 2e-4;

  // Check energy/force kernel.
  Eigen::MatrixXd positions_3, positions_4, positions_5, positions_6, cell_3,
      cell_4, cell_5, cell_6, kern_pert, kern_pert_2, kern_pert_3, kern_pert_4;
  CompactStructure test_struc_3, test_struc_4, test_struc_5, test_struc_6;
  double fin_val, exact_val, abs_diff;
  for (int p = 0; p < test_struc_2.noa; p++) {
    for (int m = 0; m < 3; m++) {
      positions_3 = positions_4 = positions_2;
      positions_3(p, m) += delta;
      positions_4(p, m) -= delta;

      test_struc_3 =
          CompactStructure(cell_2, species_2, positions_3, cutoff, dc);
      test_struc_4 =
          CompactStructure(cell_2, species_2, positions_4, cutoff, dc);

      kern_pert = kernel.struc_struc(struc_desc,
                                     test_struc_3.descriptors[0]);
      kern_pert_2 = kernel.struc_struc(struc_desc,
                                       test_struc_4.descriptors[0]);
      fin_val = -(kern_pert(0, 0) - kern_pert_2(0, 0)) / (2 * delta);
      exact_val = kernel_matrix(0, 1 + 3 * p + m);

      EXPECT_NEAR(fin_val, exact_val, thresh);
    }
  }

  // Check force/energy kernel.
  for (int p = 0; p < test_struc.noa; p++) {
    for (int m = 0; m < 3; m++) {
      positions_3 = positions_4 = positions;
      positions_3(p, m) += delta;
      positions_4(p, m) -= delta;

      test_struc_3 = CompactStructure(cell, species, positions_3, cutoff, dc);
      test_struc_4 = CompactStructure(cell, species, positions_4, cutoff, dc);

      kern_pert = kernel.struc_struc(test_struc_2.descriptors[0],
                                     test_struc_3.descriptors[0]);
      kern_pert_2 = kernel.struc_struc(test_struc_2.descriptors[0],
                                       test_struc_4.descriptors[0]);
      fin_val = -(kern_pert(0, 0) - kern_pert_2(0, 0)) / (2 * delta);
      exact_val = kernel_matrix(1 + 3 * p + m, 0);

      EXPECT_NEAR(fin_val, exact_val, thresh);
    }
  }

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
          CompactStructure(cell_3, species_2, positions_3, cutoff, dc);
      test_struc_4 =
          CompactStructure(cell_4, species_2, positions_4, cutoff, dc);

      kern_pert = kernel.struc_struc(struc_desc,
                                     test_struc_3.descriptors[0]);
      kern_pert_2 = kernel.struc_struc(struc_desc,
                                       test_struc_4.descriptors[0]);
      fin_val = -(kern_pert(0, 0) - kern_pert_2(0, 0)) / (2 * delta);
      exact_val = kernel_matrix(0, 1 + 3 * test_struc_2.noa + stress_ind_1) *
                  test_struc_2.volume;

      EXPECT_NEAR(fin_val, exact_val, thresh);

      stress_ind_1++;
    }
  }

  // Check stress/energy.
  stress_ind_1 = 0;
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

      test_struc_3 = CompactStructure(cell_3, species, positions_3, cutoff, dc);
      test_struc_4 = CompactStructure(cell_4, species, positions_4, cutoff, dc);

      kern_pert = kernel.struc_struc(test_struc_2.descriptors[0],
                                     test_struc_3.descriptors[0]);
      kern_pert_2 = kernel.struc_struc(test_struc_2.descriptors[0],
                                       test_struc_4.descriptors[0]);
      fin_val = -(kern_pert(0, 0) - kern_pert_2(0, 0)) / (2 * delta);
      exact_val = kernel_matrix(1 + 3 * test_struc.noa + stress_ind_1, 0) *
                  test_struc.volume;

      EXPECT_NEAR(fin_val, exact_val, thresh);

      stress_ind_1++;
    }
  }

  // Check force/force kernel.
  for (int m = 0; m < test_struc.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions;
      positions_4 = positions;
      positions_3(m, n) += delta;
      positions_4(m, n) -= delta;

      test_struc_3 = CompactStructure(cell, species, positions_3, cutoff, dc);
      test_struc_4 = CompactStructure(cell, species, positions_4, cutoff, dc);

      for (int p = 0; p < test_struc_2.noa; p++) {
        for (int q = 0; q < 3; q++) {
          positions_5 = positions_2;
          positions_6 = positions_2;
          positions_5(p, q) += delta;
          positions_6(p, q) -= delta;

          test_struc_5 =
              CompactStructure(cell_2, species_2, positions_5, cutoff, dc);
          test_struc_6 =
              CompactStructure(cell_2, species_2, positions_6, cutoff, dc);

          kern_pert = kernel.struc_struc(test_struc_3.descriptors[0],
                                         test_struc_5.descriptors[0]);
          kern_pert_2 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_6.descriptors[0]);
          kern_pert_3 = kernel.struc_struc(test_struc_3.descriptors[0],
                                           test_struc_6.descriptors[0]);
          kern_pert_4 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_5.descriptors[0]);

          fin_val = (kern_pert(0, 0) + kern_pert_2(0, 0) - kern_pert_3(0, 0) -
                     kern_pert_4(0, 0)) /
                    (4 * delta * delta);
          exact_val = kernel_matrix(1 + 3 * m + n, 1 + 3 * p + q);

          EXPECT_NEAR(fin_val, exact_val, thresh);
        }
      }
    }
  }

  // Check force/stress kernel.
  for (int m = 0; m < test_struc.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions;
      positions_4 = positions;
      positions_3(m, n) += delta;
      positions_4(m, n) -= delta;

      test_struc_3 = CompactStructure(cell, species, positions_3, cutoff, dc);
      test_struc_4 = CompactStructure(cell, species, positions_4, cutoff, dc);

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
              CompactStructure(cell_3, species_2, positions_5, cutoff, dc);
          test_struc_6 =
              CompactStructure(cell_4, species_2, positions_6, cutoff, dc);

          kern_pert = kernel.struc_struc(test_struc_3.descriptors[0],
                                         test_struc_5.descriptors[0]);
          kern_pert_2 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_6.descriptors[0]);
          kern_pert_3 = kernel.struc_struc(test_struc_3.descriptors[0],
                                           test_struc_6.descriptors[0]);
          kern_pert_4 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_5.descriptors[0]);

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

  // Check stress/force kernel.
  for (int m = 0; m < test_struc_2.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions_2;
      positions_4 = positions_2;
      positions_3(m, n) += delta;
      positions_4(m, n) -= delta;

      test_struc_3 =
          CompactStructure(cell_2, species_2, positions_3, cutoff, dc);
      test_struc_4 =
          CompactStructure(cell_2, species_2, positions_4, cutoff, dc);

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
              CompactStructure(cell_3, species, positions_5, cutoff, dc);
          test_struc_6 =
              CompactStructure(cell_4, species, positions_6, cutoff, dc);

          kern_pert = kernel.struc_struc(test_struc_3.descriptors[0],
                                         test_struc_5.descriptors[0]);
          kern_pert_2 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_6.descriptors[0]);
          kern_pert_3 = kernel.struc_struc(test_struc_3.descriptors[0],
                                           test_struc_6.descriptors[0]);
          kern_pert_4 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_5.descriptors[0]);

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

  // Check stress/stress kernel.
  stress_ind_1 = 0;
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

      test_struc_3 = CompactStructure(cell_3, species, positions_3, cutoff, dc);
      test_struc_4 = CompactStructure(cell_4, species, positions_4, cutoff, dc);

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
              CompactStructure(cell_5, species_2, positions_5, cutoff, dc);
          test_struc_6 =
              CompactStructure(cell_6, species_2, positions_6, cutoff, dc);

          kern_pert = kernel.struc_struc(test_struc_3.descriptors[0],
                                         test_struc_5.descriptors[0]);
          kern_pert_2 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_6.descriptors[0]);
          kern_pert_3 = kernel.struc_struc(test_struc_3.descriptors[0],
                                           test_struc_6.descriptors[0]);
          kern_pert_4 = kernel.struc_struc(test_struc_4.descriptors[0],
                                           test_struc_5.descriptors[0]);

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

// TEST_F(CompactStructureTest, TestStrucs) {
//   CompactStructures test_strucs;
//   test_strucs.add_structure(test_struc);
//   test_strucs.add_structure(test_struc);
// }
