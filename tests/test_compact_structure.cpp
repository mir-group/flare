#include "compact_structure.h"
#include "compact_environments.h"
#include "descriptor.h"
#include "local_environment.h"
#include "structure.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>

class CompactStructureTest : public ::testing::Test {
public:
  Eigen::MatrixXd cell{3, 3};
  std::vector<int> species{0, 1, 2, 3, 4};
  Eigen::MatrixXd positions{5, 3};
  B2_Calculator desc1;
  std::vector<DescriptorCalculator *> descriptor_calculators;
  CompactStructure test_struc;

  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  std::vector<double> radial_hyps{0, 5};
  std::vector<double> cutoff_hyps;
  std::vector<int> descriptor_settings{5, 5, 5};
  int descriptor_index = 0;
  double cutoff = 3;
  std::vector<double> many_body_cutoffs{cutoff};

  CompactStructureTest() {
    cell << 4.0, 0.5, 0.8, -1.2, 3.9, 0.73, -0.8, 0.1, 4.1;

    positions << 1.2, 0.7, 2.3, 3.1, 2.5, 8.9, -1.8, -5.8, 3.0, 0.2, 1.1, 2.1,
        3.2, 1.1, 3.3;

    desc1 = B2_Calculator(radial_string, cutoff_string, radial_hyps,
                          cutoff_hyps, descriptor_settings, descriptor_index);
    descriptor_calculators.push_back(&desc1);
    test_struc = CompactStructure(cell, species, positions, cutoff, &desc1);
  }
};

TEST_F(CompactStructureTest, TestDescriptor) {
  auto start = std::chrono::steady_clock::now();
  CompactStructure struc1 =
      CompactStructure(cell, species, positions, cutoff, &desc1);
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
    CompactEnvironments envs;
    std::vector<int> env_inds_1 {0, 1, 3};
    envs.add_environments(test_struc, env_inds_1);

    std::vector<int> env_inds_2 {2, 4};
    envs.add_environments(test_struc, env_inds_2);

    // Check descriptor values.
    for (int s = 0; s < 5; s ++){
      EXPECT_EQ(envs.descriptors[s].rows(), 1);
      EXPECT_EQ(envs.descriptor_norms[s].size(), 1);
      EXPECT_EQ(envs.descriptor_norms[s][0],
        test_struc.descriptor_norms[s][0]);
    }

    EXPECT_EQ(envs.n_envs, envs.c_atoms[4]+1);
}

TEST_F(CompactStructureTest, TestStrucs) {
  CompactStructures test_strucs;
  test_strucs.add_structure(test_struc);
  test_strucs.add_structure(test_struc);
}
