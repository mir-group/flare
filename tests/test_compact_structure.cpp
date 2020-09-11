#include "compact_structure.h"
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
    test_struc = CompactStructure(cell, species, positions, cutoff,
                                  descriptor_calculators);
  }
};

TEST_F(CompactStructureTest, TestWrapped) {
  auto start = std::chrono::steady_clock::now();
  CompactStructure test_struc = CompactStructure(
      cell, species, positions, cutoff, descriptor_calculators);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

  std::cout << test_struc.cumulative_neighbor_count << std::endl;
}
