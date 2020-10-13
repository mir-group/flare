#include "compact_structure.h"
#include "two_body.h"
#include "compact_descriptor.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdlib.h>

class TwoBodyTest : public ::testing::Test {
public:
  int n_atoms = 5;
  int n_species = 3;
  Eigen::MatrixXd cell;
  std::vector<int> species;
  Eigen::MatrixXd positions;

  TwoBody desc;
  std::vector<CompactDescriptor *> dc;
  CompactStructure test_struc;
  DescriptorValues struc_desc;

  double cell_size = 10;
  double cutoff = cell_size / 2;

  TwoBodyTest() {
    // Make positions.
    cell = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    // positions = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;
    positions = Eigen::MatrixXd::Random(n_atoms, 3);

    // Make random species.
    for (int i = 0; i < n_atoms; i++) {
      species.push_back(rand() % n_species);
    }

    desc = TwoBody(cutoff, n_species);
    dc.push_back(&desc);

    test_struc = CompactStructure(cell, species, positions, cutoff, dc);
    struc_desc = test_struc.descriptors[0];

    std::cout << test_struc.n_neighbors << std::endl;
  }
};

TEST_F(TwoBodyTest, TwoBodyTest) {

}
