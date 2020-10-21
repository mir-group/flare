#include "compact_structure.h"
#include "two_body.h"
#include "squared_exponential.h"
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

  std::string cutoff_name = "quadratic";
  std::vector<double> cutoff_hyps;

  TwoBody desc;
  std::vector<CompactDescriptor *> dc;
  CompactStructure test_struc;
  DescriptorValues struc_desc;
  ClusterDescriptor cluster_desc;

  double sigma = 2.0;
  double ls = 1.0;
  SquaredExponential kernel;

  double cell_size = 10;
  double cutoff = cell_size / 2;

  TwoBodyTest() {
    // Make positions.
    cell = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    // positions = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;
    positions = Eigen::MatrixXd::Random(n_atoms, 3);
    // positions << 0, 0, 0, 0, 0, 1;

    // Make random species.
    for (int i = 0; i < n_atoms; i++) {
      species.push_back(rand() % n_species);
    }

    desc = TwoBody(cutoff, n_species, cutoff_name, cutoff_hyps);
    dc.push_back(&desc);

    test_struc = CompactStructure(cell, species, positions, cutoff, dc);
    struc_desc = test_struc.descriptors[0];
    cluster_desc.add_cluster(struc_desc);

    kernel = SquaredExponential(sigma, ls);
  }
};

TEST_F(TwoBodyTest, TwoBodyTest) {
  Eigen::MatrixXd kern_mat = kernel.envs_envs(cluster_desc, cluster_desc);
  Eigen::MatrixXd envs_struc = kernel.envs_struc(cluster_desc, struc_desc);
  Eigen::MatrixXd struc_struc = kernel.struc_struc(struc_desc, struc_desc);

//   std::cout << kern_mat << std::endl;
//   std::cout << envs_struc << std::endl;
}
