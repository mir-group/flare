#include "local_environment.h"
#include "structure.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#define THRESHOLD 1e-8

class EnvironmentTest : public ::testing ::Test {
public:
  Eigen::MatrixXd cell{3, 3};
  std::vector<int> species{0, 1, 2, 3, 4};
  Eigen::MatrixXd positions{5, 3};
  B2_Calculator desc1;
  std::vector<DescriptorCalculator *> descriptor_calculators;
  StructureDescriptor test_struc;
  int atom;
  LocalEnvironment test_env;

  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  std::vector<double> radial_hyps{0, 5};
  std::vector<double> cutoff_hyps;
  std::vector<int> descriptor_settings{5, 5, 5};
  int descriptor_index = 0;
  std::vector<double> nested_cutoffs{3, 3, 3};
  double cutoff = 3;
  std::vector<double> many_body_cutoffs{cutoff};

  EnvironmentTest() {
    cell << 1.3, 0.5, 0.8, -1.2, 1, 0.73, -0.8, 0.1, 0.9;

    positions << 1.2, 0.7, 2.3, 3.1, 2.5, 8.9, -1.8, -5.8, 3.0, 0.2, 1.1, 2.1,
        3.2, 1.1, 3.3;

    desc1 = B2_Calculator(radial_string, cutoff_string, radial_hyps,
                          cutoff_hyps, descriptor_settings, descriptor_index);
    descriptor_calculators.push_back(&desc1);
    test_struc = StructureDescriptor(cell, species, positions, cutoff,
                                     many_body_cutoffs, descriptor_calculators);

    atom = 0;
    test_env = LocalEnvironment(test_struc, atom, cutoff, nested_cutoffs,
                                descriptor_calculators);
    test_env.compute_descriptors_and_gradients();
  }
};

TEST_F(EnvironmentTest, SweepTest) {
  Eigen::MatrixXd cell(3, 3);
  Eigen::MatrixXd positions(5, 3);

  EXPECT_EQ(ceil(cutoff / test_struc.single_sweep_cutoff), test_env.sweep);

  // Check that the number of atoms in the local environment is correct.
  std::vector<int> env_ind, env_spec, unique_ind;
  std::vector<double> rs, xs, ys, zs, xrel, yrel, zrel;
  int sweep_val = test_env.sweep + 3;
  LocalEnvironment ::compute_environment(
      test_struc, test_env.noa, atom, cutoff, sweep_val, env_ind, env_spec,
      unique_ind, rs, xs, ys, zs, xrel, yrel, zrel);
  int expanded_count = rs.size();
  EXPECT_EQ(test_env.rs.size(), expanded_count);
  EXPECT_EQ(test_env.neighbor_list.size(), 5);

  // Check that the relative coordinates are computed correctly.
  for (int i = 0; i < test_env.rs.size(); i++) {
    EXPECT_EQ(test_env.xs[i] / test_env.rs[i], test_env.xrel[i]);
    EXPECT_EQ(test_env.ys[i] / test_env.rs[i], test_env.yrel[i]);
    EXPECT_EQ(test_env.zs[i] / test_env.rs[i], test_env.zrel[i]);
  }
}

TEST_F(EnvironmentTest, DotTest) {
  // Calculate the descriptor norm the old fashioned way.
  double norm_val = 0;
  double val_curr;
  int no_desc = test_env.descriptor_vals[0].rows();

  for (int i = 0; i < no_desc; i++) {
    val_curr = test_env.descriptor_vals[0](i);
    norm_val += val_curr * val_curr;
  }
  norm_val = sqrt(norm_val);
  EXPECT_NEAR(norm_val, test_env.descriptor_norm[0], THRESHOLD);
}

TEST_F(EnvironmentTest, NeighborTest) {
  test_env.compute_neighbor_descriptors();

  int n_desc = test_env.descriptor_vals.size();

  int neighbor = 3;
  int neighbor_ind = test_env.neighbor_list[neighbor];
  int force_comp = 2;
  int desc_el = 1000;

  // Check that descriptors agree.
  double val1 = test_env.neighbor_descriptors[neighbor][0](desc_el);
  double val2 =
      test_struc.local_environments[neighbor_ind].descriptor_vals[0](desc_el);

  EXPECT_EQ(val1, val2);

  // Check that descriptor derivatives agree.
  double val3 = test_env.neighbor_force_dervs[neighbor][0](force_comp, desc_el);
  double val4 =
      test_struc.local_environments[neighbor_ind].descriptor_force_dervs[0](
          3 * test_env.central_index + force_comp, desc_el);

  EXPECT_EQ(val3, val4);

  // std::cout << test_struc.local_environments[neighbor_ind].descriptor_norm[0]
  // << std::endl; std::cout << test_env.neighbor_descriptor_norms[neighbor][0]
  // << std::endl; std::cout << test_env.neighbor_force_dots[neighbor][0] <<
  // std::endl;
}

// TEST_F(EnvironmentTest, NestedTest){
//     NestedEnvironment nest =  NestedEnvironment(test_struc, 0, cutoff, 3, 2,
//     1); std::cout << nest.three_body_indices.size() << std::endl; std::cout
//     << nest.cross_bond_dists.size() << std::endl;
// }
