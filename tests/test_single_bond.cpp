#include "compact_structure.h"
#include "cutoffs.h"
#include "descriptor.h"
#include "local_environment.h"
#include "radial.h"
#include "single_bond.h"
#include "structure.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>

class BondEnv : public ::testing::Test {
protected:
  double delta = 1e-8;
  int noa = 5;
  int nos = 5;

  Eigen::MatrixXd cell{3, 3}, cell_2{3, 3};
  std::vector<int> species{0, 2, 1, 3, 4};
  Eigen::MatrixXd positions_1{noa, 3}, positions_2{noa, 3};

  Structure struc1, struc2;
  LocalEnvironment env1, env2;

  double rcut = 3;
  std::vector<double> many_body_cutoffs{rcut};

  // Prepare cutoff.
  std::vector<double> cutoff_hyps;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function = cos_cutoff;

  // Prepare spherical harmonics.
  int lmax = 10;
  int number_of_harmonics = (lmax + 1) * (lmax + 1);

  // Prepare radial basis set.
  double sigma = 1;
  double first_gauss = 0;
  double final_gauss = 3;
  int N = 10;
  std::vector<double> radial_hyps = {first_gauss, final_gauss};
  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      basis_function = chebyshev;

  // Initialize matrices.
  int no_descriptors = nos * N * number_of_harmonics;
  Eigen::VectorXd single_bond_vals, single_bond_vals_2;
  Eigen::MatrixXd force_dervs, force_dervs_2, stress_dervs, stress_dervs_2;

  // Prepare descriptor calculator.
  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  std::vector<int> descriptor_settings{nos, N, lmax};
  int descriptor_index = 0;
  B2_Calculator descriptor;
  std::vector<DescriptorCalculator *> descriptors;

  CompactStructure compact_struc;

  BondEnv() {
    // Create arbitrary structure.
    cell << 13, 0.5, 0.8, -1.2, 10, 0.73, -0.8, 0.1, 9;

    positions_1 << 1.2, 0.7, 2.3, 3.1, 2.5, 8.9, -1.8, -5.8, 3.0, 0.2, 1.1, 2.1,
        3.2, 1.1, 3.3;

    struc1 = Structure(cell, species, positions_1);
    env1 = LocalEnvironment(struc1, 0, rcut);
    env1.many_body_cutoffs = many_body_cutoffs;
    env1.compute_indices();

    single_bond_vals = Eigen::VectorXd::Zero(no_descriptors);
    force_dervs = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
    stress_dervs = Eigen::MatrixXd::Zero(6, no_descriptors);

    // Create descriptor calculator.
    descriptor =
        B2_Calculator(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
                      descriptor_settings, descriptor_index);
    descriptors.push_back(&descriptor);

    // Create compact structure.
    double compact_cut = 5.0;
    compact_struc =
        CompactStructure(cell, species, positions_1, compact_cut, &descriptor);
  }
};

TEST(LammpsCheck, LammpsCheck) {
  // Prints single bond values and derivatives for comparison with the
  // LAMMPS implementation.

  double rcut = 5;
  int lmax = 3;
  int N = 5;
  int nos = 2;

  Eigen::MatrixXd cell{3, 3};
  std::vector<int> species{0, 0, 1, 0};
  cell << 100, 0, 0, 0, 100, 0, 0, 0, 100;

  int noa = 4;
  Eigen::MatrixXd positions{noa, 3};
  positions << 0, 0, 0, 0.13, 0.76, 0.31, -0.39, 0.32, 0.18, 0.76, -0.52, 0.22;

  Structure struc = Structure(cell, species, positions);
  LocalEnvironment env = LocalEnvironment(struc, 0, rcut);
  std::vector<double> many_body_cutoffs{rcut};
  env.many_body_cutoffs = many_body_cutoffs;
  env.compute_indices();

  Eigen::VectorXd single_bond_vals;
  Eigen::MatrixXd force_dervs, stress_dervs;

  int number_of_harmonics = (lmax + 1) * (lmax + 1);
  int no_descriptors = nos * N * number_of_harmonics;

  single_bond_vals = Eigen::VectorXd::Zero(no_descriptors);
  force_dervs = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
  stress_dervs = Eigen::MatrixXd::Zero(6, no_descriptors);

  std::vector<double> cutoff_hyps;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function = quadratic_cutoff;

  std::vector<double> radial_hyps = {0, rcut};
  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      basis_function = chebyshev;

  single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
                      basis_function, cutoff_function, env, 0, N, lmax,
                      radial_hyps, cutoff_hyps);

  //   std::cout << "Positions:" << std::endl;
  //   std::cout << positions << std::endl;

  //   std::cout << "Cell:" << std::endl;
  //   std::cout << cell << std::endl;

  //   std::cout << "Single bond vals:" << std::endl;
  //   std::cout << single_bond_vals << std::endl;

  //   std::cout << "Force derivatives:" << std::endl;
  //   std::cout << force_dervs.row(0) << std::endl;
}

TEST_F(BondEnv, CentTest) {
  single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
                      basis_function, cutoff_function, env1, 0, N, lmax,
                      radial_hyps, cutoff_hyps);

  // Perturb the coordinates of the central atom.
  for (int m = 0; m < 3; m++) {
    positions_2 = positions_1;
    positions_2(0, m) += delta;
    struc2 = Structure(cell, species, positions_2);
    env2 = LocalEnvironment(struc2, 0, rcut);
    env2.many_body_cutoffs = many_body_cutoffs;
    env2.compute_indices();

    // Initialize matrices.
    single_bond_vals_2 = Eigen::VectorXd::Zero(no_descriptors);
    force_dervs_2 = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
    stress_dervs_2 = Eigen::MatrixXd::Zero(6, no_descriptors);

    single_bond_sum_env(single_bond_vals_2, force_dervs_2, stress_dervs_2,
                        basis_function, cutoff_function, env2, 0, N, lmax,
                        radial_hyps, cutoff_hyps);

    double finite_diff, exact, diff;
    double tolerance = 5e-6;

    // Check derivatives.
    for (int n = 0; n < single_bond_vals.rows(); n++) {
      finite_diff = (single_bond_vals_2[n] - single_bond_vals[n]) / delta;
      exact = force_dervs(m, n);
      diff = abs(finite_diff - exact);
      EXPECT_LE(diff, tolerance);
    }
  }
}

TEST_F(BondEnv, EnvTest) {
  single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
                      basis_function, cutoff_function, env1, 0, N, lmax,
                      radial_hyps, cutoff_hyps);

  double finite_diff, exact, diff;
  double tolerance = 1e-5;

  // Perturb the coordinates of the environment atoms.
  for (int p = 1; p < noa; p++) {
    for (int m = 0; m < 3; m++) {
      positions_2 = positions_1;
      positions_2(p, m) += delta;
      struc2 = Structure(cell, species, positions_2);
      env2 = LocalEnvironment(struc2, 0, rcut);
      env2.many_body_cutoffs = many_body_cutoffs;
      env2.compute_indices();

      // Initialize matrices.
      single_bond_vals_2 = Eigen::VectorXd::Zero(no_descriptors);
      force_dervs_2 = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
      stress_dervs_2 = Eigen::MatrixXd::Zero(6, no_descriptors);

      single_bond_sum_env(single_bond_vals_2, force_dervs_2, stress_dervs_2,
                          basis_function, cutoff_function, env2, 0, N, lmax,
                          radial_hyps, cutoff_hyps);

      // Check derivatives.
      for (int n = 0; n < single_bond_vals.rows(); n++) {
        finite_diff = (single_bond_vals_2[n] - single_bond_vals[n]) / delta;
        exact = force_dervs(p * 3 + m, n);
        diff = abs(finite_diff - exact);
        EXPECT_LE(diff, tolerance);
      }
    }
  }
}

TEST_F(BondEnv, StressTest) {
  int stress_ind = 0;
  double finite_diff, exact, diff;
  double tolerance = 1e-5;

  single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
                      basis_function, cutoff_function, env1, 0, N, lmax,
                      radial_hyps, cutoff_hyps);

  // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_2 = cell;
      positions_2 = positions_1;

      // Perform strain.
      cell_2(0, m) += cell(0, n) * delta;
      cell_2(1, m) += cell(1, n) * delta;
      cell_2(2, m) += cell(2, n) * delta;
      for (int k = 0; k < noa; k++) {
        positions_2(k, m) += positions_1(k, n) * delta;
      }

      struc2 = Structure(cell_2, species, positions_2);
      env2 = LocalEnvironment(struc2, 0, rcut);
      env2.many_body_cutoffs = many_body_cutoffs;
      env2.compute_indices();

      single_bond_vals_2 = Eigen::VectorXd::Zero(no_descriptors);
      force_dervs_2 = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
      stress_dervs_2 = Eigen::MatrixXd::Zero(6, no_descriptors);

      // Calculate descriptors.
      single_bond_sum_env(single_bond_vals_2, force_dervs_2, stress_dervs_2,
                          basis_function, cutoff_function, env2, 0, N, lmax,
                          radial_hyps, cutoff_hyps);

      // Check stress derivatives.
      for (int p = 0; p < single_bond_vals.rows(); p++) {
        finite_diff = (single_bond_vals_2[p] - single_bond_vals[p]) / delta;
        exact = stress_dervs(stress_ind, p);
        diff = abs(finite_diff - exact);
        EXPECT_LE(diff, tolerance);
      }

      stress_ind++;
    }
  }
}

TEST_F(BondEnv, StrucTest) {

  Eigen::MatrixXd single_bond_vals_struc, force_dervs_struc, stress_dervs_struc,
    neighbor_coordinates;
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, descriptor_indices;

  auto start = std::chrono::steady_clock::now();
  single_bond_sum_struc(single_bond_vals_struc, force_dervs_struc,
                        stress_dervs_struc, neighbor_coordinates,
                        neighbor_count, cumulative_neighbor_count,
                        descriptor_indices, compact_struc);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << elapsed_seconds.count() << "s\n";

  std::cout << force_dervs_struc.rows() << std::endl;

  single_bond_sum_env(single_bond_vals, force_dervs, stress_dervs,
                      basis_function, cutoff_function, env1, 0, N, lmax,
                      radial_hyps, cutoff_hyps);

  // Check that the single bond values match.
  double tolerance = 1e-16;
  for (int i = 0; i < single_bond_vals.size(); i++) {
    EXPECT_EQ(single_bond_vals_struc(0, i), single_bond_vals(i));
  }

  // Check that the stress values match.
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < single_bond_vals.size(); j++) {
      EXPECT_EQ(stress_dervs_struc(i, j), stress_dervs(i, j));
    }
  }
}

TEST_F(BondEnv, BigStruc) {
  int n_atoms = 400;
  cell << 10, 0, 0, 0, 10, 0, 0, 0, 10;
  Eigen::MatrixXd positions = Eigen::MatrixXd::Random(n_atoms, 3) * 10;
  std::vector<int> species(n_atoms, 0);
  double compact_cut = 3;

  // Prepare descriptor calculator.
  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  int N = 10;
  int lmax = 10;
  std::vector<int> descriptor_settings{1, N, lmax};
  std::vector<double> radial_hyps = {0, compact_cut};
  int descriptor_index = 0;
  B2_Calculator descriptor;
  std::vector<DescriptorCalculator *> descriptors;
  descriptor =
      B2_Calculator(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
                    descriptor_settings, descriptor_index);
  descriptors.push_back(&descriptor);

  compact_struc =
      CompactStructure(cell, species, positions, compact_cut, &descriptor);

  Eigen::MatrixXd single_bond_vals_struc, force_dervs_struc,
    stress_dervs_struc, neighbor_coordinates;
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, descriptor_indices;

  auto start = std::chrono::steady_clock::now();
  single_bond_sum_struc(single_bond_vals_struc, force_dervs_struc,
                        stress_dervs_struc, neighbor_coordinates,
                        neighbor_count, cumulative_neighbor_count,
                        descriptor_indices, compact_struc);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << elapsed_seconds.count() << "s\n";

  // std::cout << force_dervs_struc.rows() << std::endl;
  // std::cout << neighbor_count << std::endl;
}
