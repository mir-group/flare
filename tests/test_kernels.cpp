#include "cutoffs.h"
#include "dot_product_kernel.h"
#include "kernels.h"
#include "local_environment.h"
#include "radial.h"
#include "structure.h"
#include "three_body_kernel.h"
#include "two_body_kernel.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#define THRESHOLD 1e-8

class KernelTest : public ::testing::Test {
public:
  // structure
  Eigen::MatrixXd cell{3, 3}, cell_2{3, 3}, cell_3{3, 3}, cell_4{3, 3};
  std::vector<int> species{0, 1, 0, 1, 0};
  Eigen::MatrixXd positions{5, 3}, positions_2{5, 3}, positions_3{5, 3},
      positions_4{5, 3};
  StructureDescriptor test_struc, test_struc_2, test_struc_3, test_struc_4;

  // environment
  Structure struc_bare_1, struc_bare_2;
  LocalEnvironment test_env, test_env_3, test_env_4, env_bare_1, env_bare_2;

  // descriptor
  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  std::vector<double> radial_hyps{0, 5};
  std::vector<double> cutoff_hyps;
  std::vector<int> descriptor_settings{2, 5, 5};
  double cutoff = 10;
  std::vector<double> nested_cutoffs{5, 5};
  std::vector<double> many_body_cutoffs{3};
  B2_Calculator desc1;
  std::vector<DescriptorCalculator *> descriptor_calculators;
  int descriptor_index = 0;

  // kernel
  double signal_variance = 2;
  double length_scale = 1;
  double power = 2;
  DotProductKernel kernel;
  TwoBodyKernel two_body_kernel;
  ThreeBodyKernel three_body_kernel;
  Eigen::VectorXd kern_vec, kern_vec_2, kern_vec_3;

  KernelTest() {
    cell << 10, 0, 0, 0, 10, 0, 0, 0, 10;

    positions << -0.68402216, 0.54343671, -0.52961224, 0.33045915, 0.40010388,
        0.59849816, -0.92832825, 0.06239221, 0.51601996, 0.75120489,
        -0.39606988, -0.34000017, -0.8242705, -0.73860995, 0.92679555;

    positions_2 << 0.69955637, -0.41619112, -0.51725003, 0.43189622, 0.88548458,
        -0.74495343, 0.31395126, -0.32179606, -0.35013419, 0.08793497,
        -0.70567732, -0.3811633, 0.35585787, -0.87190223, 0.06770428;

    desc1 = B2_Calculator(radial_string, cutoff_string, radial_hyps,
                          cutoff_hyps, descriptor_settings, 0);
    descriptor_calculators.push_back(&desc1);

    test_struc =
        StructureDescriptor(cell, species, positions, cutoff, nested_cutoffs,
                            many_body_cutoffs, descriptor_calculators);

    test_struc_2 =
        StructureDescriptor(cell, species, positions_2, cutoff, nested_cutoffs,
                            many_body_cutoffs, descriptor_calculators);
    test_env = test_struc_2.local_environments[0];

    // Construct "bare" structures, which don't contain descriptor vectors and
    // are therefore more memory-friendly.
    struc_bare_1 = Structure(cell, species, positions_2);
    struc_bare_2 = Structure(cell, species, positions);

    kernel = DotProductKernel(signal_variance, power, descriptor_index);
    two_body_kernel = TwoBodyKernel(signal_variance, length_scale,
                                    cutoff_string, cutoff_hyps);
    three_body_kernel = ThreeBodyKernel(signal_variance, length_scale,
                                        cutoff_string, cutoff_hyps);

    kern_vec = kernel.env_struc(test_env, test_struc);
    kern_vec_2 = two_body_kernel.env_struc(test_env, test_struc);
    kern_vec_3 = three_body_kernel.env_struc(test_env, test_struc);
  }
};

TEST_F(KernelTest, NormTest) {
  LocalEnvironment env1 = test_struc.local_environments[0];
  LocalEnvironment env2 = test_struc.local_environments[1];
  double kern_val = kernel.env_env(env1, env1);
  EXPECT_NEAR(kern_val, signal_variance * signal_variance, THRESHOLD);
}

TEST_F(KernelTest, EnvForceTest) {
  int atom1 = 0;
  env_bare_1 = LocalEnvironment(struc_bare_1, atom1, cutoff, many_body_cutoffs,
                                descriptor_calculators);

  // Compute descriptors of environment 1.
  env_bare_1.compute_descriptors();

  for (int atom2 = 0; atom2 < struc_bare_2.species.size(); atom2++) {
    env_bare_2 = LocalEnvironment(struc_bare_2, atom2, cutoff,
                                  many_body_cutoffs, descriptor_calculators);
    // Compute neighbor descriptors of environment 2.
    env_bare_2.compute_neighbor_descriptors();

    Eigen::VectorXd test_kern = kernel.env_env_force(env_bare_1, env_bare_2);

    for (int i = 0; i < 3; i++) {
      EXPECT_NEAR(test_kern(i), kern_vec(1 + 3 * atom2 + i), THRESHOLD);
    }
  }
}

TEST_F(KernelTest, ForceTest) {
  // Perturb the coordinates of the environment atoms.
  double thresh = 1e-6;
  int noa = 5;
  double delta = 1e-8;
  Eigen::VectorXd kern_pert;
  double fin_val, exact_val, abs_diff;
  for (int p = 0; p < noa; p++) {
    for (int m = 0; m < 3; m++) {
      positions_3 = positions;
      positions_3(p, m) += delta;
      test_struc_3 = StructureDescriptor(cell, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      kern_pert = kernel.env_struc(test_env, test_struc_3);
      fin_val = -(kern_pert(0) - kern_vec(0)) / delta;
      exact_val = kern_vec(1 + 3 * p + m);
      abs_diff = abs(fin_val - exact_val);

      EXPECT_NEAR(abs_diff, 0, thresh);
    }
  }
}

TEST_F(KernelTest, TwoBodyForceTest) {
  // Perturb the coordinates of the environment atoms.
  double thresh = 1e-5;
  int noa = 5;
  double delta = 1e-8;
  Eigen::VectorXd kern_pert;
  double fin_val, exact_val, abs_diff;
  for (int p = 0; p < noa; p++) {
    for (int m = 0; m < 3; m++) {
      positions_3 = positions;
      positions_3(p, m) += delta;
      test_struc_3 = StructureDescriptor(cell, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      kern_pert = two_body_kernel.env_struc(test_env, test_struc_3);
      fin_val = -(kern_pert(0) - kern_vec_2(0)) / delta;
      exact_val = kern_vec_2(1 + 3 * p + m);
      abs_diff = abs(fin_val - exact_val);
      EXPECT_NEAR(abs_diff, 0, thresh);
    }
  }
}

TEST_F(KernelTest, ThreeBodyForceTest) {
  double thresh = 1e-5;
  int noa = 5;
  double delta = 1e-8;
  Eigen::VectorXd kern_pert;
  double fin_val, exact_val, abs_diff;
  for (int p = 0; p < noa; p++) {
    for (int m = 0; m < 3; m++) {
      positions_3 = positions;
      positions_3(p, m) += delta;
      test_struc_3 = StructureDescriptor(cell, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      kern_pert = three_body_kernel.env_struc(test_env, test_struc_3);
      fin_val = -(kern_pert(0) - kern_vec_3(0)) / delta;
      exact_val = kern_vec_3(1 + 3 * p + m);
      abs_diff = abs(fin_val - exact_val);
      EXPECT_NEAR(abs_diff, 0, thresh);
    }
  }
}

TEST_F(KernelTest, StressTest) {
  double thresh = 1e-6;
  int noa = 5;
  double delta = 1e-8;
  Eigen::VectorXd kern_pert;
  double fin_val, exact_val, abs_diff;
  int stress_count = 0;
  // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_2 = cell;
      positions_3 = positions;

      // Perform strain.
      cell_2(0, m) += cell(0, n) * delta;
      cell_2(1, m) += cell(1, n) * delta;
      cell_2(2, m) += cell(2, n) * delta;
      for (int k = 0; k < noa; k++) {
        positions_3(k, m) += positions(k, n) * delta;
      }

      test_struc_3 = StructureDescriptor(cell_2, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      kern_pert = kernel.env_struc(test_env, test_struc_3);
      fin_val = -(kern_pert(0) - kern_vec(0)) / delta;
      exact_val = kern_vec(1 + 3 * noa + stress_count) * test_struc.volume;
      abs_diff = abs(fin_val - exact_val);

      EXPECT_NEAR(abs_diff, 0, thresh);

      stress_count++;
    }
  }
}

TEST_F(KernelTest, TwoBodyStressTest) {
  double thresh = 1e-5;
  int noa = 5;
  double delta = 1e-8;
  Eigen::VectorXd kern_pert;
  double fin_val, exact_val, abs_diff;
  int stress_count = 0;
  // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_2 = cell;
      positions_3 = positions;

      // Perform strain.
      cell_2(0, m) += cell(0, n) * delta;
      cell_2(1, m) += cell(1, n) * delta;
      cell_2(2, m) += cell(2, n) * delta;
      for (int k = 0; k < noa; k++) {
        positions_3(k, m) += positions(k, n) * delta;
      }

      test_struc_3 = StructureDescriptor(cell_2, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      kern_pert = two_body_kernel.env_struc(test_env, test_struc_3);
      fin_val = -(kern_pert(0) - kern_vec_2(0)) / delta;
      exact_val = kern_vec_2(1 + 3 * noa + stress_count) * test_struc.volume;
      abs_diff = abs(fin_val - exact_val);

      EXPECT_NEAR(abs_diff, 0, thresh);

      stress_count++;
    }
  }
}

TEST_F(KernelTest, ThreeBodyStressTest) {
  double thresh = 1e-5;
  int noa = 5;
  double delta = 1e-8;
  Eigen::VectorXd kern_pert;
  double fin_val, exact_val, abs_diff;
  int stress_count = 0;
  // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_2 = cell;
      positions_3 = positions;

      // Perform strain.
      cell_2(0, m) += cell(0, n) * delta;
      cell_2(1, m) += cell(1, n) * delta;
      cell_2(2, m) += cell(2, n) * delta;
      for (int k = 0; k < noa; k++) {
        positions_3(k, m) += positions(k, n) * delta;
      }

      test_struc_3 = StructureDescriptor(cell_2, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      kern_pert = three_body_kernel.env_struc(test_env, test_struc_3);
      fin_val = -(kern_pert(0) - kern_vec_3(0)) / delta;
      exact_val = kern_vec_3(1 + 3 * noa + stress_count) * test_struc.volume;
      abs_diff = abs(fin_val - exact_val);

      EXPECT_NEAR(abs_diff, 0, thresh);

      stress_count++;
    }
  }
}

TEST_F(KernelTest, TwoBodySelfTest) {
  // Perturb the coordinates of the environment atoms.
  Eigen::VectorXd self_kern = two_body_kernel.self_kernel_env(test_struc_2, 0);

  // Check energy/energy kernel.
  double en_val = two_body_kernel.env_env(test_env, test_env);
  EXPECT_EQ(en_val, self_kern[0]);

  // Check force/force kernel.
  double delta = 1e-5;
  double thresh = 1e-4;

  for (int n = 0; n < 3; n++) {
    positions_3 = positions_2;
    positions_3(0, n) += delta;
    positions_4 = positions_2;
    positions_4(0, n) -= delta;

    test_struc_3 =
        StructureDescriptor(cell, species, positions_3, cutoff, nested_cutoffs,
                            many_body_cutoffs, descriptor_calculators);
    test_env_3 = test_struc_3.local_environments[0];

    test_struc_4 =
        StructureDescriptor(cell, species, positions_4, cutoff, nested_cutoffs,
                            many_body_cutoffs, descriptor_calculators);
    test_env_4 = test_struc_4.local_environments[0];

    double calc1 = two_body_kernel.env_env(test_env_3, test_env_3);
    double calc2 = two_body_kernel.env_env(test_env_4, test_env_4);
    double calc3 = two_body_kernel.env_env(test_env_3, test_env_4);
    double calc4 = two_body_kernel.env_env(test_env_4, test_env_3);

    double kern_finite_diff =
        (calc1 + calc2 - calc3 - calc4) / (4 * delta * delta);

    EXPECT_NEAR(kern_finite_diff * 4, self_kern(1 + n), thresh);
  }

  // Check stress/stress kernel.
  int stress_count = 0;
  int noa = 5;
  double vol_sq = test_struc.volume * test_struc.volume;
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_3 = cell_4 = cell;
      positions_3 = positions_4 = positions_2;

      // Perform strain.
      cell_3(0, m) += cell(0, n) * delta;
      cell_3(1, m) += cell(1, n) * delta;
      cell_3(2, m) += cell(2, n) * delta;

      cell_4(0, m) -= cell(0, n) * delta;
      cell_4(1, m) -= cell(1, n) * delta;
      cell_4(2, m) -= cell(2, n) * delta;

      for (int k = 0; k < noa; k++) {
        positions_3(k, m) += positions_2(k, n) * delta;
        positions_4(k, m) -= positions_2(k, n) * delta;
      }

      test_struc_3 = StructureDescriptor(cell_3, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      test_env_3 = test_struc_3.local_environments[0];
      test_struc_4 = StructureDescriptor(cell_4, species, positions_4, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      test_env_4 = test_struc_4.local_environments[0];

      // Compute stress with finite difference.
      double calc1 = two_body_kernel.env_env(test_env_3, test_env_3);
      double calc2 = two_body_kernel.env_env(test_env_4, test_env_4);
      double calc3 = two_body_kernel.env_env(test_env_3, test_env_4);
      double calc4 = two_body_kernel.env_env(test_env_4, test_env_3);

      double kern_finite_diff =
          (calc1 + calc2 - calc3 - calc4) / (4 * delta * delta);

      // Check that the values match.
      double exact_val = self_kern(1 + 3 * noa + stress_count) * vol_sq;
      double abs_diff = abs(kern_finite_diff - exact_val);

      EXPECT_NEAR(abs_diff, 0, thresh);

      stress_count++;
    }
  }
}

// TEST_F(KernelTest, ThreeBodySelfTest){
//     // Perturb the coordinates of the environment atoms.
//     Eigen::VectorXd self_kern =
//         three_body_kernel.self_kernel_env(test_struc_2, 0);

//     // Check energy/energy kernel.
//     double en_val = three_body_kernel.env_env(test_env, test_env);
//     EXPECT_NEAR(en_val, self_kern[0], THRESHOLD);

//     // Check force/force kernel.
//     double delta = 1e-4;
//     double thresh = 1e-4;

//     for (int n = 0; n < 3; n ++){
//         positions_3 = positions_2;
//         positions_3(0, n) += delta;
//         positions_4 = positions_2;
//         positions_4(0, n) -= delta;

//         test_struc_3 = StructureDescriptor(cell, species,
//             positions_3, cutoff, nested_cutoffs, many_body_cutoffs,
//             descriptor_calculators);
//         test_env_3 = test_struc_3.local_environments[0];

//         test_struc_4 = StructureDescriptor(cell, species,
//             positions_4, cutoff, nested_cutoffs, many_body_cutoffs,
//             descriptor_calculators);
//         test_env_4 = test_struc_4.local_environments[0];

//         double calc1 = three_body_kernel.env_env(test_env_3, test_env_3);
//         double calc2 = three_body_kernel.env_env(test_env_4, test_env_4);
//         double calc3 = three_body_kernel.env_env(test_env_3, test_env_4);
//         double calc4 = three_body_kernel.env_env(test_env_4, test_env_3);

//         double kern_finite_diff = (calc1 + calc2 - calc3 - calc4) /
//             (4*delta*delta);

//         std::cout << (kern_finite_diff * 9 - self_kern(1 + n)) << std::endl;

//         // EXPECT_NEAR(kern_finite_diff * 4, self_kern(1 + n), thresh);
//     }

//     // Check stress/stress kernel.
//     int stress_count = 0;
//     int noa = 5;
//     double vol_sq = test_struc.volume * test_struc.volume;
//     for (int m = 0; m < 3; m ++){
//         for (int n = m; n < 3; n ++){
//             cell_3 = cell_4 = cell;
//             positions_3 = positions_4 = positions_2;

//             // Perform strain.
//             cell_3(0, m) += cell(0, n) * delta;
//             cell_3(1, m) += cell(1, n) * delta;
//             cell_3(2, m) += cell(2, n) * delta;

//             cell_4(0, m) -= cell(0, n) * delta;
//             cell_4(1, m) -= cell(1, n) * delta;
//             cell_4(2, m) -= cell(2, n) * delta;

//             for (int k = 0; k < noa; k ++){
//                 positions_3(k, m) += positions_2(k, n) * delta;
//                 positions_4(k, m) -= positions_2(k, n) * delta;
//             }

//             test_struc_3 = StructureDescriptor(cell_3, species, positions_3,
//                 cutoff, nested_cutoffs, many_body_cutoffs,
//                 descriptor_calculators);
//             test_env_3 = test_struc_3.local_environments[0];
//             test_struc_4 = StructureDescriptor(cell_4, species, positions_4,
//                 cutoff, nested_cutoffs, many_body_cutoffs,
//                 descriptor_calculators);
//             test_env_4 = test_struc_4.local_environments[0];

//             // Compute stress with finite difference.
//             double calc1 = three_body_kernel.env_env(test_env_3, test_env_3);
//             double calc2 = three_body_kernel.env_env(test_env_4, test_env_4);
//             double calc3 = three_body_kernel.env_env(test_env_3, test_env_4);
//             double calc4 = three_body_kernel.env_env(test_env_4, test_env_3);

//             double kern_finite_diff = (calc1 + calc2 - calc3 - calc4) /
//                 (4*delta*delta);

//             std::cout << 9 * kern_finite_diff / 4 << std::endl;

//             // // Check that the values match.
//             // double exact_val = self_kern(1 + 3 * noa + stress_count)
//             //     * vol_sq;
//             // double abs_diff = abs(kern_finite_diff - exact_val);

//             // EXPECT_NEAR(abs_diff, 0, thresh);

//             stress_count ++;
//         }
//     }

//     std::cout << "printing self kern:" << std::endl;
//     std::cout << self_kern << std::endl;
// }

TEST_F(KernelTest, TwoBodySelfStrucTest) {
  // Perturb the coordinates of the environment atoms.
  Eigen::VectorXd self_kern = two_body_kernel.self_kernel_struc(test_struc_2);

  // Check energy/energy kernel.
  double en_val = two_body_kernel.struc_struc_en(test_struc_2, test_struc_2);
  EXPECT_NEAR(en_val, self_kern[0], THRESHOLD);

  // Check force/force kernel.
  double delta = 1e-4;
  double thresh = 1e-4;

  for (int m = 0; m < test_struc_2.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions_2;
      positions_3(m, n) += delta;
      positions_4 = positions_2;
      positions_4(m, n) -= delta;

      test_struc_3 = StructureDescriptor(cell, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      test_struc_4 = StructureDescriptor(cell, species, positions_4, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      double calc1 = two_body_kernel.struc_struc_en(test_struc_3, test_struc_3);
      double calc2 = two_body_kernel.struc_struc_en(test_struc_4, test_struc_4);
      double calc3 = two_body_kernel.struc_struc_en(test_struc_3, test_struc_4);
      double calc4 = two_body_kernel.struc_struc_en(test_struc_4, test_struc_3);

      double kern_finite_diff =
          (calc1 + calc2 - calc3 - calc4) / (4 * delta * delta);

      EXPECT_NEAR(kern_finite_diff, self_kern(1 + 3 * m + n), thresh);
    }
  }

  // Check stress/stress kernel.
  int stress_count = 0;
  int noa = 5;
  double vol_sq = test_struc.volume * test_struc.volume;
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_3 = cell_4 = cell;
      positions_3 = positions_4 = positions_2;

      // Perform strain.
      cell_3(0, m) += cell(0, n) * delta;
      cell_3(1, m) += cell(1, n) * delta;
      cell_3(2, m) += cell(2, n) * delta;

      cell_4(0, m) -= cell(0, n) * delta;
      cell_4(1, m) -= cell(1, n) * delta;
      cell_4(2, m) -= cell(2, n) * delta;

      for (int k = 0; k < noa; k++) {
        positions_3(k, m) += positions_2(k, n) * delta;
        positions_4(k, m) -= positions_2(k, n) * delta;
      }

      test_struc_3 = StructureDescriptor(cell_3, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      test_struc_4 = StructureDescriptor(cell_4, species, positions_4, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      // Compute stress with finite difference.
      double calc1 = two_body_kernel.struc_struc_en(test_struc_3, test_struc_3);
      double calc2 = two_body_kernel.struc_struc_en(test_struc_4, test_struc_4);
      double calc3 = two_body_kernel.struc_struc_en(test_struc_3, test_struc_4);
      double calc4 = two_body_kernel.struc_struc_en(test_struc_4, test_struc_3);

      double kern_finite_diff =
          (calc1 + calc2 - calc3 - calc4) / (4 * delta * delta);

      // Check that the values match.
      double exact_val = self_kern(1 + 3 * noa + stress_count) * vol_sq;
      double abs_diff = abs(kern_finite_diff - exact_val);

      EXPECT_NEAR(abs_diff, 0, thresh);

      stress_count++;
    }
  }
}

TEST_F(KernelTest, ThreeBodySelfStrucTest) {
  // Perturb the coordinates of the environment atoms.
  Eigen::VectorXd self_kern = three_body_kernel.self_kernel_struc(test_struc_2);

  // Check energy/energy kernel.
  double en_val = three_body_kernel.struc_struc_en(test_struc_2, test_struc_2);

  EXPECT_NEAR(en_val, self_kern[0], THRESHOLD);

  // Check force/force kernel.
  double delta = 1e-4;
  double thresh = 1e-4;

  for (int m = 0; m < test_struc_2.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions_2;
      positions_3(m, n) += delta;
      positions_4 = positions_2;
      positions_4(m, n) -= delta;

      test_struc_3 = StructureDescriptor(cell, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      test_struc_4 = StructureDescriptor(cell, species, positions_4, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      double calc1 =
          three_body_kernel.struc_struc_en(test_struc_3, test_struc_3);
      double calc2 =
          three_body_kernel.struc_struc_en(test_struc_4, test_struc_4);
      double calc3 =
          three_body_kernel.struc_struc_en(test_struc_3, test_struc_4);
      double calc4 =
          three_body_kernel.struc_struc_en(test_struc_4, test_struc_3);

      double kern_finite_diff =
          (calc1 + calc2 - calc3 - calc4) / (4 * delta * delta);

      EXPECT_NEAR(kern_finite_diff, self_kern(1 + 3 * m + n), thresh);
    }
  }

  // Check stress/stress kernel.
  int stress_count = 0;
  int noa = 5;
  double vol_sq = test_struc.volume * test_struc.volume;
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_3 = cell_4 = cell;
      positions_3 = positions_4 = positions_2;

      // Perform strain.
      cell_3(0, m) += cell(0, n) * delta;
      cell_3(1, m) += cell(1, n) * delta;
      cell_3(2, m) += cell(2, n) * delta;

      cell_4(0, m) -= cell(0, n) * delta;
      cell_4(1, m) -= cell(1, n) * delta;
      cell_4(2, m) -= cell(2, n) * delta;

      for (int k = 0; k < noa; k++) {
        positions_3(k, m) += positions_2(k, n) * delta;
        positions_4(k, m) -= positions_2(k, n) * delta;
      }

      test_struc_3 = StructureDescriptor(cell_3, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      test_struc_4 = StructureDescriptor(cell_4, species, positions_4, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      // Compute stress with finite difference.
      double calc1 =
          three_body_kernel.struc_struc_en(test_struc_3, test_struc_3);
      double calc2 =
          three_body_kernel.struc_struc_en(test_struc_4, test_struc_4);
      double calc3 =
          three_body_kernel.struc_struc_en(test_struc_3, test_struc_4);
      double calc4 =
          three_body_kernel.struc_struc_en(test_struc_4, test_struc_3);

      double kern_finite_diff =
          (calc1 + calc2 - calc3 - calc4) / (4 * delta * delta);

      // Check that the values match.
      double exact_val = self_kern(1 + 3 * noa + stress_count) * vol_sq;
      double abs_diff = abs(kern_finite_diff - exact_val);

      EXPECT_NEAR(abs_diff, 0, thresh);

      stress_count++;
    }
  }
}

TEST_F(KernelTest, ManyBodySelfStrucTest) {
  // Perturb the coordinates of the environment atoms.
  Eigen::VectorXd self_kern = kernel.self_kernel_struc(test_struc_2);

  // std::cout << self_kern << std::endl;

  // Check energy/energy kernel.
  double en_val = kernel.struc_struc_en(test_struc_2, test_struc_2);

  EXPECT_NEAR(en_val, self_kern[0], THRESHOLD);

  // Check force/force kernel.
  double delta = 1e-4;
  double thresh = 1e-4;

  for (int m = 0; m < test_struc_2.noa; m++) {
    for (int n = 0; n < 3; n++) {
      positions_3 = positions_2;
      positions_3(m, n) += delta;
      positions_4 = positions_2;
      positions_4(m, n) -= delta;

      test_struc_3 = StructureDescriptor(cell, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      test_struc_4 = StructureDescriptor(cell, species, positions_4, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      double calc1 = kernel.struc_struc_en(test_struc_3, test_struc_3);
      double calc2 = kernel.struc_struc_en(test_struc_4, test_struc_4);
      double calc3 = kernel.struc_struc_en(test_struc_3, test_struc_4);
      double calc4 = kernel.struc_struc_en(test_struc_4, test_struc_3);

      double kern_finite_diff =
          (calc1 + calc2 - calc3 - calc4) / (4 * delta * delta);

      EXPECT_NEAR(kern_finite_diff, self_kern(1 + 3 * m + n), thresh);
    }
  }

  // Check stress/stress kernel.
  int stress_count = 0;
  int noa = 5;
  double vol_sq = test_struc.volume * test_struc.volume;
  for (int m = 0; m < 3; m++) {
    for (int n = m; n < 3; n++) {
      cell_3 = cell_4 = cell;
      positions_3 = positions_4 = positions_2;

      // Perform strain.
      cell_3(0, m) += cell(0, n) * delta;
      cell_3(1, m) += cell(1, n) * delta;
      cell_3(2, m) += cell(2, n) * delta;

      cell_4(0, m) -= cell(0, n) * delta;
      cell_4(1, m) -= cell(1, n) * delta;
      cell_4(2, m) -= cell(2, n) * delta;

      for (int k = 0; k < noa; k++) {
        positions_3(k, m) += positions_2(k, n) * delta;
        positions_4(k, m) -= positions_2(k, n) * delta;
      }

      test_struc_3 = StructureDescriptor(cell_3, species, positions_3, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);
      test_struc_4 = StructureDescriptor(cell_4, species, positions_4, cutoff,
                                         nested_cutoffs, many_body_cutoffs,
                                         descriptor_calculators);

      // Compute stress with finite difference.
      double calc1 = kernel.struc_struc_en(test_struc_3, test_struc_3);
      double calc2 = kernel.struc_struc_en(test_struc_4, test_struc_4);
      double calc3 = kernel.struc_struc_en(test_struc_3, test_struc_4);
      double calc4 = kernel.struc_struc_en(test_struc_4, test_struc_3);

      double kern_finite_diff =
          (calc1 + calc2 - calc3 - calc4) / (4 * delta * delta);

      // Check that the values match.
      double exact_val = self_kern(1 + 3 * noa + stress_count) * vol_sq;
      double abs_diff = abs(kern_finite_diff - exact_val);

      EXPECT_NEAR(abs_diff, 0, thresh);

      stress_count++;
    }
  }
}

TEST(MatTest, MatTest){
    std::cout << "hello world" << std::endl;
    Eigen::VectorXd test(3);
    test << 1, 2, 3;
    std::cout << test << std::endl;
}