#include "descriptor.h"
#include "dot_product_kernel.h"
#include "omp.h"
#include "sparse_gp.h"
#include "three_body_kernel.h"
#include "two_body_kernel.h"
#include "gtest/gtest.h"
#include <chrono>
#include <stdio.h>

class SparseTest : public ::testing::Test {
public:
  // structure
  int n_atoms = 4;
  Eigen::MatrixXd cell{3, 3}, cell_2{3, 3};
  std::vector<int> species{0, 1, 0, 1};
  Eigen::MatrixXd positions{n_atoms, 3};
  StructureDescriptor test_struc;

  // labels
  Eigen::VectorXd energy{1}, forces{n_atoms * 3}, stresses{6};

  // descriptor
  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  std::vector<double> radial_hyps{0, 5}, radial_hyps_2{0, 3};
  std::vector<double> cutoff_hyps;
  std::vector<int> descriptor_settings{2, 3, 3};
  double cutoff = 5;
  std::vector<double> nested_cutoffs{5, 5};
  std::vector<double> many_body_cutoffs{5, 3};
  B2_Calculator desc1, desc2;
  std::vector<DescriptorCalculator *> calcs;

  // kernel
  double signal_variance = 1;
  double length_scale = 1;
  double power = 2;
  DotProductKernel kernel;
  TwoBodyKernel two_body_kernel;
  ThreeBodyKernel three_body_kernel;
  DotProductKernel many_body_kernel, many_body_kernel_2;
  std::vector<Kernel *> kernels;

  SparseTest() {
    cell << 100, 0, 0, 0, 100, 0, 0, 0, 100;

    positions = Eigen::MatrixXd::Random(n_atoms, 3);
    energy = Eigen::VectorXd::Random(1);
    forces = Eigen::VectorXd::Random(n_atoms * 3);
    stresses = Eigen::VectorXd::Random(6);

    desc1 = B2_Calculator(radial_string, cutoff_string, radial_hyps,
                          cutoff_hyps, descriptor_settings, 0);
    desc2 = B2_Calculator(radial_string, cutoff_string, radial_hyps_2,
                          cutoff_hyps, descriptor_settings, 1);
    calcs.push_back(&desc1);
    calcs.push_back(&desc2);

    test_struc = StructureDescriptor(cell, species, positions, cutoff,
                                     nested_cutoffs, many_body_cutoffs, calcs);
    test_struc.energy = energy;
    test_struc.forces = forces;
    test_struc.stresses = stresses;

    two_body_kernel = TwoBodyKernel(signal_variance, length_scale,
                                    cutoff_string, cutoff_hyps);
    three_body_kernel = ThreeBodyKernel(signal_variance, length_scale,
                                        cutoff_string, cutoff_hyps);
    many_body_kernel =
        DotProductKernel(signal_variance, power, 0);
    many_body_kernel_2 =
        DotProductKernel(signal_variance, power, 1);

    kernels = std::vector<Kernel *>{&many_body_kernel, &many_body_kernel_2};

    // kernels =
    //     std::vector<Kernel *> {&two_body_kernel, &three_body_kernel};
  }
};
