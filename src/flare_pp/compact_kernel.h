#ifndef COMPACT_KERNEL_H
#define COMPACT_KERNEL_H

#include <Eigen/Dense>
#include <vector>

class CompactStructure;

class CompactKernel{
public:
  double sigma, sig2, power;

  CompactKernel();

  CompactKernel(double sigma, double power);

  double envs_envs(const std::vector<Eigen::MatrixXd> &envs1,
                   const std::vector<Eigen::MatrixXd> &envs2);

  Eigen::VectorXd envs_struc(const std::vector<Eigen::MatrixXd> &envs1,
                             const CompactStructure &struc);

  Eigen::VectorXd self_kernel_struc(const CompactStructure &struc);
};

#endif
