#ifndef COMPACT_KERNEL_H
#define COMPACT_KERNEL_H

#include <Eigen/Dense>
#include <vector>

class CompactStructure;
class CompactEnvironments;

class CompactKernel{
public:
  double sigma, sig2, power;

  CompactKernel();

  CompactKernel(double sigma, double power);

  Eigen::MatrixXd envs_envs(const CompactEnvironments &envs1,
                            const CompactEnvironments &envs2);

  Eigen::MatrixXd envs_struc(const CompactEnvironments &envs,
                             const CompactStructure &struc);

  Eigen::VectorXd self_kernel_struc(const CompactStructure &struc);
};

#endif
