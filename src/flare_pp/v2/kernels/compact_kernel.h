#ifndef COMPACT_KERNEL_H
#define COMPACT_KERNEL_H

#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;

// TODO: Rename to DotProductKernel.
class CompactKernel {
public:
  double sigma, sig2, power;

  CompactKernel();

  CompactKernel(double sigma, double power);

  Eigen::MatrixXd envs_envs(const ClusterDescriptor &envs1,
                            const ClusterDescriptor &envs2);

  Eigen::MatrixXd envs_struc(const ClusterDescriptor &envs,
                             const DescriptorValues &struc);

  Eigen::VectorXd self_kernel_struc(DescriptorValues struc);

  Eigen::MatrixXd struc_struc(DescriptorValues struc1,
                              DescriptorValues struc2);
};

#endif
