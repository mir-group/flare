#ifndef KERNEL_H
#define KERNEL_H

#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;

class CompactKernel {
public:
  Eigen::VectorXd kernel_hyperparameters;

  CompactKernel();

  CompactKernel(Eigen::VectorXd kernel_hyperparameters);

  virtual Eigen::MatrixXd envs_envs(const ClusterDescriptor &envs1,
                                    const ClusterDescriptor &envs2) = 0;

  virtual Eigen::MatrixXd envs_struc(const ClusterDescriptor &envs,
                                     const DescriptorValues &struc) = 0;

  virtual Eigen::VectorXd self_kernel_struc(DescriptorValues struc) = 0;

  virtual Eigen::MatrixXd struc_struc(DescriptorValues struc1,
                                      DescriptorValues struc2) = 0;

  virtual ~CompactKernel() = default;
};

#endif
