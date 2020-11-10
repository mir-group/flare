#ifndef KERNEL_H
#define KERNEL_H

#include <Eigen/Dense>
#include <vector>
#include "compact_structure.h"
#include "compact_descriptor.h"

class DescriptorValues;
class ClusterDescriptor;

class CompactKernel {
public:
  Eigen::VectorXd kernel_hyperparameters;

  CompactKernel();

  CompactKernel(Eigen::VectorXd kernel_hyperparameters);

  virtual Eigen::MatrixXd envs_envs(const ClusterDescriptor &envs1,
                                    const ClusterDescriptor &envs2) = 0;

  virtual Eigen::MatrixXd envs_envs_grad(const ClusterDescriptor &envs1,
                                         const ClusterDescriptor &envs2,
                                         const Eigen::MatrixXd &Kuu) = 0;

  virtual Eigen::MatrixXd envs_struc(const ClusterDescriptor &envs,
                                     const DescriptorValues &struc) = 0;

  virtual Eigen::VectorXd self_kernel_struc(DescriptorValues struc) = 0;

  virtual Eigen::MatrixXd struc_struc(DescriptorValues struc1,
                                      DescriptorValues struc2) = 0;

  std::vector<Eigen::MatrixXd> kernel_transform(
    const ClusterDescriptor &sparse_descriptors,
    const std::vector<CompactStructure> &training_structures,
    int kernel_index, Eigen::VectorXd new_hyps);

  virtual void set_hyperparameters(Eigen::VectorXd new_hyps) = 0;

  virtual ~CompactKernel() = default;
};

#endif
