#ifndef KERNEL_H
#define KERNEL_H

#include "compact_descriptor.h"
#include "compact_structure.h"
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
                                    const ClusterDescriptor &envs2,
                                    const Eigen::VectorXd &hyps) = 0;

  virtual std::vector<Eigen::MatrixXd>
  envs_envs_grad(const ClusterDescriptor &envs1,
                 const ClusterDescriptor &envs2,
                 const Eigen::VectorXd &hyps) = 0;

  virtual Eigen::MatrixXd envs_struc(const ClusterDescriptor &envs,
                                     const DescriptorValues &struc,
                                     const Eigen::VectorXd &hyps) = 0;

  virtual std::vector<Eigen::MatrixXd>
  envs_struc_grad(const ClusterDescriptor &envs, const DescriptorValues &struc,
                  const Eigen::VectorXd &hyps) = 0;

  virtual Eigen::VectorXd self_kernel_struc(const DescriptorValues &struc,
                                            const Eigen::VectorXd &hyps) = 0;

  virtual Eigen::MatrixXd struc_struc(const DescriptorValues &struc1,
                                      const DescriptorValues &struc2,
                                      const Eigen::VectorXd &hyps) = 0;

  std::vector<Eigen::MatrixXd>
  Kuu_grad(const ClusterDescriptor &envs, const Eigen::MatrixXd &Kuu,
           const Eigen::VectorXd &hyps);

  std::vector<Eigen::MatrixXd>
  Kuf_grad(const ClusterDescriptor &envs,
           const std::vector<CompactStructure> &strucs, int kernel_index,
           const Eigen::MatrixXd &Kuf, const Eigen::VectorXd &hyps);

  std::vector<Eigen::MatrixXd>
  kernel_transform(const ClusterDescriptor &sparse_descriptors,
                   const std::vector<CompactStructure> &training_structures,
                   int kernel_index, Eigen::VectorXd hyps);

  virtual void set_hyperparameters(Eigen::VectorXd hyps) = 0;

  virtual ~CompactKernel() = default;
};

#endif
