#ifndef SQUARED_EXPONENTIAL_H
#define SQUARED_EXPONENTIAL_H

#include "kernel.h"
#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;

class SquaredExponential : public CompactKernel {
public:
  double sigma, ls, sig2, ls2;

  SquaredExponential();

  SquaredExponential(double sigma, double ls);

  Eigen::MatrixXd envs_envs(const ClusterDescriptor &envs1,
                            const ClusterDescriptor &envs2,
                            const Eigen::VectorXd &hyps);

  std::vector<Eigen::MatrixXd> envs_envs_grad(const ClusterDescriptor &envs1,
                                              const ClusterDescriptor &envs2,
                                              const Eigen::VectorXd &hyps);

  Eigen::MatrixXd envs_struc(const ClusterDescriptor &envs,
                             const DescriptorValues &struc,
                             const Eigen::VectorXd &hyps);

  std::vector<Eigen::MatrixXd> envs_struc_grad(const ClusterDescriptor &envs,
                                               const DescriptorValues &struc,
                                               const Eigen::VectorXd &hyps);

  Eigen::VectorXd self_kernel_struc(const DescriptorValues &struc,
                                    const Eigen::VectorXd &hyps);

  Eigen::MatrixXd struc_struc(const DescriptorValues &struc1,
                              const DescriptorValues &struc2,
                              const Eigen::VectorXd &hyps);

  void set_hyperparameters(Eigen::VectorXd hyps);
};

#endif
