#ifndef NORMALIZED_DOT_PRODUCT_H
#define NORMALIZED_DOT_PRODUCT_H

#include "kernel.h"
#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;

// TODO: Make this an abstract class.
class NormalizedDotProduct : public CompactKernel {
public:
  double sigma, sig2, power;

  NormalizedDotProduct();

  NormalizedDotProduct(double sigma, double power);

  Eigen::MatrixXd envs_envs(const ClusterDescriptor &envs1,
                            const ClusterDescriptor &envs2);

  Eigen::MatrixXd envs_envs_grad(const ClusterDescriptor &envs1,
                                 const ClusterDescriptor &envs2,
                                 const Eigen::MatrixXd &Kuu);

  Eigen::MatrixXd envs_struc(const ClusterDescriptor &envs,
                             const DescriptorValues &struc);

  Eigen::VectorXd self_kernel_struc(DescriptorValues struc);

  Eigen::MatrixXd struc_struc(DescriptorValues struc1,
                              DescriptorValues struc2);

  void set_hyperparameters(Eigen::VectorXd new_hyps);
};

#endif
