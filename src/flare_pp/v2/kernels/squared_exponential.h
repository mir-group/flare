#ifndef SQUARED_EXPONENTIAL_H
#define SQUARED_EXPONENTIAL_H

#include "kernel.h"
#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;

// TODO: Make this an abstract class.
class SquaredExponential : public CompactKernel {
public:
  double sigma, ls;

  SquaredExponential();

  SquaredExponential(double sigma, double ls);

  Eigen::MatrixXd envs_envs(const ClusterDescriptor &envs1,
                            const ClusterDescriptor &envs2);

  Eigen::MatrixXd envs_struc(const ClusterDescriptor &envs,
                             const DescriptorValues &struc);

  Eigen::VectorXd self_kernel_struc(DescriptorValues struc);

  Eigen::MatrixXd struc_struc(DescriptorValues struc1,
                              DescriptorValues struc2);
};

#endif
