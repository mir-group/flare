#ifndef DOT_PRODUCT_KERNEL_H
#define DOT_PRODUCT_KERNEL_H

#include "kernels.h"
#include <Eigen/Dense>
#include <vector>

class LocalEnvironment;
class StructureDescriptor;

class DotProductKernel : public Kernel {
public:
  double sigma, sig2, power;
  int descriptor_index;

  DotProductKernel();

  DotProductKernel(double sigma, double power, int descriptor_index);

  double env_env(const LocalEnvironment &env1, const LocalEnvironment &env2);

  Eigen::VectorXd env_env_force(const LocalEnvironment &env1,
                                const LocalEnvironment &env2);

  Eigen::VectorXd self_kernel_env(const StructureDescriptor &struc1, int atom);

  Eigen::VectorXd self_kernel_struc(const StructureDescriptor &struc);

  Eigen::VectorXd env_struc_partial(const LocalEnvironment &env1,
                                    const StructureDescriptor &struc1,
                                    int atom);

  Eigen::VectorXd env_struc(const LocalEnvironment &env1,
                            const StructureDescriptor &struc1);

  Eigen::MatrixXd kernel_transform(Eigen::MatrixXd kernels,
                                   std::vector<double> new_hyps);
};

#endif
