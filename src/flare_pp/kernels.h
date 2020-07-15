#ifndef KERNELS_H
#define KERNELS_H

#include <Eigen/Dense>
#include <vector>

class LocalEnvironment;
class StructureDescriptor;

class Kernel {
public:
  std::vector<double> kernel_hyperparameters;

  Kernel();

  Kernel(std::vector<double> kernel_hyperparameters);

  virtual double env_env(const LocalEnvironment &env1,
                         const LocalEnvironment &env2) = 0;

  virtual Eigen::VectorXd env_env_force(const LocalEnvironment &env1,
                                        const LocalEnvironment &env2) = 0;

  double struc_struc_en(const StructureDescriptor &struc1,
                        const StructureDescriptor &struc2);

  virtual Eigen::VectorXd self_kernel_env(const StructureDescriptor &struc1,
                                          int atom) = 0;

  virtual Eigen::VectorXd env_struc_partial(const LocalEnvironment &env1,
                                            const StructureDescriptor &struc1,
                                            int atom) = 0;

  virtual Eigen::VectorXd env_struc(const LocalEnvironment &env1,
                                    const StructureDescriptor &struc1) = 0;

  virtual Eigen::VectorXd
    self_kernel_struc(const StructureDescriptor &struc) = 0;

  virtual ~Kernel() = default;
};

#endif
