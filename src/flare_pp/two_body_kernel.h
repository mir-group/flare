#ifndef TWO_BODY_KERNEL_H
#define TWO_BODY_KERNEL_H

#include "kernels.h"
#include <Eigen/Dense>
#include <vector>

class LocalEnvironment;
class StructureDescriptor;

class TwoBodyKernel : public Kernel {
public:
  double sigma, sig2, ls, ls1, ls2, ls3;
  void (*cutoff_pointer)(double *, double, double, std::vector<double>);
  std::vector<double> cutoff_hyps;

  TwoBodyKernel();

  TwoBodyKernel(double sigma, double ls, const std::string &cutoff_function,
                std::vector<double> cutoff_hyps);

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
                                   Eigen::VectorXd new_hyps);

  std::vector<Eigen::MatrixXd> kernel_gradient(Eigen::MatrixXd kernels,
                                               Eigen::VectorXd new_hyps);

  void set_hyperparameters(Eigen::VectorXd new_hyps){
      // Not implemented.
  };
};

double force_helper(double rel1_rel2, double diff_rel1, double diff_rel2,
                    double diff_sq, double fi, double fj, double fdi,
                    double fdj, double l1, double l2, double l3, double s2);

#endif