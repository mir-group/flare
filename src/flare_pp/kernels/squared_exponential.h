#ifndef SQUARED_EXPONENTIAL_H
#define SQUARED_EXPONENTIAL_H

#include "kernel.h"
#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;

class SquaredExponential : public Kernel {
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

  Eigen::MatrixXd compute_mapping_coefficients(const SparseGP &gp_model,
                                               int kernel_index);
  Eigen::MatrixXd compute_varmap_coefficients(const SparseGP &gp_model,
                                               int kernel_index);
  void write_info(std::ofstream &coeff_file);

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SquaredExponential, sigma, ls, sig2, ls2,
    kernel_name)

  nlohmann::json return_json();
};

#endif
