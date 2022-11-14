#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include "kernel.h"
#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;

class DotProduct : public Kernel {
public:
  double sigma, sig2, power;

  DotProduct();

  DotProduct(double sigma, double power);

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

  // Because of the simplicity of this kernel, Kuu_grad and Kuf_grad can
  // be significantly accelerated over the default implementation, which
  // reconstructs the covariance matrices from scratch.
  std::vector<Eigen::MatrixXd> Kuu_grad(const ClusterDescriptor &envs,
                                        const Eigen::MatrixXd &Kuu,
                                        const Eigen::VectorXd &new_hyps);

  std::vector<Eigen::MatrixXd> Kuf_grad(const ClusterDescriptor &envs,
                                        const std::vector<Structure> &strucs,
                                        int kernel_index,
                                        const Eigen::MatrixXd &Kuf,
                                        const Eigen::VectorXd &new_hyps);

  void set_hyperparameters(Eigen::VectorXd new_hyps);

  Eigen::MatrixXd compute_map_coeff_pow1(const SparseGP &gp_model,
                                         int kernel_index);
  Eigen::MatrixXd compute_map_coeff_pow2(const SparseGP &gp_model,
                                         int kernel_index);

  Eigen::MatrixXd compute_mapping_coefficients(const SparseGP &gp_model,
                                               int kernel_index);
  Eigen::MatrixXd compute_varmap_coefficients(const SparseGP &gp_model,
                                              int kernel_index);
  void write_info(std::ofstream &coeff_file);

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DotProduct,
    sigma, sig2, power, kernel_name, kernel_hyperparameters)

  nlohmann::json return_json();
};

#endif
