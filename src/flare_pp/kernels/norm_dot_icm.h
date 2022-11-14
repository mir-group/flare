#ifndef NORMALIZED_DOT_PRODUCT_ICM_H
#define NORMALIZED_DOT_PRODUCT_ICM_H

#include "kernel.h"
#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;

class NormalizedDotProduct_ICM : public Kernel {
public:
  double sigma, sig2, power;
  int no_types, n_icm_coeffs;
  Eigen::MatrixXd icm_coeffs;

  NormalizedDotProduct_ICM();

  NormalizedDotProduct_ICM(double sigma, double power,
                           Eigen::MatrixXd icm_coeffs);

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

  void set_hyperparameters(Eigen::VectorXd new_hyps);

  Eigen::MatrixXd compute_mapping_coefficients(const SparseGP &gp_model,
                                               int kernel_index);
  Eigen::MatrixXd compute_varmap_coefficients(const SparseGP &gp_model,
                                               int kernel_index);
  void write_info(std::ofstream &coeff_file);

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(NormalizedDotProduct_ICM,
    sigma, sig2, power, no_types, n_icm_coeffs, icm_coeffs, kernel_name)

  nlohmann::json return_json();
};

int get_icm_index(int s1, int s2, int n_types);

#endif
