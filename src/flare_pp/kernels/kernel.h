#ifndef KERNEL_H
#define KERNEL_H

#include "descriptor.h"
#include "structure.h"
#include <Eigen/Dense>
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

class DescriptorValues;
class ClusterDescriptor;
class SparseGP;

class Kernel {
public:
  Eigen::VectorXd kernel_hyperparameters;
  std::string kernel_name;

  Kernel();

  Kernel(Eigen::VectorXd kernel_hyperparameters);

  virtual Eigen::MatrixXd envs_envs(const ClusterDescriptor &envs1,
                                    const ClusterDescriptor &envs2,
                                    const Eigen::VectorXd &hyps) = 0;

  virtual std::vector<Eigen::MatrixXd>
  envs_envs_grad(const ClusterDescriptor &envs1, const ClusterDescriptor &envs2,
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

  virtual Eigen::MatrixXd compute_mapping_coefficients(const SparseGP &gp_model,
                                                       int kernel_index) = 0;
  virtual Eigen::MatrixXd compute_varmap_coefficients(const SparseGP &gp_model,
                                                       int kernel_index) = 0;
  virtual void write_info(std::ofstream &coeff_file) = 0;

  virtual std::vector<Eigen::MatrixXd> Kuu_grad(const ClusterDescriptor &envs,
                                                const Eigen::MatrixXd &Kuu,
                                                const Eigen::VectorXd &hyps);

  virtual std::vector<Eigen::MatrixXd> Kuf_grad(const ClusterDescriptor &envs,
                                                const std::vector<Structure> &strucs,
                                                int kernel_index,
                                                const Eigen::MatrixXd &Kuf,
                                                const Eigen::VectorXd &hyps);

  virtual void set_hyperparameters(Eigen::VectorXd hyps) = 0;

  virtual ~Kernel() = default;

  virtual nlohmann::json return_json() = 0;
};

void to_json(nlohmann::json& j, const std::vector<Kernel*> & kernels);
void from_json(const nlohmann::json& j, std::vector<Kernel*> & kernels);

#endif
