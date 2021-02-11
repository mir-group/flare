#ifndef KERNEL_H
#define KERNEL_H

#include "descriptor.h"
#include "structure.h"
#include <Eigen/Dense>
#include <vector>

class DescriptorValues;
class ClusterDescriptor;
class SparseGP;

class Kernel {
public:
  Eigen::VectorXd kernel_hyperparameters;

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

  std::vector<Eigen::MatrixXd> Kuu_grad(const ClusterDescriptor &envs,
                                        const Eigen::MatrixXd &Kuu,
                                        const Eigen::VectorXd &hyps);

  std::vector<Eigen::MatrixXd> Kuf_grad(const ClusterDescriptor &envs,
                                        const std::vector<Structure> &strucs,
                                        int kernel_index,
                                        const Eigen::MatrixXd &Kuf,
                                        const Eigen::VectorXd &hyps);

  virtual void set_hyperparameters(Eigen::VectorXd hyps) = 0;

  virtual ~Kernel() = default;
};

#endif
