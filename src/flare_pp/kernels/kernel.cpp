#include "kernel.h"
#include "cutoffs.h"
#include <cmath>
#include <iostream>

CompactKernel ::CompactKernel(){};
CompactKernel ::CompactKernel(Eigen::VectorXd kernel_hyperparameters) {
  this->kernel_hyperparameters = kernel_hyperparameters;
};

std::vector<Eigen::MatrixXd>
CompactKernel ::Kuu_grad(const ClusterDescriptor &envs,
                         const Eigen::MatrixXd &Kuu,
                         const Eigen::VectorXd &hyps) {

  std::vector<Eigen::MatrixXd> Kuu_grad = envs_envs_grad(envs, envs, hyps);

  return Kuu_grad;
}

std::vector<Eigen::MatrixXd>
CompactKernel ::Kuf_grad(const ClusterDescriptor &envs,
                         const std::vector<CompactStructure> &strucs,
                         int kernel_index, const Eigen::MatrixXd &Kuf,
                         const Eigen::VectorXd &hyps) {

  int n_sparse = envs.n_clusters;
  int n_hyps = hyps.size();

  // Count labels.
  int n_strucs = strucs.size();
  Eigen::VectorXd label_count = Eigen::VectorXd::Zero(n_strucs + 1);
  int n_labels = 0;
  for (int i = 0; i < n_strucs; i++){
    int current_count = 0;
    if (strucs[i].energy.size() != 0){
        current_count += 1;
    }

    if (strucs[i].forces.size() != 0){
        current_count += strucs[i].forces.size();
    }

    if (strucs[i].stresses.size() != 0){
        current_count += strucs[i].stresses.size();
    }

    label_count(i + 1) = current_count;
    n_labels += current_count;
  }

  // Initialize gradient matrices.
  std::vector<Eigen::MatrixXd> Kuf_grad;
  for (int i = 0; i < n_hyps + 1; i++) {
    Kuf_grad.push_back(Eigen::MatrixXd::Zero(n_sparse, n_labels));
  }

#pragma omp parallel for
  for (int i = 0; i < strucs.size(); i++) {
    std::vector<Eigen::MatrixXd> envs_struc =
        envs_struc_grad(envs, strucs[i].descriptors[kernel_index], hyps);
    int n_atoms = strucs[i].noa;

    for (int j = 0; j < n_hyps + 1; j++) {
      int current_count = 0;

      if (strucs[i].energy.size() != 0) {
        Kuf_grad[j].block(0, label_count(i), n_sparse, 1) =
            envs_struc[j].block(0, 0, n_sparse, 1);
        current_count += 1;
      }

      if (strucs[i].forces.size() != 0) {
        Kuf_grad[j].block(0, label_count(i) + current_count, n_sparse,
                          n_atoms * 3) =
            envs_struc[j].block(0, 1, n_sparse, n_atoms * 3);
        current_count += n_atoms * 3;
      }

      if (strucs[i].stresses.size() != 0) {
        Kuf_grad[j].block(0, label_count(i) + current_count, n_sparse, 6) =
            envs_struc[j].block(0, 1 + n_atoms * 3, n_sparse, 6);
      }
    }
  }

  return Kuf_grad;
}
