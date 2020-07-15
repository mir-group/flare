#include "sparse_gp_dtc.h"

SparseGP_DTC ::SparseGP_DTC() {}

SparseGP_DTC ::SparseGP_DTC(std::vector<Kernel *> kernels, double sigma_e,
                            double sigma_f, double sigma_s)
    : SparseGP(kernels, sigma_e, sigma_f, sigma_s) {}

void SparseGP_DTC ::update_matrices(){
  // Combine Kuf_struc and Kuf_env.
  int n_sparse = Kuf_struc.rows();
  int n_struc_labels = Kuf_struc.cols();
  int n_env_labels = Kuf_env.cols();
  Eigen::MatrixXd Kuf =
      Eigen::MatrixXd::Zero(n_sparse, n_struc_labels + n_env_labels);
  Kuf.block(0, 0, n_sparse, n_struc_labels) = Kuf_struc;
  Kuf.block(0, n_struc_labels, n_sparse, n_env_labels) = Kuf_env;

  // Combine noise_struc and noise_env.
  Eigen::VectorXd noise = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  noise.segment(0, n_struc_labels) = noise_struc;
  noise.segment(n_struc_labels, n_env_labels) = noise_env;

  // Combine training labels.
  Eigen::VectorXd y = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  y.segment(0, n_struc_labels) = y_struc;
  y.segment(n_struc_labels, n_env_labels) = y_env;

  // Calculate Sigma.
  Eigen::MatrixXd sigma_inv =
      Kuu + Kuf * noise.asDiagonal() * Kuf.transpose() +
      Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());

  Sigma = sigma_inv.inverse();

  // Calculate Kuu inverse.
  Kuu_inverse = Kuu.inverse();

  // Calculate alpha.
  alpha = Sigma * Kuf * noise.asDiagonal() * y;
}

void SparseGP_DTC ::predict_DTC(StructureDescriptor test_structure,
    Eigen::VectorXd & mean_vector, Eigen::VectorXd & variance_vector,
    std::vector<Eigen::VectorXd> & mean_contributions){

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;
  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();

  // Store kernel matrices for each kernel.
  std::vector<Eigen::MatrixXd> kern_mats;
  for (int i = 0; i < n_kernels; i ++){
      kern_mats.push_back(Eigen::MatrixXd::Zero(n_out, n_sparse));
  }
  Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(n_out, n_sparse);

// Compute the kernel between the test structure and each sparse
// environment, parallelizing over environments.
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_sparse; i++) {
    for (int j = 0; j < n_kernels; j++) {
      kern_mats[j].col(i) +=
          kernels[j]->env_struc(sparse_environments[i], test_structure);
    }
  }

  // Sum the kernels.
  for (int i = 0; i < n_kernels; i ++){
      kern_mat += kern_mats[i];
  }

  // Compute mean contributions and total mean prediction.
  mean_vector = kern_mat * alpha;

  // Compute variances.
  Eigen::VectorXd Q_self, K_self = Eigen::VectorXd::Zero(n_out);

  // Note: Calculation of the self-kernel can be parallelized.
  for (int i = 0; i < n_kernels; i ++){
      K_self += kernels[i]->self_kernel_struc(test_structure);
  }

  // TODO next: Calculate Q_self. Note that only the trace is needed of the
  // full matrix product is needed.
}
