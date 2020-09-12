#ifndef SPARSE_GP_DTC_H
#define SPARSE_GP_DTC_H

#include "sparse_gp.h"
#define EIGEN_USE_LAPACKE
#include <Eigen/Dense>

class SparseGP_DTC : public SparseGP {
public:
  // TODO: Modify "add" methods to keep track of each kernel contribution.
  std::vector<Eigen::MatrixXd> Kuf_env_kernels, Kuu_kernels, Kuf_struc_energy,
      Kuf_struc_force, Kuf_struc_stress;
  Eigen::VectorXd noise_vector, y, energy_labels, force_labels, stress_labels;
  Eigen::MatrixXd Sigma, Kuu_inverse, Kuf;

  int n_energy_labels = 0, n_force_labels = 0, n_stress_labels = 0, n_labels;
  int is_positive = 0;

  // Likelihood attributes.
  double log_marginal_likelihood, data_fit, complexity_penalty, trace_term,
      constant_term;
  Eigen::VectorXd likelihood_gradient;

  SparseGP_DTC();
  SparseGP_DTC(std::vector<Kernel *> kernels, double sigma_e, double sigma_f,
               double sigma_s);

  // Methods for augmenting the training set.
  void add_sparse_environments(const std::vector<LocalEnvironment> &envs);
  void add_training_structure(const StructureDescriptor &training_structure);

  // Not implemented.
  void add_sparse_environment(const LocalEnvironment &env);
  void add_training_environment(const LocalEnvironment &training_environment);
  void add_training_environments(const std::vector<LocalEnvironment> &envs);

  // Update matrices and vectors needed for mean and variance prediction:
  // Sigma, Kuu_inverse, and alpha.
  void update_matrices();

  // Update matrices with QR decomposition. Expected to be more stable than
  // explicit inversion.
  void update_matrices_QR();

  // Compute the DTC mean and variance.
  void predict_on_structure(StructureDescriptor &test_structure);

  // Calculate the log marginal likelihood of the current hyperparameters.
  // TODO: Change compute likelihood to set likelihood, and set both the
  // likelihood and its gradient.
  void compute_likelihood();
  double compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters);

  // Change the model hyperparameters and covariance matrices.
  void set_hyperparameters(Eigen::VectorXd hyperparameters);
};

#endif
