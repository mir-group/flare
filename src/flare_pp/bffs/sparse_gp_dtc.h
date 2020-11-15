#ifndef SPARSE_GP_DTC_H
#define SPARSE_GP_DTC_H

#include "compact_descriptor.h"
#include "kernel.h"
#include "structure.h"
#include <Eigen/Dense>
#include <vector>

class SparseGP_DTC {
public:
  Eigen::VectorXd hyperparameters;

  // Kernel attributes.
  std::vector<CompactKernel *> kernels;
  std::vector<Eigen::MatrixXd> Kuu_kernels, Kuf_kernels;
  Eigen::MatrixXd Kuu, Kuf;
  double Kuu_jitter;

  // Solution attributes.
  Eigen::MatrixXd Sigma, Kuu_inverse;
  Eigen::VectorXd alpha;

  // Training and sparse points.
  std::vector<ClusterDescriptor> sparse_descriptors;
  std::vector<Structure> training_structures;

  // Label attributes.
  Eigen::VectorXd noise_vector, y, label_count;
  int n_energy_labels = 0, n_force_labels = 0, n_stress_labels = 0,
      n_sparse = 0, n_labels = 0;
  double energy_noise, force_noise, stress_noise;

  // Likelihood attributes.
  double log_marginal_likelihood, data_fit, complexity_penalty, trace_term,
      constant_term;
  Eigen::VectorXd likelihood_gradient;

  // Constructors.
  SparseGP_DTC();
  SparseGP_DTC(std::vector<CompactKernel *> kernels, double energy_noise,
               double force_noise, double stress_noise);

  // TODO: Add sparse environments above an energy uncertainty threshold.
  void add_sparse_environments(const Structure &structure);
  void add_training_structure(const Structure &structure);
  void update_Kuu();
  void update_Kuf();

  void update_matrices_QR();
  void predict_on_structure(Structure &structure);

  void compute_likelihood();

  double compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters);
  void set_hyperparameters(Eigen::VectorXd hyps);
};

#endif
