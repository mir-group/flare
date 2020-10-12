#ifndef COMPACT_GP_H
#define COMPACT_GP_H

#include "compact_structure.h"
#include "compact_descriptor.h"
#include "kernels.h"
#include <Eigen/Dense>
#include <vector>

class CompactGP {
  Eigen::VectorXd hyperparameters;

  // Kernel attributes.
  // TODO: Make a separate class for this. May need to store more information
  // to facilitate hyperparameter optimization.
  std::vector<Eigen::MatrixXd> Kuu_kernels, Kuf_energy, Kuf_force, Kuf_stress;
  Eigen::MatrixXd Kuu, Kuf;

  // Solution attributes.
  Eigen::MatrixXd Sigma, Kuu_inverse;
  Eigen::VectorXd alpha;

  // Training and sparse points.
  std::vector<ClusterDescriptor> sparse_descriptors;
  std::vector<CompactStructure> training_structures;

  // Label attributes.
  Eigen::VectorXd noise_vector, y, energy_labels, force_labels, stress_labels;
  int n_energy_labels = 0, n_force_labels = 0, n_stress_labels = 0, n_labels;

  // Likelihood attributes.
  double log_marginal_likelihood, data_fit, complexity_penalty, trace_term,
      constant_term;
  Eigen::VectorXd likelihood_gradient;

  CompactGP();
  CompactGP(std::vector<Kernel *> kernels, double energy_noise,
            double force_noise, double stress_noise);

  void add_sparse_environments(const CompactStructure &structure);
  void add_training_structure(const CompactStructure &structure);

  void update_matrices_QR();
  void predict_on_structure(CompactStructure &structure);

  void compute_likelihood();
  double compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters);
  void set_hyperparameters(Eigen::VectorXd hyperparameters);

};

#endif
