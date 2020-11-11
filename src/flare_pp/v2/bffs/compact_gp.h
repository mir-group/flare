#ifndef COMPACT_GP_H
#define COMPACT_GP_H

#include "compact_structure.h"
#include "compact_descriptor.h"
#include "kernel.h"
#include <Eigen/Dense>
#include <vector>

class CompactGP {
public:
  Eigen::VectorXd hyperparameters;

  // Kernel attributes.
  std::vector<CompactKernel *> kernels;
  std::vector<Eigen::MatrixXd> Kuu_kernels, Kuf_kernels, Kuf_energy,
    Kuf_force, Kuf_stress;
  Eigen::MatrixXd Kuu, Kuf;
  double Kuu_jitter;

  // Solution attributes.
  Eigen::MatrixXd Sigma, Kuu_inverse;
  Eigen::VectorXd alpha;

  // Training and sparse points.
  std::vector<ClusterDescriptor> sparse_descriptors;
  std::vector<CompactStructure> training_structures;

  // Label attributes.
  Eigen::VectorXd noise_vector, y, energy_labels, force_labels, stress_labels;
  int n_energy_labels = 0, n_force_labels = 0, n_stress_labels = 0,
    n_sparse = 0, n_labels = 0;
  double energy_noise, force_noise, stress_noise;

  // Likelihood attributes.
  double log_marginal_likelihood, data_fit, complexity_penalty, trace_term,
      constant_term;
  Eigen::VectorXd likelihood_gradient;

  // Constructors.
  CompactGP();
  CompactGP(std::vector<CompactKernel *> kernels, double energy_noise,
            double force_noise, double stress_noise);

  // TODO: Add sparse environments above an energy uncertainty threshold.
  void add_sparse_environments(const CompactStructure &structure);
  void add_training_structure(const CompactStructure &structure);
  void update_Kuu();
  void update_Kuf();

  void update_matrices_QR();
  void predict_on_structure(CompactStructure &structure);

  void compute_likelihood();
  double compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters);
  void set_hyperparameters(Eigen::VectorXd hyperparameters);

};

#endif
