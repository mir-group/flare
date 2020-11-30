#ifndef GP_H
#define GP_H

#include "descriptor.h"
#include "kernel.h"
#include "structure.h"
#include <Eigen/Dense>
#include <vector>

class GP {
public:
  Eigen::VectorXd hyperparameters;

  // Kernel attributes.
  std::vector<Kernel *> kernels;
  std::vector<Eigen::MatrixXd> Kff_kernels;
  Eigen::MatrixXd Kff;
  int n_kernels = 0;

  // Solution attributes.
  Eigen::MatrixXd Kff_inverse;
  Eigen::VectorXd alpha;

  // Training points.
  std::vector<Structure> training_structures;

  // Label attributes.
  Eigen::VectorXd noise_vector, y, label_count;
  int n_energy_labels = 0, n_force_labels = 0, n_stress_labels = 0,
    n_labels = 0, n_strucs = 0;
  double energy_noise, force_noise, stress_noise;

  // Likelihood attributes.
  double log_marginal_likelihood, data_fit, complexity_penalty, trace_term,
      constant_term;
  Eigen::VectorXd likelihood_gradient;

  GP();
  GP(std::vector<Kernel *> kernels, double energy_noise, double force_noise,
     double stress_noise);

  void add_training_structure(const Structure &structure);
  void update_matrices();
  void predict(Structure &structure);
  void compute_likelihood();
  double compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters);
  void set_hyperparameters(Eigen::VectorXd hyps);
};

#endif
