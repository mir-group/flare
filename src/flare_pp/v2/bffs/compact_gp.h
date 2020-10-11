#ifndef COMPACT_GP_H
#define COMPACT_GP_H

#include "compact_structures.h"
#include "kernels.h"
#include <Eigen/Dense>
#include <vector>

class CompactGP {

  Eigen::MatrixXd Kuu, Kuf_energy, Kuf_force, Kuf_stress;

  std::vector<Eigen::MatrixXd> sparse_descriptors;
  CompactStructures training_structures;

  Eigen::VectorXd noise_vector, y, energy_labels, force_labels, stress_labels;

  Eigen::MatrixXd Sigma, Kuu_inverse, Kuf;

  CompactGP();
  CompactGP(Kernel *kernel, double sigma_e, double sigma_f, double sigma_s);

  void add_sparse_environments(const CompactStructure &structure,
                               std::vector<int> environments);
  void add_training_structure(const CompactStructure &structure);
};

#endif