#ifndef COMPACT_GP_H
#define COMPACT_GP_H

#include <Eigen/Dense>
#include <vector>
#include "compact_structure.h"
#include "kernels.h"

class CompactGP {

  Eigen::MatrixXd Kuu, Kuf_energy, Kuf_force, Kuf_stress;

  std::vector<Eigen::MatrixXd> sparse_descriptors, structure_descriptors,
   structure_force_dervs, structure_stress_dervs;

  std::vector<Eigen::VectorXd> descriptor_norms, force_dots, stress_dots;

  std::vector<Eigen::VectorXi> local_neighbor_counts,
    cumulative_local_neighbor_counts, atom_indices, neighbor_indices;

  // Number of atoms and neighbors in each training structure by species.
  // Rows = # training structures, columns = # species
  Eigen::ArrayXXi atom_counts, cumulative_atom_counts, neighbor_counts,
    cumulative_neighbor_counts;

  Eigen::VectorXd noise_vector, y, energy_labels, force_labels, stress_labels;

  Eigen::MatrixXd Sigma, Kuu_inverse, Kuf;

  CompactGP();
  CompactGP(Kernel * kernel, double sigma_e, double sigma_f, double sigma_s);

  void add_sparse_environments(const CompactStructure &structure,
    std::vector<int> environments);
  void add_training_structure(const CompactStructure &structure);

};

#endif