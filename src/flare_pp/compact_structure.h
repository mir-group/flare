#ifndef COMPACT_STRUCTURE_H
#define COMPACT_STRUCTURE_H

#include "descriptor.h"
#include "structure.h"
#include <vector>

class CompactStructures {
public:
  CompactStructures();

  // Descriptor matrices. Stacked to accelerate kernel calculations.
  int n_descriptors, n_species;
  int n_strucs = 0;
  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
      descriptor_stress_dervs;

  // Atoms in each structure.
  std::vector<int> n_atoms, c_atoms;

  // Number of atoms and neighbors in each training structure by species.
  // 1st index: structure; 2nd index: species
  // Used to index into descriptor matrices.
  std::vector<std::vector<int>> atom_counts, cumulative_atom_counts,
      neighbor_counts, cumulative_neighbor_counts;

  // 1st index: structure; 2nd index: species; 3rd index: atom
  std::vector<std::vector<Eigen::VectorXd>> descriptor_norms, force_dots,
      stress_dots;
  std::vector<std::vector<Eigen::VectorXi>> local_neighbor_counts,
      cumulative_local_neighbor_counts, atom_indices, neighbor_indices;

  void add_structure(const CompactStructure &structure);
};

class CompactStructure : public Structure {
public:
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, structure_indices,
      neighbor_species;
  std::vector<int> n_atoms_by_species, n_neighbors_by_species;
  Eigen::MatrixXd relative_positions;

  Eigen::MatrixXd descriptor_vals;

  // Store descriptors and gradients by species.
  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
      descriptor_stress_dervs;
  std::vector<Eigen::VectorXd> descriptor_norms, descriptor_force_dots,
      descriptor_stress_dots;
  std::vector<Eigen::VectorXi> neighbor_counts, cumulative_neighbor_counts,
      atom_indices, neighbor_indices;
  DescriptorCalculator *descriptor_calculator;
  double cutoff;
  int sweep, n_neighbors;

  CompactStructure();

  CompactStructure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
                   const Eigen::MatrixXd &positions, double cutoff,
                   DescriptorCalculator *descriptor_calculator);

  void compute_neighbors();
  void compute_descriptors();
};

#endif
