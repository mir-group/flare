#ifndef COMPACT_STRUCTURE_H
#define COMPACT_STRUCTURE_H

#include "descriptor.h"
#include "structure.h"

class CompactStructures{
  public:
      CompactStructures();

      // Initialize with a list of structures.
      CompactStructures(std::vector<CompactStructure> structures);

      std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
        descriptor_stress_dervs;

      // 1st index: structure; 2nd index: species; 3rd index: atom
      std::vector<std::vector<Eigen::VectorXd>> descriptor_norms, force_dots,
        stress_dots;
      std::vector<Eigen::VectorXi> local_neighbor_counts,
        cumulative_local_neighbor_counts, atom_indices, neighbor_indices;

      // Number of atoms and neighbors in each training structure by species.
      // 1st index: structure; 2nd index: species
      // Used to index into descriptor matrices.
      Eigen::ArrayXXi atom_counts, cumulative_atom_counts, neighbor_counts,
        cumulative_neighbor_counts;

      void add_structure(const CompactStructure &structure);
};

class CompactStructure : public Structure {
public:
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, structure_indices,
    neighbor_species;
  Eigen::MatrixXd relative_positions;

  Eigen::MatrixXd descriptor_vals;

  // Store descriptors and gradients by species.
  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
    descriptor_stress_dervs;
  std::vector<Eigen::VectorXd> descriptor_norms, descriptor_force_dots,
    descriptor_stress_dots;
  std::vector<Eigen::VectorXi> neighbor_counts, cumulative_neighbor_counts,
    atom_indices, neighbor_indices;
  DescriptorCalculator * descriptor_calculator;
  double cutoff;
  int sweep, n_neighbors;

  CompactStructure();

  CompactStructure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
                   const Eigen::MatrixXd &positions, double cutoff,
                   DescriptorCalculator * descriptor_calculator);

  void compute_neighbors();
  void compute_descriptors();
};

#endif
