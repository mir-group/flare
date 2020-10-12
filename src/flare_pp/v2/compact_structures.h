#ifndef COMPACT_STRUCTURES_H
#define COMPACT_STRUCTURES_H

#include "compact_structure.h"
#include "descriptor.h"
#include "structure.h"
#include <vector>

// Note: This class may not be needed; might be roughly as efficient to work
// with individual structures.
class CompactStructures {
public:
  CompactStructures();

  // Descriptor matrices. Stacked to accelerate kernel calculations.
  int n_descriptors, n_species;
  int n_strucs = 0;
  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs;

  // Atoms in each structure.
  std::vector<int> n_atoms, c_atoms;

  // Number of atoms and neighbors in each training structure by species.
  // 1st index: structure; 2nd index: species
  // Used to index into descriptor matrices.
  std::vector<std::vector<int>> atom_counts, cumulative_atom_counts,
      neighbor_counts, cumulative_neighbor_counts;

  // 1st index: structure; 2nd index: species; 3rd index: atom
  std::vector<std::vector<Eigen::VectorXd>> descriptor_norms, force_dots;
  std::vector<std::vector<Eigen::VectorXi>> local_neighbor_counts,
      cumulative_local_neighbor_counts, atom_indices, neighbor_indices;

  void add_structure(const CompactStructure &structure);
};

#endif
