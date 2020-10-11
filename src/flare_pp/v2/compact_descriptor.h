#ifndef COMPACT_DESCRIPTOR_H
#define COMPACT_DESCRIPTOR_H

#include <Eigen/Dense>
#include <vector>

class LocalEnvironment;
class CompactStructure;

// Descriptor calculator.
class CompactDescriptor {
public:
  // Descriptor attributes.
  Eigen::VectorXi species_indices;
  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
    neighbor_coordinates;
  std::vector<Eigen::VectorXd> descriptor_norms, descriptor_force_dots;
  std::vector<Eigen::VectorXi> neighbor_counts, cumulative_neighbor_counts,
      atom_indices, neighbor_indices;
  std::vector<int> n_atoms_by_species, n_neighbors_by_species;

  CompactDescriptor();

  virtual void compute_struc(CompactStructure &structure) = 0;
};


#endif
