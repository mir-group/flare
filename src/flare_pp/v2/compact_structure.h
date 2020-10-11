#ifndef COMPACT_STRUCTURE_H
#define COMPACT_STRUCTURE_H

#include "compact_descriptor.h"
#include "structure.h"
#include <vector>

class CompactStructure : public Structure {
public:
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, structure_indices,
      neighbor_species;
  Eigen::MatrixXd relative_positions;
  double cutoff;
  int sweep, n_neighbors;

  // TODO: Allow for multiple descriptors.
  CompactDescriptor *descriptor;
  DescriptorValues description;

  CompactStructure();

  CompactStructure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
                   const Eigen::MatrixXd &positions, double cutoff,
                   CompactDescriptor *descriptor);

  void compute_neighbors();
  void compute_descriptors();
};

#endif
