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

  std::vector<CompactDescriptor *> descriptor_calculators;
  std::vector<DescriptorValues> descriptors;

  // Make structure labels empty by default.
  Eigen::VectorXd energy, forces, stresses, mean_efs, variance_efs;
  std::vector<Eigen::VectorXd> mean_contributions;

  CompactStructure();

  CompactStructure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
                   const Eigen::MatrixXd &positions, double cutoff,
                   std::vector<CompactDescriptor *> descriptor_calculators);

  void compute_neighbors();
  void compute_descriptors();
};

#endif
