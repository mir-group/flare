#ifndef COMPACT_STRUCTURE_H
#define COMPACT_STRUCTURE_H

#include "descriptor.h"
#include "structure.h"

class CompactStructure : public Structure {
public:
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, structure_indices,
    neighbor_species;
  Eigen::MatrixXd relative_positions;

  // Store descriptors and gradients by species.
  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
    descriptor_stress_dervs;
  std::vector<Eigen::VectorXi> neighbor_counts, cumulative_neighbor_counts,
    atom_indices, neighbor_indices;
  DescriptorCalculator * descriptor_calculator;
  double cutoff;
  int sweep, n_neighbors;

  CompactStructure();

  CompactStructure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
                   const Eigen::MatrixXd &positions,
                   double cutoff,
                   DescriptorCalculator * descriptor_calculator);

  void compute_neighbors();
  void compute_descriptors();
};

#endif
