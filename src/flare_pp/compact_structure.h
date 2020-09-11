#ifndef COMPACT_STRUCTURE_H
#define COMPACT_STRUCTURE_H

#include "descriptor.h"
#include "structure.h"
#include <Eigen/Dense>
#include <vector>

class CompactStructure : public Structure {
public:
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, structure_indices;
  Eigen::MatrixXd relative_positions;
  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
      descriptor_stress_dervs;
  std::vector<DescriptorCalculator *> descriptor_calculators;
  double cutoff;
  int sweep;

  CompactStructure();

  CompactStructure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
                   const Eigen::MatrixXd &positions,
                   double cutoff,
                   std::vector<DescriptorCalculator *> descriptor_calculators);

  void compute_neighbors();
  void compute_descriptors();
};

#endif
