#ifndef COMPACT_STRUCTURE_H
#define COMPACT_STRUCTURE_H

#include "descriptor.h"
#include "structure.h"
#include <Eigen/Dense>
#include <vector>

class CompactStructure : public Structure {
public:
  std::vector<Eigen::VectorXd> neighbor_count, structure_indices,
    position_indices;
  Eigen::MatrixXd relative_positions;
  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
      descriptor_stress_dervs;
  std::vector<DescriptorCalculator *> descriptor_calculators;
  std::vector<double> cutoffs;
  double max_cutoff;
  int sweep;

  CompactStructure();

  CompactStructure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
                   const Eigen::MatrixXd &positions,
                   std::vector<double> cutoffs,
                   std::vector<DescriptorCalculator *> descriptor_calculators);

  void compute_neighbor_lists();
  void compute_descriptors();
};

#endif
