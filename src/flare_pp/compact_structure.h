#ifndef COMPACT_STRUCTURE_H
#define COMPACT_STRUCTURE_H

#include <Eigen/Dense>
#include <vector>
#include "structure.h"
#include "descriptor.h"

class CompactStructure : public Structure {
public:
  std::vector<Eigen::VectorXd> neighbor_count;
  std::vector<Eigen::MatrixXd> neighbor_lists, relative_positions,
    descriptors, descriptor_force_dervs, descriptor_stress_dervs;
  std::vector<DescriptorCalculator *> descriptor_calculators;

  CompactStructure();

  CompactStructure(const Eigen::MatrixXd &cell,
                   const std::vector<int> &species,
                   const Eigen::MatrixXd &positions,
                   std::vector<double> cutoffs,
                   std::vector<DescriptorCalculator *> descriptor_calculators);

  void compute_neighbor_lists();
  void compute_descriptors();
};

#endif
