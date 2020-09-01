#include "compact_structure.h"

CompactStructure ::CompactStructure() {}

CompactStructure ::CompactStructure(
    const Eigen::MatrixXd &cell, const std::vector<int> &species,
    const Eigen::MatrixXd &positions, std::vector<double> cutoffs,
    std::vector<DescriptorCalculator *> descriptor_calculators)
    : Structure(cell, species, positions) {

  this->cutoffs = cutoffs;
  double max_cut = 0;
  for (int i = 0; i < cutoffs.size(); i++){
      if (cutoffs[i] > max_cut) max_cut = cutoffs[i];
  }

  max_cutoff = max_cut;
  this->descriptor_calculators = descriptor_calculators;
  sweep = ceil(max_cutoff / single_sweep_cutoff);

  for (int i = 0; i < cutoffs.size(); i++){
      neighbor_count.push_back(Eigen::VectorXd::Zero(noa));
  }
}

void CompactStructure ::compute_neighbor_lists() {
    // Count the neighbors of each atom and compute the relative positions
    // of all candidate neighbors.
    Eigen::MatrixXd all_positions =
        Eigen::MatrixXd::Zero(noa * noa * sweep, 4);

    // Compute neighbor lists and relative positions.
}
