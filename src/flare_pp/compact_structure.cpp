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
  this->max_cutoff = max_cut;
  this->descriptor_calculators = descriptor_calculators;
}

void CompactStructure ::compute_neighbor_lists() {
    // Count the neighbors of each atom.

    // Compute neighbor lists and relative positions.
}
