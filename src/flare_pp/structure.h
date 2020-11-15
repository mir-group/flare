#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "compact_descriptor.h"
#include <vector>

class Structure {
public:
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, structure_indices,
      neighbor_species;
  Eigen::MatrixXd cell, cell_transpose, cell_transpose_inverse, cell_dot,
      cell_dot_inverse, positions, wrapped_positions, relative_positions;
  double cutoff, single_sweep_cutoff, volume;
  int sweep, n_neighbors;
  std::vector<int> species;
  int noa;

  std::vector<CompactDescriptor *> descriptor_calculators;
  std::vector<DescriptorValues> descriptors;

  // Make structure labels empty by default.
  Eigen::VectorXd energy, forces, stresses, mean_efs, variance_efs;
  std::vector<Eigen::VectorXd> mean_contributions;

  Structure();

  Structure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
            const Eigen::MatrixXd &positions);

  Structure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
            const Eigen::MatrixXd &positions, double cutoff,
            std::vector<CompactDescriptor *> descriptor_calculators);

  Eigen::MatrixXd wrap_positions();
  double get_single_sweep_cutoff();
  void compute_neighbors();
  void compute_descriptors();
};

#endif
