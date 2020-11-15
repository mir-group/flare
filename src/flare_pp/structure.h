#ifndef STRUCTURE_H
#define STRUCTURE_H

#include <Eigen/Dense>
#include <vector>

// Structure class.
class Structure {
public:
  Eigen::MatrixXd cell, cell_transpose, cell_transpose_inverse, cell_dot,
      cell_dot_inverse, positions, wrapped_positions;
  std::vector<int> species;
  double single_sweep_cutoff, volume;
  int noa;

  Structure();

  Structure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
            const Eigen::MatrixXd &positions);

  Eigen::MatrixXd wrap_positions();

  double get_single_sweep_cutoff();
};

#endif
