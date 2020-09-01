#include "compact_structure.h"
#include "omp.h"
#include <iostream>

CompactStructure ::CompactStructure() {}

CompactStructure ::CompactStructure(
    const Eigen::MatrixXd &cell, const std::vector<int> &species,
    const Eigen::MatrixXd &positions, std::vector<double> cutoffs,
    std::vector<DescriptorCalculator *> descriptor_calculators)
    : Structure(cell, species, positions) {

  this->cutoffs = cutoffs;
  double max_cut = 0;
  for (int i = 0; i < cutoffs.size(); i++) {
    if (cutoffs[i] > max_cut)
      max_cut = cutoffs[i];
  }

  max_cutoff = max_cut;
  this->descriptor_calculators = descriptor_calculators;
  sweep = ceil(max_cutoff / single_sweep_cutoff);

  // Initialize neighbor count.
  for (int i = 0; i < cutoffs.size(); i++) {
    neighbor_count.push_back(Eigen::VectorXi::Zero(noa));
  }

  compute_neighbors();
}

void CompactStructure ::compute_neighbors() {
  // Count the neighbors of each atom and compute the relative positions
  // of all candidate neighbors.
  int sweep_unit = 2 * sweep + 1;
  int sweep_no = sweep_unit * sweep_unit * sweep_unit;
  Eigen::MatrixXd all_positions =
      Eigen::MatrixXd::Zero(noa * noa * sweep_no, 4);
  Eigen::VectorXi max_neighbor_count = Eigen::VectorXi::Zero(noa);

// Compute neighbor lists and relative positions.
#pragma omp parallel for
  for (int i = 0; i < noa; i++) {
    Eigen::MatrixXd pos_atom = wrapped_positions.row(i);
    int i_index = i * noa * sweep_no;
    int counter = 0;
    for (int j = 0; j < noa; j++) {
      Eigen::MatrixXd diff_curr = wrapped_positions.row(j) - pos_atom;
      for (int s1 = -sweep; s1 < sweep + 1; s1++) {
        for (int s2 = -sweep; s2 < sweep + 1; s2++) {
          for (int s3 = -sweep; s3 < sweep + 1; s3++) {
            Eigen::MatrixXd im = diff_curr + s1 * cell.row(0) +
                                 s2 * cell.row(1) + s3 * cell.row(2);
            double dist = sqrt(im(0) * im(0) + im(1) * im(1) + im(2) * im(2));

            // Store coordinates and distance.
            if ((dist < max_cutoff) && (dist != 0)) {
              max_neighbor_count(i)++;
              int row_index = i_index + counter;
              all_positions(row_index, 0) = im(0);
              all_positions(row_index, 1) = im(1);
              all_positions(row_index, 2) = im(2);
              all_positions(row_index, 3) = dist;
              counter++;
            }
          }
        }
      }
    }
  }
}
