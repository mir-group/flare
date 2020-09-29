#include "compact_structure.h"
#include "omp.h"
#include <iostream>

CompactStructure ::CompactStructure() {}

CompactStructure ::CompactStructure(
    const Eigen::MatrixXd &cell, const std::vector<int> &species,
    const Eigen::MatrixXd &positions, double cutoff,
    DescriptorCalculator * descriptor_calculator)
    : Structure(cell, species, positions) {

  this->cutoff = cutoff;
  this->descriptor_calculator = descriptor_calculator;
  sweep = ceil(cutoff / single_sweep_cutoff);

  // Initialize neighbor count.
  neighbor_count = Eigen::VectorXi::Zero(noa);
  cumulative_neighbor_count = Eigen::VectorXi::Zero(noa + 1);

  compute_neighbors();
  this->descriptor_calculator->compute_struc(*this);
}

void CompactStructure ::compute_neighbors() {
  // Count the neighbors of each atom and compute the relative positions
  // of all candidate neighbors.
  int sweep_unit = 2 * sweep + 1;
  int sweep_no = sweep_unit * sweep_unit * sweep_unit;
  Eigen::MatrixXd all_positions =
      Eigen::MatrixXd::Zero(noa * noa * sweep_no, 4);
  Eigen::VectorXi all_indices = Eigen::VectorXi::Zero(noa * noa * sweep_no);

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
            if ((dist < cutoff) && (dist != 0)) {
              neighbor_count(i)++;
              int row_index = i_index + counter;
              all_positions(row_index, 0) = dist;
              all_positions(row_index, 1) = im(0);
              all_positions(row_index, 2) = im(1);
              all_positions(row_index, 3) = im(2);
              all_indices(row_index) = j;
              counter++;
            }
          }
        }
      }
    }
  }

  // Store cumulative neighbor counts.
  for (int i = 1; i < noa + 1; i++) {
    cumulative_neighbor_count(i) +=
        cumulative_neighbor_count(i - 1) + neighbor_count(i - 1);
  }

  // Store relative positions.
  n_neighbors = cumulative_neighbor_count(noa);
  relative_positions = Eigen::MatrixXd::Zero(n_neighbors, 4);
  structure_indices = Eigen::VectorXi::Zero(n_neighbors);
  neighbor_species = Eigen::VectorXi::Zero(n_neighbors);
#pragma omp parallel for
  for (int i = 0; i < noa; i++) {
    int n_neighbors = neighbor_count(i);
    int rel_index = cumulative_neighbor_count(i);
    int all_index = i * noa * sweep_no;
    for (int j = 0; j < n_neighbors; j++) {
      int current_index = all_indices(all_index + j);
      structure_indices(rel_index + j) = current_index;
      neighbor_species(rel_index + j) = species[current_index];
      for (int k = 0; k < 4; k++) {
        relative_positions(rel_index + j, k) = all_positions(all_index + j, k);
      }
    }
  }
}
