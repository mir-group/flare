#include "structure.h"
#include <fstream> // File operations
#include <iostream>

Structure ::Structure() {}

Structure ::Structure(const Eigen::MatrixXd &cell,
                      const std::vector<int> &species,
                      const Eigen::MatrixXd &positions) {
  // Set cell, species, and positions.
  this->cell = cell;
  this->species = species;
  this->positions = positions;
  single_sweep_cutoff = get_single_sweep_cutoff();
  volume = abs(cell.determinant());
  noa = species.size();

  cell_transpose = cell.transpose();
  cell_transpose_inverse = cell_transpose.inverse();
  cell_dot = cell * cell_transpose;
  cell_dot_inverse = cell_dot.inverse();

  // Store wrapped positions.
  this->wrapped_positions = wrap_positions();
}

Structure ::Structure(const Eigen::MatrixXd &cell,
                      const std::vector<int> &species,
                      const Eigen::MatrixXd &positions, double cutoff,
                      std::vector<Descriptor *> descriptor_calculators)
    : Structure(cell, species, positions) {

  this->cutoff = cutoff;
  this->descriptor_calculators = descriptor_calculators;
  sweep = ceil(cutoff / single_sweep_cutoff);

  // Initialize neighbor count.
  neighbor_count = Eigen::VectorXi::Zero(noa);
  cumulative_neighbor_count = Eigen::VectorXi::Zero(noa + 1);

  compute_neighbors();
  compute_descriptors();
}

void Structure ::compute_descriptors(){
  descriptors.clear();
  for (int i = 0; i < descriptor_calculators.size(); i++){
    descriptors.push_back(descriptor_calculators[i]->compute_struc(*this));
  }
}

void Structure ::compute_neighbors() {
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

Eigen::MatrixXd Structure ::wrap_positions() {
  // Convert Cartesian coordinates to relative coordinates.
  Eigen::MatrixXd relative_positions =
      (positions * this->cell_transpose) * this->cell_dot_inverse;

  // Calculate wrapped relative coordinates by subtracting the floor.
  Eigen::MatrixXd relative_floor = relative_positions.array().floor();
  Eigen::MatrixXd relative_wrapped = relative_positions - relative_floor;

  Eigen::MatrixXd wrapped_positions =
      (relative_wrapped * this->cell_dot) * this->cell_transpose_inverse;

  return wrapped_positions;
}

double Structure ::get_single_sweep_cutoff() {
  Eigen::MatrixXd vec1 = cell.row(0);
  Eigen::MatrixXd vec2 = cell.row(1);
  Eigen::MatrixXd vec3 = cell.row(2);

  double max_candidates[6];

  double a_dot_b = vec1(0) * vec2(0) + vec1(1) * vec2(1) + vec1(2) * vec2(2);
  double a_dot_c = vec1(0) * vec3(0) + vec1(1) * vec3(1) + vec1(2) * vec3(2);
  double b_dot_c = vec2(0) * vec3(0) + vec2(1) * vec3(1) + vec2(2) * vec3(2);

  double a = sqrt(vec1(0) * vec1(0) + vec1(1) * vec1(1) + vec1(2) * vec1(2));
  double b = sqrt(vec2(0) * vec2(0) + vec2(1) * vec2(1) + vec2(2) * vec2(2));
  double c = sqrt(vec3(0) * vec3(0) + vec3(1) * vec3(1) + vec3(2) * vec3(2));

  max_candidates[0] = a * sqrt(1 - pow(a_dot_b / (a * b), 2));
  max_candidates[1] = b * sqrt(1 - pow(a_dot_b / (a * b), 2));
  max_candidates[2] = a * sqrt(1 - pow(a_dot_c / (a * c), 2));
  max_candidates[3] = c * sqrt(1 - pow(a_dot_c / (a * c), 2));
  max_candidates[4] = b * sqrt(1 - pow(b_dot_c / (b * c), 2));
  max_candidates[5] = c * sqrt(1 - pow(b_dot_c / (b * c), 2));

  double single_sweep_cutoff = max_candidates[0];
  for (int i = 0; i < 6; i++) {
    if (max_candidates[i] < single_sweep_cutoff) {
      single_sweep_cutoff = max_candidates[i];
    }
  }

  return single_sweep_cutoff;
}


void Structure ::to_json(std::string file_name, const Structure & struc){
  std::ofstream struc_file(file_name);
  nlohmann::json j = struc;
  struc_file << j;
}

Structure Structure ::from_json(std::string file_name){
  std::ifstream struc_file(file_name);
  nlohmann::json j;
  struc_file >> j;
  return j;
}
