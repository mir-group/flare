#include "two_body.h"
#include "cutoffs.h"
#include <iostream>

TwoBody ::TwoBody() {}

TwoBody ::TwoBody(double cutoff, int n_species, const std::string &cutoff_name,
                  const std::vector<double> &cutoff_hyps) {

  this->cutoff = cutoff;
  this->n_species = n_species;

  this->cutoff_name = cutoff_name;
  this->cutoff_hyps = cutoff_hyps;

  set_cutoff(cutoff_name, this->cutoff_function);
}

DescriptorValues TwoBody ::compute_struc(Structure &structure) {

  // Initialize descriptor values.
  DescriptorValues desc = DescriptorValues();

  desc.n_descriptors = 1;
  desc.n_types = n_species * (n_species + 1) / 2;
  desc.n_atoms = structure.noa;
  desc.volume = structure.volume;

  // Count types.
  Eigen::VectorXi type_count = Eigen::VectorXi::Zero(desc.n_types);
  for (int i = 0; i < desc.n_atoms; i++) {
    int i_species = structure.species[i];
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      int neigh_index = rel_index + j;
      int j_species = structure.neighbor_species(neigh_index);
      int struc_index = structure.structure_indices(neigh_index);
      // Avoid counting the same pair twice.
      if ((j_species > i_species) ||
          ((j_species == i_species) && (struc_index >= i))) {
        int species_diff = j_species - i_species;
        int current_type =
            desc.n_types -
            (n_species - i_species) * (n_species - i_species + 1) / 2 +
            species_diff;
        double r = structure.relative_positions(neigh_index, 0);
        // Check that atom is within descriptor cutoff.
        if (r <= cutoff) {
          // Update type count.
          type_count(current_type)++;
        }
      }
    }
  }

  // Initialize arrays.
  desc.cumulative_type_count.push_back(0);
  for (int s = 0; s < desc.n_types; s++) {
    int n_s = type_count(s);

    // Record species and neighbor count.
    // (For the 2-body descriptor, there is 1 neighbor.)
    desc.n_clusters_by_type.push_back(n_s);
    desc.cumulative_type_count.push_back(desc.cumulative_type_count[s] + n_s);
    desc.n_clusters += n_s;
    desc.n_neighbors_by_type.push_back(n_s);

    desc.descriptors.push_back(Eigen::MatrixXd::Zero(n_s, 1));
    desc.descriptor_force_dervs.push_back(Eigen::MatrixXd::Zero(n_s * 3, 1));
    desc.neighbor_coordinates.push_back(Eigen::MatrixXd::Zero(n_s, 3));

    desc.descriptor_norms.push_back(Eigen::VectorXd::Zero(n_s));
    desc.descriptor_force_dots.push_back(Eigen::VectorXd::Zero(n_s * 3));
    desc.cutoff_values.push_back(Eigen::VectorXd::Zero(n_s));
    desc.cutoff_dervs.push_back(Eigen::VectorXd::Zero(n_s * 3));

    desc.neighbor_counts.push_back(Eigen::VectorXi::Zero(n_s));
    desc.cumulative_neighbor_counts.push_back(Eigen::VectorXi::Zero(n_s));
    desc.atom_indices.push_back(Eigen::VectorXi::Zero(n_s));
    desc.neighbor_indices.push_back(Eigen::VectorXi::Zero(n_s));
  }

  // Store descriptors.
  Eigen::VectorXi type_counter = Eigen::VectorXi::Zero(desc.n_types);
  std::vector<double> cutoff_values(2, 0);
  for (int i = 0; i < desc.n_atoms; i++) {
    int i_species = structure.species[i];
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      int neigh_index = rel_index + j;
      int j_species = structure.neighbor_species(neigh_index);
      int struc_index = structure.structure_indices(neigh_index);
      // Avoid counting the same pair twice.
      if ((j_species > i_species) ||
          ((j_species == i_species) && (struc_index >= i))) {
        int species_diff = j_species - i_species;
        int current_type =
            desc.n_types -
            (n_species - i_species) * (n_species - i_species + 1) / 2 +
            species_diff;
        double r = structure.relative_positions(neigh_index, 0);
        // Check that atom is within descriptor cutoff.
        if (r <= cutoff) {
          int count = type_counter(current_type);
          desc.descriptors[current_type](count, 0) = r;

          // Compute cutoff values.
          cutoff_function(cutoff_values, r, cutoff, cutoff_hyps);
          desc.cutoff_values[current_type](count) = cutoff_values[0];

          for (int k = 0; k < 3; k++) {
            double neighbor_coordinate =
                structure.relative_positions(neigh_index, k + 1);
            desc.descriptor_force_dervs[current_type](count * 3 + k, 0) =
                neighbor_coordinate / r;
            desc.neighbor_coordinates[current_type](count, k) =
                neighbor_coordinate;
            desc.cutoff_dervs[current_type](count * 3 + k) =
                cutoff_values[1] * neighbor_coordinate / r;
            desc.descriptor_force_dots[current_type](count * 3 + k) =
                desc.descriptor_force_dervs[current_type](count * 3 + k, 0) *
                desc.descriptors[current_type](count);
          }

          desc.descriptor_norms[current_type](count) = r;
          desc.neighbor_counts[current_type](count) = 1;
          desc.cumulative_neighbor_counts[current_type](count) = count;
          desc.atom_indices[current_type](count) = i;
          desc.neighbor_indices[current_type](count) = struc_index;
          type_counter(current_type)++;
        }
      }
    }
  }

  return desc;
}

// TODO: Implement.
nlohmann::json TwoBody ::return_json(){
  nlohmann::json j;
  return j;
}
