#include "three_body_wide.h"
#include "cutoffs.h"
#include <iostream>

ThreeBodyWide ::ThreeBodyWide() {}

ThreeBodyWide ::ThreeBodyWide(double cutoff, int n_species,
                              const std::string &cutoff_name,
                              const std::vector<double> &cutoff_hyps) {

  this->cutoff = cutoff;
  this->n_species = n_species;

  this->cutoff_name = cutoff_name;
  this->cutoff_hyps = cutoff_hyps;

  set_cutoff(cutoff_name, this->cutoff_function);
}

DescriptorValues ThreeBodyWide ::compute_struc(Structure &structure) {

  // Initialize descriptor values.
  DescriptorValues desc = DescriptorValues();

  desc.n_descriptors = 3;
  desc.n_types = n_species * (n_species + 1) * (n_species + 2) / 6;
  desc.n_atoms = structure.noa;
  desc.volume = structure.volume;
  int n_neighbors = structure.n_neighbors;

  // Count types.
  Eigen::VectorXi type_count = Eigen::VectorXi::Zero(desc.n_types);
  // TODO: Consider parallelizing.
  for (int i = 0; i < desc.n_atoms; i++) {
    int i_species = structure.species[i];
    int t1 = desc.n_types - (n_species - i_species) *
                                (n_species - i_species + 1) *
                                (n_species - i_species + 2) / 6;
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      int neigh_index_1 = rel_index + j;
      int j_species = structure.neighbor_species(neigh_index_1);
      if (j_species < i_species)
        continue;
      int t2 = (n_species - i_species) * (n_species - i_species + 1) / 2 -
               (n_species - j_species) * (n_species - j_species + 1) / 2;
      double r1 = structure.relative_positions(neigh_index_1, 0);
      if (r1 > cutoff)
        continue;
      double x1 = structure.relative_positions(neigh_index_1, 1);
      double y1 = structure.relative_positions(neigh_index_1, 2);
      double z1 = structure.relative_positions(neigh_index_1, 3);
      for (int k = 0; k < i_neighbors; k++) {
        if (j == k)
          continue;
        int neigh_index_2 = rel_index + k;
        int k_species = structure.neighbor_species(neigh_index_2);
        if (k_species < j_species)
          continue;
        int t3 = k_species - j_species;
        double r2 = structure.relative_positions(neigh_index_2, 0);
        if (r2 > cutoff)
          continue;
        int current_type = t1 + t2 + t3;
        type_count(current_type)++;
      }
    }
  }

  // Initialize arrays.
  desc.cumulative_type_count.push_back(0);
  for (int s = 0; s < desc.n_types; s++) {
    int n_s = type_count(s);
    int n_neigh = n_s * 2;
    int n_d = 3;

    // Record species and neighbor count.
    // (For the 3-body descriptor, there are 2 neighbors.)
    desc.n_clusters_by_type.push_back(n_s);
    desc.cumulative_type_count.push_back(desc.cumulative_type_count[s] + n_s);
    desc.n_clusters += n_s;
    desc.n_neighbors_by_type.push_back(n_neigh);

    desc.descriptors.push_back(Eigen::MatrixXd::Zero(n_s, n_d));
    desc.descriptor_force_dervs.push_back(
        Eigen::MatrixXd::Zero(n_neigh * 3, n_d));
    desc.neighbor_coordinates.push_back(Eigen::MatrixXd::Zero(n_neigh, 3));

    desc.descriptor_norms.push_back(Eigen::VectorXd::Zero(n_s));
    desc.descriptor_force_dots.push_back(Eigen::VectorXd::Zero(n_neigh * 3));
    desc.cutoff_values.push_back(Eigen::VectorXd::Zero(n_s));
    desc.cutoff_dervs.push_back(Eigen::VectorXd::Zero(n_neigh * 3));

    desc.neighbor_counts.push_back(Eigen::VectorXi::Zero(n_s));
    desc.cumulative_neighbor_counts.push_back(Eigen::VectorXi::Zero(n_s));
    desc.atom_indices.push_back(Eigen::VectorXi::Zero(n_s));
    desc.neighbor_indices.push_back(Eigen::VectorXi::Zero(n_neigh));
  }

  // Store descriptors.
  Eigen::VectorXi type_counter = Eigen::VectorXi::Zero(desc.n_types);
  std::vector<double> cut1(2, 0), cut2(2, 0), cut3(3, 0);
  for (int i = 0; i < desc.n_atoms; i++) {
    int i_species = structure.species[i];
    int t1 = desc.n_types - (n_species - i_species) *
                                (n_species - i_species + 1) *
                                (n_species - i_species + 2) / 6;
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      int neigh_index_1 = rel_index + j;
      int j_species = structure.neighbor_species(neigh_index_1);
      if (j_species < i_species)
        continue;
      int t2 = (n_species - i_species) * (n_species - i_species + 1) / 2 -
               (n_species - j_species) * (n_species - j_species + 1) / 2;
      int struc_index_1 = structure.structure_indices(neigh_index_1);
      double r1 = structure.relative_positions(neigh_index_1, 0);
      if (r1 > cutoff)
        continue;
      double x1 = structure.relative_positions(neigh_index_1, 1);
      double y1 = structure.relative_positions(neigh_index_1, 2);
      double z1 = structure.relative_positions(neigh_index_1, 3);
      for (int k = 0; k < i_neighbors; k++) {
        if (j == k)
          continue;
        int neigh_index_2 = rel_index + k;
        int k_species = structure.neighbor_species(neigh_index_2);
        if (k_species < j_species)
          continue;
        int t3 = k_species - j_species;
        int struc_index_2 = structure.structure_indices(neigh_index_2);
        double r2 = structure.relative_positions(neigh_index_2, 0);
        if (r2 > cutoff)
          continue;
        double x2 = structure.relative_positions(neigh_index_2, 1);
        double y2 = structure.relative_positions(neigh_index_2, 2);
        double z2 = structure.relative_positions(neigh_index_2, 3);
        double r3 = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
        int current_type = t1 + t2 + t3;

        int count = type_counter(current_type);
        desc.descriptors[current_type](count, 0) = r1;
        desc.descriptors[current_type](count, 1) = r2;
        desc.descriptors[current_type](count, 2) = r3;

        // Compute cutoff values.
        cutoff_function(cut1, r1, cutoff, cutoff_hyps);
        cutoff_function(cut2, r2, cutoff, cutoff_hyps);
        desc.cutoff_values[current_type](count) = cut1[0] * cut2[0];

        for (int k = 0; k < 3; k++) {
          double neigh_coord_1 =
              structure.relative_positions(neigh_index_1, k + 1);
          double neigh_coord_2 =
              structure.relative_positions(neigh_index_2, k + 1);
          double coord_diff = neigh_coord_2 - neigh_coord_1;

          // First neighbor.
          desc.descriptor_force_dervs[current_type](count * 2 * 3 + k, 0) =
              neigh_coord_1 / r1;
          desc.descriptor_force_dervs[current_type](count * 2 * 3 + k, 2) =
              -coord_diff / r3;

          desc.neighbor_coordinates[current_type](count * 2, k) = neigh_coord_1;
          desc.cutoff_dervs[current_type](count * 2 * 3 + k) =
              cut1[1] * cut2[0] * neigh_coord_1 / r1;

          desc.descriptor_force_dots[current_type](count * 2 * 3 + k) =
              desc.descriptor_force_dervs[current_type]
                  .row(count * 2 * 3 + k)
                  .dot(desc.descriptors[current_type].row(count));

          // Second neighbor.
          desc.descriptor_force_dervs[current_type](count * 2 * 3 + 3 + k, 1) =
              neigh_coord_2 / r2;
          desc.descriptor_force_dervs[current_type](count * 2 * 3 + 3 + k, 2) =
              coord_diff / r3;

          desc.descriptor_force_dots[current_type](count * 2 * 3 + 3 + k) =
              desc.descriptor_force_dervs[current_type]
                  .row(count * 2 * 3 + 3 + k)
                  .dot(desc.descriptors[current_type].row(count));

          desc.neighbor_coordinates[current_type](count * 2 + 1, k) =
              neigh_coord_2;
          desc.cutoff_dervs[current_type](count * 2 * 3 + 3 + k) =
              cut1[0] * cut2[1] * neigh_coord_2 / r2;
        }

        desc.descriptor_norms[current_type](count) =
            sqrt(r1 * r1 + r2 * r2 + r3 * r3);
        desc.neighbor_counts[current_type](count) = 2;
        desc.cumulative_neighbor_counts[current_type](count) = count * 2;
        desc.atom_indices[current_type](count) = i;
        desc.neighbor_indices[current_type](count * 2) = struc_index_1;
        desc.neighbor_indices[current_type](count * 2 + 1) = struc_index_2;

        type_counter(current_type)++;
      }
    }
  }

  return desc;
}

// TODO: Implement.
nlohmann::json ThreeBodyWide ::return_json(){
  nlohmann::json j;
  return j;
}
