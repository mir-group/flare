#include "two_body.h"
#include <iostream>

TwoBody ::TwoBody() {}

TwoBody ::TwoBody(double cutoff, int n_species){
    this->cutoff = cutoff;
    this->n_species = n_species;
}

DescriptorValues TwoBody ::compute_struc(CompactStructure &structure){

  // Initialize descriptor values.
  DescriptorValues desc = DescriptorValues();

  desc.n_descriptors = 1;
  desc.n_types = n_species * (n_species + 1) / 2;
  desc.n_atoms = structure.noa;
  desc.volume = structure.volume;
  int n_neighbors = structure.n_neighbors;

  // Count types.
  Eigen::VectorXi type_count = Eigen::VectorXi::Zero(desc.n_types);
  Eigen::VectorXi store_neighbors = Eigen::VectorXi::Zero(n_neighbors);
#pragma omp parallel for
  for (int i = 0; i < desc.n_atoms; i++) {
    int i_species = structure.species[i];
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      int j_species = structure.species[j];
      int neigh_index = rel_index + j;
      int struc_index = structure.structure_indices(neigh_index);
      // Avoid counting the same pair twice.
      if ((j_species > i_species) ||
          ((j_species == i_species) && (struc_index >= i))){
        int species_diff = j_species - i_species;
        int current_type =
          i_species * n_species - (i_species * (i_species - 1)) / 2 +
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

  std::cout << type_count << std::endl;

  // Initialize arrays.

  // Store descriptors.

  return desc;
}
