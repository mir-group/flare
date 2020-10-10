#include "compact_environments.h"
#include "compact_structure.h"
#include <iostream>

CompactEnvironments ::CompactEnvironments() {}

void CompactEnvironments ::add_environments(const CompactStructure &structure,
                                            std::vector<int> environments) {

  // If first time adding environments, initialize attributes.
  if (n_envs == 0) {
    n_species = structure.descriptors.size();
    n_descriptors = structure.descriptors[0].cols();

    Eigen::MatrixXd empty_mat;
    std::vector<double> empty_vec;
    for (int s = 0; s < n_species; s++) {
      descriptors.push_back(empty_mat);
      descriptor_norms.push_back(empty_vec);
      n_atoms.push_back(0);
      c_atoms.push_back(0);
    }
  }

  // Count species.
  Eigen::ArrayXi species_count = Eigen::ArrayXi::Zero(n_species);
  for (int i = 0; i < environments.size(); i++) {
    int s = structure.species[environments[i]];
    species_count(s)++;
  }

  // Resize descriptor matrix.
  for (int s = 0; s < n_species; s++) {
    descriptors[s].conservativeResize(n_atoms[s] + species_count[s],
                                      n_descriptors);
  }

  // Update attributes.
  species_count = Eigen::ArrayXi::Zero(n_species);
  for (int i = 0; i < environments.size(); i++) {
    int atom_index = environments[i];
    int species_index = structure.species_indices(i);
    int s = structure.species[atom_index];
    descriptors[s].row(n_atoms[s] + species_count(s)) =
        structure.descriptors[s].row(species_index);
    descriptor_norms[s].push_back(structure.descriptor_norms[s](species_index));
    species_count(s)++;
  }

  for (int s = 0; s < n_species; s++) {
    n_atoms[s] += species_count(s);
    if (s > 0)
      c_atoms[s] = c_atoms[s - 1] + n_atoms[s - 1];
  }
  n_envs += environments.size();
}
