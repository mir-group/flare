#include "compact_structure.h"

CompactStructures ::CompactStructures() {}

void CompactStructures ::add_structure(const CompactStructure &structure) {
  // If this is the first structure added, initialize descriptor matrices.
  if (n_strucs == 0) {
    n_species = structure.descriptors.size();
    n_descriptors = structure.descriptors[0].cols();
    for (int i = 0; i < n_species; i++) {
      descriptors.push_back(structure.descriptors[i]);
      descriptor_force_dervs.push_back(structure.descriptor_force_dervs[i]);
    }
  } else {
    // Update descriptor matrices.
    for (int i = 0; i < n_species; i++) {
      // Resize.
      int e_rows = descriptors[i].rows();
      int f_rows = descriptor_force_dervs[i].rows();

      int e_rows_struc = structure.descriptors[i].rows();
      int f_rows_struc = structure.descriptor_force_dervs[i].rows();

      descriptors[i].conservativeResize(e_rows + e_rows_struc, n_descriptors);
      descriptor_force_dervs[i].conservativeResize(f_rows + f_rows_struc,
                                                   n_descriptors);

      // Update.
      descriptors[i].block(e_rows, 0, e_rows_struc, n_descriptors) =
          structure.descriptors[i];
      descriptor_force_dervs[i].block(f_rows, 0, f_rows_struc, n_descriptors) =
          structure.descriptor_force_dervs[i];
    }
  }

  descriptor_norms.push_back(structure.descriptor_norms);
  force_dots.push_back(structure.descriptor_force_dots);

  local_neighbor_counts.push_back(structure.neighbor_counts);
  cumulative_local_neighbor_counts.push_back(
      structure.cumulative_neighbor_counts);
  atom_indices.push_back(structure.atom_indices);
  neighbor_indices.push_back(structure.neighbor_indices);

  // Update structure counts.
  atom_counts.push_back(structure.n_atoms_by_species);
  neighbor_counts.push_back(structure.n_neighbors_by_species);

  std::vector<int> cum_atom, cum_neigh;
  if (n_strucs == 0) {
    for (int i = 0; i < n_species; i++) {
      cum_atom.push_back(0);
      cum_neigh.push_back(0);
    }
  } else {
    for (int i = 0; i < n_species; i++) {
      cum_atom.push_back(cumulative_atom_counts[n_strucs - 1][i] +
                         atom_counts[n_strucs - 1][i]);
      cum_neigh.push_back(cumulative_neighbor_counts[n_strucs - 1][i] +
                          neighbor_counts[n_strucs - 1][i]);
    }
  }
  cumulative_atom_counts.push_back(cum_atom);
  cumulative_neighbor_counts.push_back(cum_neigh);
  n_atoms.push_back(structure.noa);
  if (n_strucs == 0)
    c_atoms.push_back(0);
  else
    c_atoms.push_back(c_atoms[n_strucs - 1] + n_atoms[n_strucs - 1]);

  n_strucs++;
}
