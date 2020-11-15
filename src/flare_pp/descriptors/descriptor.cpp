#include "descriptor.h"
#include "structure.h"
#include "cutoffs.h"
#include "radial.h"
#include <cmath>
#include <iostream>

Descriptor::Descriptor() {}

DescriptorValues::DescriptorValues() {}

ClusterDescriptor::ClusterDescriptor() {}

ClusterDescriptor::ClusterDescriptor(const DescriptorValues &structure) {
  add_cluster(structure);
}

void ClusterDescriptor ::initialize_cluster(int n_types, int n_descriptors) {
  this->n_types = n_types;
  this->n_descriptors = n_descriptors;

  Eigen::MatrixXd empty_mat;
  Eigen::VectorXd empty_vec;
  for (int s = 0; s < n_types; s++) {
    descriptors.push_back(empty_mat);
    descriptor_norms.push_back(empty_vec);
    cutoff_values.push_back(empty_vec);
    type_count.push_back(0);
    cumulative_type_count.push_back(0);
  }
}

void ClusterDescriptor ::add_cluster(const DescriptorValues &structure) {

  // If this is the first time adding a cluster, initialize attributes.
  if (n_clusters == 0) {
    initialize_cluster(structure.n_types, structure.n_descriptors);
  }

  // Resize descriptor matrices.
  for (int s = 0; s < n_types; s++) {
    descriptors[s].conservativeResize(
        type_count[s] + structure.n_atoms_by_type[s], n_descriptors);
    descriptor_norms[s].conservativeResize(type_count[s] +
                                           structure.n_atoms_by_type[s]);
    cutoff_values[s].conservativeResize(type_count[s] +
                                        structure.n_atoms_by_type[s]);
  }

  // Update descriptors.
  for (int s = 0; s < n_types; s++) {
    for (int i = 0; i < structure.n_atoms_by_type[s]; i++) {
      descriptors[s].row(type_count[s] + i) = structure.descriptors[s].row(i);
      descriptor_norms[s](type_count[s] + i) = structure.descriptor_norms[s](i);
      cutoff_values[s](type_count[s] + i) = structure.cutoff_values[s](i);
    }
  }

  // Update type counts.
  for (int s = 0; s < n_types; s++) {
    type_count[s] += structure.n_atoms_by_type[s];
    n_clusters += structure.n_atoms_by_type[s];
    if (s > 0)
      cumulative_type_count[s] =
          cumulative_type_count[s - 1] + type_count[s - 1];
  }
}
