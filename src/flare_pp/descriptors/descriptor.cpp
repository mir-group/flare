#include "descriptor.h"
#include "cutoffs.h"
#include "radial.h"
#include "structure.h"
#include <cmath>
#include <iostream>

Descriptor::Descriptor() {}

DescriptorValues::DescriptorValues() {}

ClusterDescriptor::ClusterDescriptor() {}

ClusterDescriptor::ClusterDescriptor(const DescriptorValues &structure) {
  add_all_clusters(structure);
}

ClusterDescriptor::ClusterDescriptor(
  const DescriptorValues &structure,
  const std::vector<std::vector<int>> &clusters) {

  add_clusters(structure, clusters);
}

void ClusterDescriptor ::initialize_cluster(int n_types, int n_descriptors) {
  if (n_clusters != 0) return;

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

void ClusterDescriptor ::add_clusters(
  const DescriptorValues &structure,
  const std::vector<std::vector<int>> &clusters){

  initialize_cluster(structure.n_types, structure.n_descriptors);

  // Resize descriptor matrices.
  for (int s = 0; s < n_types; s++){
    descriptors[s].conservativeResize(
      type_count[s] + clusters[s].size(), n_descriptors);
    descriptor_norms[s].conservativeResize(type_count[s] + clusters[s].size());
    cutoff_values[s].conservativeResize(type_count[s] + clusters[s].size());
  }

  // Update descriptors.
  for (int s = 0; s < n_types; s++) {
    for (int i = 0; i < clusters[s].size(); i++) {
      descriptors[s].row(type_count[s] + i) =
        structure.descriptors[s].row(clusters[s][i]);
      descriptor_norms[s](type_count[s] + i) =
        structure.descriptor_norms[s](clusters[s][i]);
      cutoff_values[s](type_count[s] + i) =
        structure.cutoff_values[s](clusters[s][i]);
    }
  }

  // Update type counts.
  for (int s = 0; s < n_types; s++) {
    type_count[s] += clusters[s].size();
    n_clusters += clusters[s].size();
    if (s > 0)
      cumulative_type_count[s] =
          cumulative_type_count[s - 1] + type_count[s - 1];
  }
}

void ClusterDescriptor ::add_all_clusters(const DescriptorValues &structure) {

  initialize_cluster(structure.n_types, structure.n_descriptors);

  // Resize descriptor matrices.
  for (int s = 0; s < n_types; s++) {
    descriptors[s].conservativeResize(
        type_count[s] + structure.n_clusters_by_type[s], n_descriptors);
    descriptor_norms[s].conservativeResize(type_count[s] +
                                           structure.n_clusters_by_type[s]);
    cutoff_values[s].conservativeResize(type_count[s] +
                                        structure.n_clusters_by_type[s]);
  }

  // Update descriptors.
  for (int s = 0; s < n_types; s++) {
    for (int i = 0; i < structure.n_clusters_by_type[s]; i++) {
      descriptors[s].row(type_count[s] + i) = structure.descriptors[s].row(i);
      descriptor_norms[s](type_count[s] + i) = structure.descriptor_norms[s](i);
      cutoff_values[s](type_count[s] + i) = structure.cutoff_values[s](i);
    }
  }

  // Update type counts.
  for (int s = 0; s < n_types; s++) {
    type_count[s] += structure.n_clusters_by_type[s];
    n_clusters += structure.n_clusters_by_type[s];
    if (s > 0)
      cumulative_type_count[s] =
          cumulative_type_count[s - 1] + type_count[s - 1];
  }
}
