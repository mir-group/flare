#include "descriptor.h"
#include "cutoffs.h"
#include "radial.h"
#include "structure.h"
#include "b2.h"
#include <cmath>
#include <iostream>

Descriptor::Descriptor() {}

void Descriptor::write_to_file(std::ofstream &coeff_file, int coeff_size) {
  std::cout << "Mapping this descriptor is not implemented yet." << std::endl;
  return;
}

void to_json(nlohmann::json& j, const std::vector<Descriptor*> & p){
  int n_desc = p.size();
  for (int i = 0; i < n_desc; i++){
    j.push_back(p[i]->return_json());
  }
}

void from_json(const nlohmann::json& j, std::vector<Descriptor*> & p){
  int n_desc = j.size();
  for (int i = 0; i < n_desc; i++){
    nlohmann::json j_desc = j[i];
    std::string descriptor_name = j_desc.at("descriptor_name");
    if (descriptor_name == "B2"){
      // Consider using smart pointers instead to handle deallocation.
      B2* b2_pointer = new B2;
      *b2_pointer = j_desc;
      p.push_back(b2_pointer);
    }
    // TODO: Implement to/from json methods for remaining descriptors.
    else{
      p.push_back(nullptr);
    }
  }
}

DescriptorValues::DescriptorValues() {}

ClusterDescriptor::ClusterDescriptor() {}

ClusterDescriptor::ClusterDescriptor(const DescriptorValues &structure) {
  add_all_clusters(structure);
}

ClusterDescriptor::ClusterDescriptor(
    const DescriptorValues &structure,
    const std::vector<std::vector<int>> &clusters) {

  add_clusters_by_type(structure, clusters);
}

ClusterDescriptor::ClusterDescriptor(const DescriptorValues &structure,
                                     const std::vector<int> &clusters) {

  add_clusters(structure, clusters);
}

void ClusterDescriptor ::initialize_cluster(int n_types, int n_descriptors) {
  if (n_clusters != 0)
    return;

  this->n_types = n_types;
  this->n_descriptors = n_descriptors;

  Eigen::MatrixXd empty_mat;
  Eigen::VectorXd empty_vec;
  for (int s = 0; s < n_types; s++) {
    descriptors.push_back(empty_mat);
    descriptor_norms.push_back(empty_vec);
    cutoff_values.push_back(empty_vec);
    n_clusters_by_type.push_back(0);
    cumulative_type_count.push_back(0);
  }
}

void ClusterDescriptor ::add_clusters(const DescriptorValues &structure,
                                      const std::vector<int> &clusters) {

  // Determine the type of each cluster.
  std::vector<std::vector<int>> clusters_by_type(structure.n_types);
  for (int i = 0; i < clusters.size(); i++) {
    int cluster_val = clusters[i];
    int type, val;
    for (int j = 0; j < structure.n_types; j++) {
      int ccount = structure.cumulative_type_count[j];
      int ccount_p1 = structure.cumulative_type_count[j + 1];
      if ((cluster_val >= ccount) && (cluster_val < ccount_p1)) {
        type = j;
        val = cluster_val - ccount;
        break;
      }
    }
    clusters_by_type[type].push_back(val);
  }

  // Add clusters.
  add_clusters_by_type(structure, clusters_by_type);
}

void ClusterDescriptor ::add_clusters_by_type(
    const DescriptorValues &structure,
    const std::vector<std::vector<int>> &clusters) {

  initialize_cluster(structure.n_types, structure.n_descriptors);

  // Resize descriptor matrices.
  for (int s = 0; s < n_types; s++) {
    descriptors[s].conservativeResize(
        n_clusters_by_type[s] + clusters[s].size(), n_descriptors);
    descriptor_norms[s].conservativeResize(n_clusters_by_type[s] +
                                           clusters[s].size());
    cutoff_values[s].conservativeResize(n_clusters_by_type[s] +
                                        clusters[s].size());
  }

  // Update descriptors.
  for (int s = 0; s < n_types; s++) {
    for (int i = 0; i < clusters[s].size(); i++) {
      descriptors[s].row(n_clusters_by_type[s] + i) =
          structure.descriptors[s].row(clusters[s][i]);
      descriptor_norms[s](n_clusters_by_type[s] + i) =
          structure.descriptor_norms[s](clusters[s][i]);
      cutoff_values[s](n_clusters_by_type[s] + i) =
          structure.cutoff_values[s](clusters[s][i]);
    }
  }

  // Update type counts.
  for (int s = 0; s < n_types; s++) {
    n_clusters_by_type[s] += clusters[s].size();
    n_clusters += clusters[s].size();
    if (s > 0)
      cumulative_type_count[s] =
          cumulative_type_count[s - 1] + n_clusters_by_type[s - 1];
  }
}

void ClusterDescriptor ::add_all_clusters(const DescriptorValues &structure) {

  initialize_cluster(structure.n_types, structure.n_descriptors);

  // Resize descriptor matrices.
  for (int s = 0; s < n_types; s++) {
    descriptors[s].conservativeResize(
        n_clusters_by_type[s] + structure.n_clusters_by_type[s], n_descriptors);
    descriptor_norms[s].conservativeResize(n_clusters_by_type[s] +
                                           structure.n_clusters_by_type[s]);
    cutoff_values[s].conservativeResize(n_clusters_by_type[s] +
                                        structure.n_clusters_by_type[s]);
  }

  // Update descriptors.
  for (int s = 0; s < n_types; s++) {
    for (int i = 0; i < structure.n_clusters_by_type[s]; i++) {
      descriptors[s].row(n_clusters_by_type[s] + i) =
          structure.descriptors[s].row(i);
      descriptor_norms[s](n_clusters_by_type[s] + i) =
          structure.descriptor_norms[s](i);
      cutoff_values[s](n_clusters_by_type[s] + i) =
          structure.cutoff_values[s](i);
    }
  }

  // Update type counts.
  for (int s = 0; s < n_types; s++) {
    n_clusters_by_type[s] += structure.n_clusters_by_type[s];
    n_clusters += structure.n_clusters_by_type[s];
    if (s > 0)
      cumulative_type_count[s] =
          cumulative_type_count[s - 1] + n_clusters_by_type[s - 1];
  }
}
