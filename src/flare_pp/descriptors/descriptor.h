#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <Eigen/Dense>
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

class Structure;
class DescriptorValues;

class Descriptor {
public:
  Descriptor();

  std::string descriptor_name;

  virtual DescriptorValues compute_struc(Structure &structure) = 0;

  virtual ~Descriptor() = default;

  virtual void write_to_file(std::ofstream &coeff_file, int coeff_size);

  virtual nlohmann::json return_json() = 0;
};

void to_json(nlohmann::json& j, const std::vector<Descriptor*> & p);
void from_json(const nlohmann::json& j, std::vector<Descriptor*> & p);

// DescriptorValues holds the descriptor values for a single structure.
class DescriptorValues {
public:
  DescriptorValues();

  // Descriptor attributes.
  int n_descriptors, n_types, n_atoms;
  double volume;

  std::vector<Eigen::MatrixXd> descriptors, descriptor_force_dervs,
      neighbor_coordinates;
  std::vector<Eigen::VectorXd> descriptor_norms, descriptor_force_dots,
      cutoff_values, cutoff_dervs;
  std::vector<Eigen::VectorXi> neighbor_counts, cumulative_neighbor_counts,
      atom_indices, neighbor_indices;

  int n_clusters = 0;
  std::vector<int> n_clusters_by_type, cumulative_type_count,
      n_neighbors_by_type;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DescriptorValues,
    n_descriptors, n_types, n_atoms, volume, descriptors,
    descriptor_force_dervs, neighbor_coordinates, descriptor_norms,
    descriptor_force_dots, cutoff_values, cutoff_dervs, neighbor_counts,
    cumulative_neighbor_counts, atom_indices, neighbor_indices,
    n_clusters, n_clusters_by_type, cumulative_type_count,
    n_neighbors_by_type)
};

// ClusterDescriptor holds the descriptor values for a collection of clusters
// (excluding partial force derivatives).
class ClusterDescriptor {
public:
  ClusterDescriptor();
  ClusterDescriptor(const DescriptorValues &structure);

  // Specify clusters of each type.
  ClusterDescriptor(const DescriptorValues &structure,
                    const std::vector<std::vector<int>> &clusters);

  // Specify clusters of any type.
  ClusterDescriptor(const DescriptorValues &structure,
                    const std::vector<int> &clusters);

  std::vector<Eigen::MatrixXd> descriptors;
  std::vector<Eigen::VectorXd> descriptor_norms, cutoff_values;
  std::vector<int> n_clusters_by_type, cumulative_type_count;
  int n_descriptors, n_types;
  int n_clusters = 0;

  void initialize_cluster(int n_types, int n_descriptors);
  void add_clusters_by_type(const DescriptorValues &structure,
                            const std::vector<std::vector<int>> &clusters);
  void add_clusters(const DescriptorValues &structure,
                    const std::vector<int> &clusters);
  void add_all_clusters(const DescriptorValues &structure);

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ClusterDescriptor,
    descriptors, descriptor_norms, cutoff_values, n_clusters_by_type,
    cumulative_type_count, n_descriptors, n_types, n_clusters)
};

#endif
