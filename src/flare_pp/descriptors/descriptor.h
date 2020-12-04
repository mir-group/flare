#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <Eigen/Dense>
#include <vector>

class Structure;
class DescriptorValues;

class Descriptor {
public:
  Descriptor();

  virtual DescriptorValues compute_struc(Structure &structure) = 0;

  virtual ~Descriptor() = default;

  void write_to_file(std::ofstream &coeff_file, int coeff_size);
};

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

  std::vector<int> n_clusters_by_type, n_neighbors_by_type;
};

class ClusterDescriptor {
public:
  ClusterDescriptor();
  ClusterDescriptor(const DescriptorValues &structure);

  // Specify clusters of each type.
  ClusterDescriptor(const DescriptorValues &structure,
                    const std::vector<std::vector<int>> &clusters);

  // Specify clusters of any type.
  ClusterDescriptor(const DescriptorValues &structure,
                    const std::vector<int> & clusters);

  std::vector<Eigen::MatrixXd> descriptors;
  std::vector<Eigen::VectorXd> descriptor_norms, cutoff_values;
  std::vector<int> type_count, cumulative_type_count;
  int n_descriptors, n_types;
  int n_clusters = 0;

  void initialize_cluster(int n_types, int n_descriptors);
  void add_clusters_by_type(const DescriptorValues &structure,
                            const std::vector<std::vector<int>> &clusters);
  void add_clusters(const DescriptorValues &structure,
                    const std::vector<int> &clusters);
  void add_all_clusters(const DescriptorValues &structure);
};

#endif
