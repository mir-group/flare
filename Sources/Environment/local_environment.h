#ifndef LOCAL_ENVIRONMENT_H
#define LOCAL_ENVIRONMENT_H

#include "Descriptor/descriptor.h"
#include "Parameters/hyps_mask.h"
#include "Structure/structure.h"

// Local environment class.
class LocalEnvironment {
public:
  std::vector<int> environment_indices, environment_species, neighbor_list;
  int central_index, central_species, noa, sweep;
  std::vector<double> rs, xs, ys, zs, xrel, yrel, zrel;
  double cutoff, structure_volume;
  Structure structure;
  Eigen::VectorXd force; // Force on the central atom

  // Neighbor descriptors and derivatives.
  // TODO: add neighbor lists for each many body cutoff.
  std::vector<std::vector<Eigen::VectorXd>> neighbor_descriptors;
  std::vector<std::vector<Eigen::MatrixXd>> neighbor_force_dervs,
      neighbor_force_dots;
  std::vector<std::vector<double>> neighbor_descriptor_norms;

  // Store cutoffs for each kernel and indices of atoms inside each cutoff
  // sphere.
  std::vector<double> n_body_cutoffs, many_body_cutoffs;
  std::vector<std::vector<int>> n_body_indices, many_body_indices;

  // Triplet indices and cross bond distances are stored only if the 3-body
  // kernel is used.
  std::vector<std::vector<int>> three_body_indices;
  std::vector<double> cross_bond_dists;

  // Add attributes to ensure compatibility with the Python version.
  // 2-body attributes.
  std::vector<int> etypes;
  std::vector<int> bond_inds;
  Eigen::MatrixXd bond_array_2;

  // 3-body attributes.
  Eigen::MatrixXd bond_array_3;
  Eigen::MatrixXi cross_bond_inds;
  Eigen::MatrixXd cross_bond_dists_py;
  Eigen::VectorXi triplet_counts;

  // Store descriptor calculators for each many body cutoff.
  std::vector<DescriptorCalculator *> descriptor_calculators;
  std::vector<Eigen::VectorXd> descriptor_vals, descriptor_squared;
  std::vector<Eigen::MatrixXd> descriptor_force_dervs, descriptor_stress_dervs,
      force_dot, stress_dot;
  std::vector<double> descriptor_norm;

  // The cutoffs mask attribute gives the option to assign different pairs of
  // species different cutoffs.
  HypsMask cutoffs_mask;

  LocalEnvironment();

  LocalEnvironment(const Structure &structure, int atom, double cutoff);

  // To maintain consistency with the Python version, include a constructor that
  // takes in the cutoffs as a dictionary.
  LocalEnvironment(const Structure &structure, int atom,
                   std::unordered_map<std::string, double> cutoffs,
                   HypsMask cutoffs_mask = HypsMask{});

  // n-body
  LocalEnvironment(const Structure &structure, int atom, double cutoff,
                   std::vector<double> n_body_cutoffs);

  // many-body
  LocalEnvironment(const Structure &structure, int atom, double cutoff,
                   std::vector<double> many_body_cutoffs,
                   std::vector<DescriptorCalculator *> descriptor_calculators);

  // n-body + many-body
  LocalEnvironment(const Structure &structure, int atom, double cutoff,
                   std::vector<double> n_body_cutoffs,
                   std::vector<double> many_body_cutoffs,
                   std::vector<DescriptorCalculator *> descriptor_calculator);

  void compute_environment();

  // Compute descriptor and descriptor norm of a bare environment.
  void set_attributes(const Structure &structure, int atom, double cutoff);
  void compute_descriptors();
  void compute_descriptors_and_gradients();
  void compute_neighbor_descriptors();
  void compute_descriptor_squared();
  void compute_indices();
};

void compute_neighbor_descriptors(std::vector<LocalEnvironment> &envs);
void compute_descriptors(std::vector<LocalEnvironment> &envs);

#endif
