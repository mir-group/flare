#include "b2_norm.h"
#include "cutoffs.h"
#include "descriptor.h"
#include "radial.h"
#include "structure.h"
#include "y_grad.h"
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>

B2_Norm ::B2_Norm() {}

B2_Norm ::B2_Norm(const std::string &radial_basis,
                  const std::string &cutoff_function,
                  const std::vector<double> &radial_hyps,
                  const std::vector<double> &cutoff_hyps,
                  const std::vector<int> &descriptor_settings)
    : B2(radial_basis, cutoff_function, radial_hyps, cutoff_hyps,
         descriptor_settings) {}

DescriptorValues B2_Norm ::compute_struc(Structure &structure) {

  // Initialize descriptor values.
  DescriptorValues desc = DescriptorValues();

  // Compute single bond values.
  Eigen::MatrixXd single_bond_vals, force_dervs, neighbor_coords;
  Eigen::VectorXi unique_neighbor_count, cumulative_neighbor_count,
      descriptor_indices;

  int nos = descriptor_settings[0];
  int N = descriptor_settings[1];
  int lmax = descriptor_settings[2];

  compute_single_bond(single_bond_vals, force_dervs, neighbor_coords,
                      unique_neighbor_count, cumulative_neighbor_count,
                      descriptor_indices, radial_pointer, cutoff_pointer, nos,
                      N, lmax, radial_hyps, cutoff_hyps, structure);

  // Compute descriptor values.
  Eigen::MatrixXd B2_vals, B2_force_dervs;
  Eigen::VectorXd B2_norms, B2_force_dots;

  compute_b2_norm(B2_vals, B2_force_dervs, B2_norms, B2_force_dots,
                  single_bond_vals, force_dervs, unique_neighbor_count,
                  cumulative_neighbor_count, descriptor_indices, nos, N, lmax);

  // Gather species information.
  int noa = structure.noa;
  Eigen::VectorXi species_count = Eigen::VectorXi::Zero(nos);
  Eigen::VectorXi neighbor_count = Eigen::VectorXi::Zero(nos);
  for (int i = 0; i < noa; i++) {
    int s = structure.species[i];
    int n_neigh = unique_neighbor_count(i);
    species_count(s)++;
    neighbor_count(s) += n_neigh;
  }

  // Initialize arrays.
  int n_d = B2_vals.cols();
  desc.n_descriptors = n_d;
  desc.n_types = nos;
  desc.n_atoms = noa;
  desc.volume = structure.volume;
  desc.cumulative_type_count.push_back(0);
  for (int s = 0; s < nos; s++) {
    int n_s = species_count(s);
    int n_neigh = neighbor_count(s);

    // Record species and neighbor count.
    desc.n_clusters_by_type.push_back(n_s);
    desc.cumulative_type_count.push_back(desc.cumulative_type_count[s] + n_s);
    desc.n_clusters += n_s;
    desc.n_neighbors_by_type.push_back(n_neigh);

    desc.descriptors.push_back(Eigen::MatrixXd::Zero(n_s, n_d));
    desc.descriptor_force_dervs.push_back(
        Eigen::MatrixXd::Zero(n_neigh * 3, n_d));
    desc.neighbor_coordinates.push_back(Eigen::MatrixXd::Zero(n_neigh, 3));

    desc.cutoff_values.push_back(Eigen::VectorXd::Ones(n_s));
    desc.cutoff_dervs.push_back(Eigen::VectorXd::Zero(n_neigh * 3));
    desc.descriptor_norms.push_back(Eigen::VectorXd::Zero(n_s));
    desc.descriptor_force_dots.push_back(Eigen::VectorXd::Zero(n_neigh * 3));

    desc.neighbor_counts.push_back(Eigen::VectorXi::Zero(n_s));
    desc.cumulative_neighbor_counts.push_back(Eigen::VectorXi::Zero(n_s));
    desc.atom_indices.push_back(Eigen::VectorXi::Zero(n_s));
    desc.neighbor_indices.push_back(Eigen::VectorXi::Zero(n_neigh));
  }

  // Assign to structure.
  Eigen::VectorXi species_counter = Eigen::VectorXi::Zero(nos);
  Eigen::VectorXi neighbor_counter = Eigen::VectorXi::Zero(nos);
  for (int i = 0; i < noa; i++) {
    int s = structure.species[i];
    int s_count = species_counter(s);
    int n_neigh = unique_neighbor_count(i);
    int n_count = neighbor_counter(s);
    int cum_neigh = cumulative_neighbor_count(i);

    desc.descriptors[s].row(s_count) = B2_vals.row(i);
    desc.descriptor_force_dervs[s].block(n_count * 3, 0, n_neigh * 3, n_d) =
        B2_force_dervs.block(cum_neigh * 3, 0, n_neigh * 3, n_d);
    desc.neighbor_coordinates[s].block(n_count, 0, n_neigh, 3) =
        neighbor_coords.block(cum_neigh, 0, n_neigh, 3);

    desc.descriptor_norms[s](s_count) = B2_norms(i);
    desc.descriptor_force_dots[s].segment(n_count * 3, n_neigh * 3) =
        B2_force_dots.segment(cum_neigh * 3, n_neigh * 3);

    desc.neighbor_counts[s](s_count) = n_neigh;
    desc.cumulative_neighbor_counts[s](s_count) = n_count;
    desc.atom_indices[s](s_count) = i;
    desc.neighbor_indices[s].segment(n_count, n_neigh) =
        descriptor_indices.segment(cum_neigh, n_neigh);

    species_counter(s)++;
    neighbor_counter(s) += n_neigh;
  }

  return desc;
}

void compute_b2_norm(
  Eigen::MatrixXd &B2_vals, Eigen::MatrixXd &B2_force_dervs,
  Eigen::VectorXd &B2_norms, Eigen::VectorXd &B2_force_dots,
  const Eigen::MatrixXd &single_bond_vals,
  const Eigen::MatrixXd &single_bond_force_dervs,
  const Eigen::VectorXi &unique_neighbor_count,
  const Eigen::VectorXi &cumulative_neighbor_count,
  const Eigen::VectorXi &descriptor_indices, int nos, int N, int lmax) {

  int n_atoms = single_bond_vals.rows();
  int n_neighbors = cumulative_neighbor_count(n_atoms);
  int n_radial = nos * N;
  int n_harmonics = (lmax + 1) * (lmax + 1);
  int n_bond = n_radial * n_harmonics;
  int n_d = (n_radial * (n_radial + 1) / 2) * (lmax + 1);

  double empty_thresh = 1e-8;

  // Compute unnormalized B2 values.
  Eigen::MatrixXd B2_vals_1, B2_force_dervs_1;
  Eigen::VectorXd B2_norms_1, B2_force_dots_1;
  compute_b2(B2_vals_1, B2_force_dervs_1, B2_norms_1, B2_force_dots_1,
             single_bond_vals, single_bond_force_dervs, unique_neighbor_count,
             cumulative_neighbor_count, descriptor_indices, nos, N, lmax);

  // Normalize the descriptor values.
  B2_vals = Eigen::MatrixXd::Zero(n_atoms, n_d);
  B2_force_dervs = Eigen::MatrixXd::Zero(n_neighbors * 3, n_d);
  B2_norms = Eigen::VectorXd::Zero(n_atoms);
  B2_force_dots = Eigen::VectorXd::Zero(n_neighbors * 3);

  for (int i = 0; i < n_atoms; i++){
    double norm_val = B2_norms_1(i);

    // Continue if atom i has no neighbors.
    if (norm_val < empty_thresh)
      continue;

    double norm_val_3 = norm_val * norm_val * norm_val;
    B2_vals.row(i) = B2_vals_1.row(i) / norm_val;
    B2_norms(i) = 1.0;
    int n_atom_neighbors = unique_neighbor_count(i);
    int force_start = cumulative_neighbor_count(i) * 3;
    for (int j = 0; j < n_atom_neighbors; j++){
      for (int k = 0; k < 3; k++){
        int ind = force_start + j * 3 + k;
        B2_force_dervs.row(ind) =
          B2_force_dervs_1.row(ind) / norm_val -
          B2_vals_1.row(i) * B2_force_dots_1(ind) / norm_val_3;
      }
    }
    B2_force_dots.segment(force_start, n_atom_neighbors * 3) =
      B2_force_dervs.block(force_start, 0, n_atom_neighbors * 3, n_d) *
      B2_vals.row(i).transpose();
  }
}

// TODO: Implement.
nlohmann::json B2_Norm ::return_json(){
  nlohmann::json j;
  return j;
}
