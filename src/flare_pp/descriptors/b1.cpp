#include "b1.h"
#include "b2.h"
#include "cutoffs.h"
#include "descriptor.h"
#include "radial.h"
#include "structure.h"
#include "y_grad.h"
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>

B1 ::B1() {}

B1 ::B1(const std::string &radial_basis, const std::string &cutoff_function,
        const std::vector<double> &radial_hyps,
        const std::vector<double> &cutoff_hyps,
        const std::vector<int> &descriptor_settings) {

  this->radial_basis = radial_basis;
  this->cutoff_function = cutoff_function;
  this->radial_hyps = radial_hyps;
  this->cutoff_hyps = cutoff_hyps;
  this->descriptor_settings = descriptor_settings;

  set_radial_basis(radial_basis, this->radial_pointer);
  set_cutoff(cutoff_function, this->cutoff_pointer);

  // Create cutoff matrix.
  int n_species = descriptor_settings[0];
  double cutoff_val = radial_hyps[1];
  cutoffs = Eigen::MatrixXd::Constant(n_species, n_species, cutoff_val);
}

B1 ::B1(const std::string &radial_basis, const std::string &cutoff_function,
        const std::vector<double> &radial_hyps,
        const std::vector<double> &cutoff_hyps,
        const std::vector<int> &descriptor_settings,
        const Eigen::MatrixXd &cutoffs) {

  this->radial_basis = radial_basis;
  this->cutoff_function = cutoff_function;
  this->radial_hyps = radial_hyps;
  this->cutoff_hyps = cutoff_hyps;
  this->descriptor_settings = descriptor_settings;

  set_radial_basis(radial_basis, this->radial_pointer);
  set_cutoff(cutoff_function, this->cutoff_pointer);

  // Assign cutoff matrix.
  this->cutoffs = cutoffs;
}

void B1 ::write_to_file(std::ofstream &coeff_file, int coeff_size) {
  // Report radial basis set.
  coeff_file << radial_basis << "\n";

  // Record number of species, nmax, lmax, and the cutoff.
  int n_species = descriptor_settings[0];
  int n_max = descriptor_settings[1];
  int l_max = 0;
  double cutoff = radial_hyps[1];

  coeff_file << n_species << " " << n_max << " " << l_max << " ";
  coeff_file << coeff_size << "\n";
  coeff_file << cutoff_function << "\n";

  // Report cutoffs to 2 decimal places.
  coeff_file << std::fixed << std::setprecision(2);
  for (int i = 0; i < n_species; i ++){
    for (int j = 0; j < n_species; j ++){
      coeff_file << cutoffs(i, j) << " ";
    }
  }
  coeff_file << "\n";
}

DescriptorValues B1 ::compute_struc(Structure &structure) {

  // Initialize descriptor values.
  DescriptorValues desc = DescriptorValues();

  // Compute single bond values.
  Eigen::MatrixXd single_bond_vals, force_dervs, neighbor_coords;
  Eigen::VectorXi unique_neighbor_count, cumulative_neighbor_count,
      descriptor_indices;

  int nos = descriptor_settings[0];
  int N = descriptor_settings[1];
  int lmax = 0;

  single_bond_multiple_cutoffs(
    single_bond_vals, force_dervs, neighbor_coords, unique_neighbor_count,
    cumulative_neighbor_count, descriptor_indices, radial_pointer,
    cutoff_pointer, nos, N, lmax, radial_hyps, cutoff_hyps, structure,
    cutoffs);

  // Compute descriptor values.
  Eigen::MatrixXd B1_vals, B1_force_dervs;
  Eigen::VectorXd B1_norms, B1_force_dots;

  compute_b1(B1_vals, B1_force_dervs, B1_norms, B1_force_dots, single_bond_vals,
             force_dervs, unique_neighbor_count, cumulative_neighbor_count,
             descriptor_indices, nos, N, lmax);

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
  int n_d = B1_vals.cols();
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

    desc.descriptors[s].row(s_count) = B1_vals.row(i);
    desc.descriptor_force_dervs[s].block(n_count * 3, 0, n_neigh * 3, n_d) =
        B1_force_dervs.block(cum_neigh * 3, 0, n_neigh * 3, n_d);
    desc.neighbor_coordinates[s].block(n_count, 0, n_neigh, 3) =
        neighbor_coords.block(cum_neigh, 0, n_neigh, 3);

    desc.descriptor_norms[s](s_count) = B1_norms(i);
    desc.descriptor_force_dots[s].segment(n_count * 3, n_neigh * 3) =
        B1_force_dots.segment(cum_neigh * 3, n_neigh * 3);

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

void compute_b1(Eigen::MatrixXd &B1_vals, Eigen::MatrixXd &B1_force_dervs,
                Eigen::VectorXd &B1_norms, Eigen::VectorXd &B1_force_dots,
                const Eigen::MatrixXd &single_bond_vals,
                const Eigen::MatrixXd &single_bond_force_dervs,
                const Eigen::VectorXi &unique_neighbor_count,
                const Eigen::VectorXi &cumulative_neighbor_count,
                const Eigen::VectorXi &descriptor_indices, int nos, int N,
                int lmax) {

  int n_atoms = single_bond_vals.rows();
  int n_neighbors = cumulative_neighbor_count(n_atoms);
  int n_radial = nos * N;
  assert(lmax == 0); // for b1, lmax = m = 0
  int n_harmonics = (lmax + 1) * (lmax + 1);
  int n_bond = n_radial * n_harmonics;
  int n_d = n_radial; 

  // Initialize arrays.
  B1_vals = Eigen::MatrixXd::Zero(n_atoms, n_d);
  B1_force_dervs = Eigen::MatrixXd::Zero(n_neighbors * 3, n_d);
  B1_norms = Eigen::VectorXd::Zero(n_atoms);
  B1_force_dots = Eigen::VectorXd::Zero(n_neighbors * 3);

#pragma omp parallel for
  for (int atom = 0; atom < n_atoms; atom++) {
    int n_atom_neighbors = unique_neighbor_count(atom);
    int force_start = cumulative_neighbor_count(atom) * 3;
    int n1, n2, l, m, n1_l, n2_l;
    int counter = 0;
    for (int n1 = 0; n1 < n_radial; n1++) {
      n1_l = n1 * n_harmonics;
      B1_vals(atom, counter) += single_bond_vals(atom, n1_l);

      // Store force derivatives.
      for (int n = 0; n < n_atom_neighbors; n++) {
        for (int comp = 0; comp < 3; comp++) {
          int ind = force_start + n * 3 + comp;
          B1_force_dervs(ind, counter) +=
              single_bond_force_dervs(ind, n1_l);
        }
      }
      counter++;
    }
    // Compute descriptor norm and force dot products.
    B1_norms(atom) = sqrt(B1_vals.row(atom).dot(B1_vals.row(atom)));
    B1_force_dots.segment(force_start, n_atom_neighbors * 3) =
        B1_force_dervs.block(force_start, 0, n_atom_neighbors * 3, n_d) *
        B1_vals.row(atom).transpose();
  }
}

// TODO: Implement.
nlohmann::json B1 ::return_json(){
  nlohmann::json j;
  return j;
}
