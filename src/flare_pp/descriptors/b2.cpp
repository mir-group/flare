#include "b2.h"
#include "cutoffs.h"
#include "descriptor.h"
#include "radial.h"
#include "structure.h"
#include "y_grad.h"
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>

B2 ::B2() {}

B2 ::B2(const std::string &radial_basis, const std::string &cutoff_function,
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

B2 ::B2(const std::string &radial_basis, const std::string &cutoff_function,
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

void B2 ::write_to_file(std::ofstream &coeff_file, int coeff_size) {
  // Report radial basis set.
  coeff_file << radial_basis << "\n";

  // Record number of species, nmax, lmax, and the cutoff.
  int n_species = descriptor_settings[0];
  int n_max = descriptor_settings[1];
  int l_max = descriptor_settings[2];
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

DescriptorValues B2 ::compute_struc(Structure &structure) {

  // Initialize descriptor values.
  DescriptorValues desc = DescriptorValues();

  // Compute single bond values.
  Eigen::MatrixXd single_bond_vals, force_dervs, neighbor_coords;
  Eigen::VectorXi unique_neighbor_count, cumulative_neighbor_count,
      descriptor_indices;

  int nos = descriptor_settings[0];
  int N = descriptor_settings[1];
  int lmax = descriptor_settings[2];

  single_bond_multiple_cutoffs(
    single_bond_vals, force_dervs, neighbor_coords, unique_neighbor_count,
    cumulative_neighbor_count, descriptor_indices, radial_pointer,
    cutoff_pointer, nos, N, lmax, radial_hyps, cutoff_hyps, structure,
    cutoffs);

  // Compute descriptor values.
  Eigen::MatrixXd B2_vals, B2_force_dervs;
  Eigen::VectorXd B2_norms, B2_force_dots;

  compute_b2(B2_vals, B2_force_dervs, B2_norms, B2_force_dots, single_bond_vals,
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

void compute_b2(Eigen::MatrixXd &B2_vals, Eigen::MatrixXd &B2_force_dervs,
                Eigen::VectorXd &B2_norms, Eigen::VectorXd &B2_force_dots,
                const Eigen::MatrixXd &single_bond_vals,
                const Eigen::MatrixXd &single_bond_force_dervs,
                const Eigen::VectorXi &unique_neighbor_count,
                const Eigen::VectorXi &cumulative_neighbor_count,
                const Eigen::VectorXi &descriptor_indices, int nos, int N,
                int lmax) {

  int n_atoms = single_bond_vals.rows();
  int n_neighbors = cumulative_neighbor_count(n_atoms);
  int n_radial = nos * N;
  int n_harmonics = (lmax + 1) * (lmax + 1);
  int n_bond = n_radial * n_harmonics;
  int n_d = (n_radial * (n_radial + 1) / 2) * (lmax + 1);

  // Initialize arrays.
  B2_vals = Eigen::MatrixXd::Zero(n_atoms, n_d);
  B2_force_dervs = Eigen::MatrixXd::Zero(n_neighbors * 3, n_d);
  B2_norms = Eigen::VectorXd::Zero(n_atoms);
  B2_force_dots = Eigen::VectorXd::Zero(n_neighbors * 3);

#pragma omp parallel for
  for (int atom = 0; atom < n_atoms; atom++) {
    int n_atom_neighbors = unique_neighbor_count(atom);
    int force_start = cumulative_neighbor_count(atom) * 3;
    int n1, n2, l, m, n1_l, n2_l;
    int counter = 0;
    for (int n1 = 0; n1 < n_radial; n1++) {
      for (int n2 = n1; n2 < n_radial; n2++) { // can be simplified
        for (int l = 0; l < (lmax + 1); l++) {
          for (int m = 0; m < (2 * l + 1); m++) {
            n1_l = n1 * n_harmonics + (l * l + m);
            n2_l = n2 * n_harmonics + (l * l + m);
            B2_vals(atom, counter) +=
                single_bond_vals(atom, n1_l) * single_bond_vals(atom, n2_l);

            // Store force derivatives.
            for (int n = 0; n < n_atom_neighbors; n++) {
              for (int comp = 0; comp < 3; comp++) {
                int ind = force_start + n * 3 + comp;
                B2_force_dervs(ind, counter) +=
                    single_bond_vals(atom, n1_l) *
                        single_bond_force_dervs(ind, n2_l) +
                    single_bond_force_dervs(ind, n1_l) *
                        single_bond_vals(atom, n2_l);
              }
            }
          }
          counter++;
        }
      }
    }
    // Compute descriptor norm and force dot products.
    B2_norms(atom) = sqrt(B2_vals.row(atom).dot(B2_vals.row(atom)));
    if (n_atom_neighbors > 0) {
      B2_force_dots.segment(force_start, n_atom_neighbors * 3) =
          B2_force_dervs.block(force_start, 0, n_atom_neighbors * 3, n_d) *
          B2_vals.row(atom).transpose();
    }
  }
}

void single_bond_multiple_cutoffs(
    Eigen::MatrixXd &single_bond_vals, Eigen::MatrixXd &force_dervs,
    Eigen::MatrixXd &neighbor_coordinates, Eigen::VectorXi &neighbor_count,
    Eigen::VectorXi &cumulative_neighbor_count,
    Eigen::VectorXi &neighbor_indices,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        radial_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int nos, int N, int lmax, const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, const Structure &structure,
    const Eigen::MatrixXd &cutoffs) {

  int n_atoms = structure.noa;
  int n_neighbors = structure.n_neighbors;

  // Count atoms inside the descriptor cutoff.
  neighbor_count = Eigen::VectorXi::Zero(n_atoms);
  Eigen::VectorXi store_neighbors = Eigen::VectorXi::Zero(n_neighbors);
#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    int central_species = structure.species[i];
    for (int j = 0; j < i_neighbors; j++) {
      int current_count = neighbor_count(i);
      int neigh_index = rel_index + j;
      int neighbor_species = structure.neighbor_species(neigh_index);
      double rcut = cutoffs(central_species, neighbor_species);
      double r = structure.relative_positions(neigh_index, 0);
      // Check that atom is within descriptor cutoff.
      if (r <= rcut) {
        int struc_index = structure.structure_indices(neigh_index);
        // Update neighbor list.
        store_neighbors(rel_index + current_count) = struc_index;
        neighbor_count(i)++;
      }
    }
  }

  // Count cumulative number of unique neighbors.
  cumulative_neighbor_count = Eigen::VectorXi::Zero(n_atoms + 1);
  for (int i = 1; i < n_atoms + 1; i++) {
    cumulative_neighbor_count(i) +=
        cumulative_neighbor_count(i - 1) + neighbor_count(i - 1);
  }

  // Record neighbor indices.
  int bond_neighbors = cumulative_neighbor_count(n_atoms);
  neighbor_indices = Eigen::VectorXi::Zero(bond_neighbors);
#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = neighbor_count(i);
    int ind1 = cumulative_neighbor_count(i);
    int ind2 = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      neighbor_indices(ind1 + j) = store_neighbors(ind2 + j);
    }
  }

  // Initialize single bond arrays.
  int number_of_harmonics = (lmax + 1) * (lmax + 1);
  int no_bond_vals = N * number_of_harmonics;
  int single_bond_size = no_bond_vals * nos;

  single_bond_vals = Eigen::MatrixXd::Zero(n_atoms, single_bond_size);
  force_dervs = Eigen::MatrixXd::Zero(bond_neighbors * 3, single_bond_size);
  neighbor_coordinates = Eigen::MatrixXd::Zero(bond_neighbors, 3);

#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    int neighbor_index = cumulative_neighbor_count(i);
    int central_species = structure.species[i];

    // Initialize radial hyperparameters.
    std::vector<double> new_radial_hyps = radial_hyps;

    // Initialize radial and spherical harmonic vectors.
    std::vector<double> g = std::vector<double>(N, 0);
    std::vector<double> gx = std::vector<double>(N, 0);
    std::vector<double> gy = std::vector<double>(N, 0);
    std::vector<double> gz = std::vector<double>(N, 0);

    std::vector<double> h = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hx = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hy = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hz = std::vector<double>(number_of_harmonics, 0);

    double x, y, z, r, bond, bond_x, bond_y, bond_z, g_val, gx_val, gy_val,
        gz_val, h_val;
    int s, neigh_index, descriptor_counter, unique_ind;
    for (int j = 0; j < i_neighbors; j++) {
      neigh_index = rel_index + j;
      int neighbor_species = structure.neighbor_species(neigh_index);
      double rcut = cutoffs(central_species, neighbor_species);
      r = structure.relative_positions(neigh_index, 0);
      if (r > rcut)
        continue; // Skip if outside cutoff.
      x = structure.relative_positions(neigh_index, 1);
      y = structure.relative_positions(neigh_index, 2);
      z = structure.relative_positions(neigh_index, 3);
      s = structure.neighbor_species(neigh_index);

      // Reset the endpoint of the radial basis set.
      new_radial_hyps[1] = rcut;

      // Store neighbor coordinates.
      neighbor_coordinates(neighbor_index, 0) = x;
      neighbor_coordinates(neighbor_index, 1) = y;
      neighbor_coordinates(neighbor_index, 2) = z;

      // Compute radial basis values and spherical harmonics.
      calculate_radial(g, gx, gy, gz, radial_function, cutoff_function, x, y, z,
                       r, rcut, N, new_radial_hyps, cutoff_hyps);
      get_Y(h, hx, hy, hz, x, y, z, lmax);

      // Store the products and their derivatives.
      descriptor_counter = s * no_bond_vals;

      for (int radial_counter = 0; radial_counter < N; radial_counter++) {
        // Retrieve radial values.
        g_val = g[radial_counter];
        gx_val = gx[radial_counter];
        gy_val = gy[radial_counter];
        gz_val = gz[radial_counter];

        for (int angular_counter = 0; angular_counter < number_of_harmonics;
             angular_counter++) {

          // Compute single bond value.
          h_val = h[angular_counter];
          bond = g_val * h_val;

          // Calculate derivatives with the product rule.
          bond_x = gx_val * h_val + g_val * hx[angular_counter];
          bond_y = gy_val * h_val + g_val * hy[angular_counter];
          bond_z = gz_val * h_val + g_val * hz[angular_counter];

          // Update single bond arrays.
          single_bond_vals(i, descriptor_counter) += bond;

          force_dervs(neighbor_index * 3, descriptor_counter) += bond_x;
          force_dervs(neighbor_index * 3 + 1, descriptor_counter) += bond_y;
          force_dervs(neighbor_index * 3 + 2, descriptor_counter) += bond_z;

          descriptor_counter++;
        }
      }
      neighbor_index++;
    }
  }
}

void compute_single_bond(
    Eigen::MatrixXd &single_bond_vals, Eigen::MatrixXd &force_dervs,
    Eigen::MatrixXd &neighbor_coordinates, Eigen::VectorXi &neighbor_count,
    Eigen::VectorXi &cumulative_neighbor_count,
    Eigen::VectorXi &neighbor_indices,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        radial_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int nos, int N, int lmax, const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, const Structure &structure) {

  int n_atoms = structure.noa;
  int n_neighbors = structure.n_neighbors;

  // TODO: Make rcut an attribute of the descriptor calculator.
  double rcut = radial_hyps[1];

  // Count atoms inside the descriptor cutoff.
  neighbor_count = Eigen::VectorXi::Zero(n_atoms);
  Eigen::VectorXi store_neighbors = Eigen::VectorXi::Zero(n_neighbors);
#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      int current_count = neighbor_count(i);
      int neigh_index = rel_index + j;
      double r = structure.relative_positions(neigh_index, 0);
      // Check that atom is within descriptor cutoff.
      if (r <= rcut) {
        int struc_index = structure.structure_indices(neigh_index);
        // Update neighbor list.
        store_neighbors(rel_index + current_count) = struc_index;
        neighbor_count(i)++;
      }
    }
  }

  // Count cumulative number of unique neighbors.
  cumulative_neighbor_count = Eigen::VectorXi::Zero(n_atoms + 1);
  for (int i = 1; i < n_atoms + 1; i++) {
    cumulative_neighbor_count(i) +=
        cumulative_neighbor_count(i - 1) + neighbor_count(i - 1);
  }

  // Record neighbor indices.
  int bond_neighbors = cumulative_neighbor_count(n_atoms);
  neighbor_indices = Eigen::VectorXi::Zero(bond_neighbors);
#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = neighbor_count(i);
    int ind1 = cumulative_neighbor_count(i);
    int ind2 = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      neighbor_indices(ind1 + j) = store_neighbors(ind2 + j);
    }
  }

  // Initialize single bond arrays.
  int number_of_harmonics = (lmax + 1) * (lmax + 1);
  int no_bond_vals = N * number_of_harmonics;
  int single_bond_size = no_bond_vals * nos;

  single_bond_vals = Eigen::MatrixXd::Zero(n_atoms, single_bond_size);
  force_dervs = Eigen::MatrixXd::Zero(bond_neighbors * 3, single_bond_size);
  neighbor_coordinates = Eigen::MatrixXd::Zero(bond_neighbors, 3);

#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    int neighbor_index = cumulative_neighbor_count(i);

    // Initialize radial and spherical harmonic vectors.
    std::vector<double> g = std::vector<double>(N, 0);
    std::vector<double> gx = std::vector<double>(N, 0);
    std::vector<double> gy = std::vector<double>(N, 0);
    std::vector<double> gz = std::vector<double>(N, 0);

    std::vector<double> h = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hx = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hy = std::vector<double>(number_of_harmonics, 0);
    std::vector<double> hz = std::vector<double>(number_of_harmonics, 0);

    double x, y, z, r, bond, bond_x, bond_y, bond_z, g_val, gx_val, gy_val,
        gz_val, h_val;
    int s, neigh_index, descriptor_counter, unique_ind;
    for (int j = 0; j < i_neighbors; j++) {
      neigh_index = rel_index + j;
      r = structure.relative_positions(neigh_index, 0);
      if (r > rcut)
        continue; // Skip if outside cutoff.
      x = structure.relative_positions(neigh_index, 1);
      y = structure.relative_positions(neigh_index, 2);
      z = structure.relative_positions(neigh_index, 3);
      s = structure.neighbor_species(neigh_index);

      // Store neighbor coordinates.
      neighbor_coordinates(neighbor_index, 0) = x;
      neighbor_coordinates(neighbor_index, 1) = y;
      neighbor_coordinates(neighbor_index, 2) = z;

      // Compute radial basis values and spherical harmonics.
      calculate_radial(g, gx, gy, gz, radial_function, cutoff_function, x, y, z,
                       r, rcut, N, radial_hyps, cutoff_hyps);
      get_Y(h, hx, hy, hz, x, y, z, lmax);

      // Store the products and their derivatives.
      descriptor_counter = s * no_bond_vals;

      for (int radial_counter = 0; radial_counter < N; radial_counter++) {
        // Retrieve radial values.
        g_val = g[radial_counter];
        gx_val = gx[radial_counter];
        gy_val = gy[radial_counter];
        gz_val = gz[radial_counter];

        for (int angular_counter = 0; angular_counter < number_of_harmonics;
             angular_counter++) {

          // Compute single bond value.
          h_val = h[angular_counter];
          bond = g_val * h_val;

          // Calculate derivatives with the product rule.
          bond_x = gx_val * h_val + g_val * hx[angular_counter];
          bond_y = gy_val * h_val + g_val * hy[angular_counter];
          bond_z = gz_val * h_val + g_val * hz[angular_counter];

          // Update single bond arrays.
          single_bond_vals(i, descriptor_counter) += bond;

          force_dervs(neighbor_index * 3, descriptor_counter) += bond_x;
          force_dervs(neighbor_index * 3 + 1, descriptor_counter) += bond_y;
          force_dervs(neighbor_index * 3 + 2, descriptor_counter) += bond_z;

          descriptor_counter++;
        }
      }
      neighbor_index++;
    }
  }
}

void to_json(nlohmann::json& j, const B2 & p){
  j = nlohmann::json{
    {"radial_basis", p.radial_basis},
    {"cutoff_function", p.cutoff_function},
    {"radial_hyps", p.radial_hyps},
    {"cutoff_hyps", p.cutoff_hyps},
    {"descriptor_settings", p.descriptor_settings},
    {"cutoffs", p.cutoffs},
    {"descriptor_name", p.descriptor_name}
  };
}

void from_json(const nlohmann::json& j, B2 & p){
  p = B2(
    j.at("radial_basis"),
    j.at("cutoff_function"),
    j.at("radial_hyps"),
    j.at("radial_hyps"),
    j.at("descriptor_settings"),
    j.at("cutoffs")
  );
}

nlohmann::json B2 ::return_json(){
  nlohmann::json j;
  to_json(j, *this);
  return j;
}
