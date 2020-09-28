#include "descriptor.h"
#include "cutoffs.h"
#include "local_environment.h"
#include "radial.h"
#include "single_bond.h"
#include <cmath>
#include <iostream>

DescriptorCalculator::DescriptorCalculator() {}

DescriptorCalculator::DescriptorCalculator(
    const std::string &radial_basis, const std::string &cutoff_function,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps,
    const std::vector<int> &descriptor_settings, int descriptor_index) {

  this->radial_basis = radial_basis;
  this->cutoff_function = cutoff_function;
  this->radial_hyps = radial_hyps;
  this->cutoff_hyps = cutoff_hyps;
  this->descriptor_settings = descriptor_settings;
  this->descriptor_index = descriptor_index;

  // Set the radial basis.
  if (radial_basis == "chebyshev") {
    this->radial_pointer = chebyshev;
  } else if (radial_basis == "weighted_chebyshev") {
    this->radial_pointer = weighted_chebyshev;
  } else if (radial_basis == "equispaced_gaussians") {
    this->radial_pointer = equispaced_gaussians;
  } else if (radial_basis == "weighted_positive_chebyshev") {
    this->radial_pointer = weighted_positive_chebyshev;
  } else if (radial_basis == "positive_chebyshev") {
    this->radial_pointer = positive_chebyshev;
  }

  // Set the cutoff function.
  if (cutoff_function == "quadratic") {
    this->cutoff_pointer = quadratic_cutoff;
  } else if (cutoff_function == "hard") {
    this->cutoff_pointer = hard_cutoff;
  } else if (cutoff_function == "cosine") {
    this->cutoff_pointer = cos_cutoff;
  }
}

void DescriptorCalculator::destroy_matrices() {
  single_bond_vals.resize(0);
  descriptor_vals.resize(0);
  single_bond_force_dervs.resize(0, 0);
  single_bond_stress_dervs.resize(0, 0);
  descriptor_force_dervs.resize(0, 0);
  descriptor_stress_dervs.resize(0, 0);
}

void B2_descriptor_struc(Eigen::VectorXd &B2_vals,
                         Eigen::MatrixXd &B2_force_dervs,
                         Eigen::MatrixXd &B2_stress_dervs,
                         const Eigen::VectorXd &single_bond_vals,
                         const Eigen::MatrixXd &single_bond_force_dervs,
                         const Eigen::MatrixXd &single_bond_stress_dervs,
                         const Eigen::VectorXi &unique_neighbor_count,
                         const Eigen::VectorXi &cumulative_neighbor_count,
                         const Eigen::VectorXi &descriptor_indices,
                         const CompactStructure &structure,
                         int nos, int N, int lmax){

  int n_atoms = single_bond_vals.rows();
  int n_neighbors = cumulative_neighbor_count(n_atoms);
  int n_radial = nos * N;
  int n_harmonics = (lmax + 1) * (lmax + 1);
  int n_bond = n_radial * n_harmonics;
  int n_descriptors = (n_radial * (n_radial + 1) / 2) * (lmax + 1);

  // Initialize matrices.
  B2_vals = Eigen::VectorXd::Zero(n_descriptors);
  B2_force_dervs = Eigen::MatrixXd::Zero(n_neighbors * 3, n_descriptors);
  B2_stress_dervs = Eigen::MatrixXd::Zero(n_atoms * 6, n_descriptors);

  for (int atom = 0; atom < n_atoms; atom++){
    int n1, n2, l, m, n1_l, n2_l;
    int counter = 0;
    for (int n1 = 0; n1 < n_radial; n1++){
      for (int n2 = n1; n2 < n_radial; n2++){
        for (int l = 0; l < (lmax + 1); l++) {
          for (int m = 0; m < (2 * l + 1); m++) {
            n1_l = n1 * n_harmonics + (l * l + m);
            n2_l = n2 * n_harmonics + (l * l + m);
            B2_vals(counter) += single_bond_vals(n1_l) * single_bond_vals(n2_l);
          }
          counter ++;
        }
      }
    }
  }
}

void B2_descriptor(Eigen::VectorXd &B2_vals, Eigen::MatrixXd &B2_force_dervs,
                   Eigen::MatrixXd &B2_stress_dervs,
                   const Eigen::VectorXd &single_bond_vals,
                   const Eigen::MatrixXd &single_bond_force_dervs,
                   const Eigen::MatrixXd &single_bond_stress_dervs,
                   const LocalEnvironment &env, int nos, int N, int lmax) {

  int neigh_size = env.neighbor_list.size();
  int cent_ind = env.central_index;
  int no_radial = nos * N;
  int no_harmonics = (lmax + 1) * (lmax + 1);

  int n1_l, n2_l, env_ind;
  int counter;
  int n1_count;
  int n2_count;

  for (int n1 = no_radial - 1; n1 >= 0; n1--) {
    for (int n2 = n1; n2 < no_radial; n2++) {
      for (int l = 0; l < (lmax + 1); l++) {
        for (int m = 0; m < (2 * l + 1); m++) {
          n1_l = n1 * no_harmonics + (l * l + m);
          n2_l = n2 * no_harmonics + (l * l + m);

          n1_count = (n1 * (2 * no_radial - n1 + 1)) / 2;
          n2_count = n2 - n1;
          counter = l + (n1_count + n2_count) * (lmax + 1);

          // Store B2 value.
          B2_vals(counter) += single_bond_vals(n1_l) * single_bond_vals(n2_l);

          // Store force derivatives.
          // TODO: loop over many body indices, not entire neighbor list
          for (int atom_index = 0; atom_index < neigh_size; atom_index++) {
            env_ind = env.neighbor_list[atom_index];
            for (int comp = 0; comp < 3; comp++) {
              B2_force_dervs(env_ind * 3 + comp, counter) +=
                  single_bond_vals(n1_l) *
                      single_bond_force_dervs(env_ind * 3 + comp, n2_l) +
                  single_bond_force_dervs(env_ind * 3 + comp, n1_l) *
                      single_bond_vals(n2_l);
            }
          }

          // Store stress derivatives.
          for (int p = 0; p < 6; p++) {
            B2_stress_dervs(p, counter) +=
                single_bond_vals(n1_l) * single_bond_stress_dervs(p, n2_l) +
                single_bond_stress_dervs(p, n1_l) * single_bond_vals(n2_l);
          }
        }
      }
    }
  }
};

B1_Calculator ::B1_Calculator() {}

B1_Calculator ::B1_Calculator(const std::string &radial_basis,
                              const std::string &cutoff_function,
                              const std::vector<double> &radial_hyps,
                              const std::vector<double> &cutoff_hyps,
                              const std::vector<int> &descriptor_settings,
                              int descriptor_index)
    : DescriptorCalculator(radial_basis, cutoff_function, radial_hyps,
                           cutoff_hyps, descriptor_settings, descriptor_index) {
}

void B1_Calculator ::compute_struc(CompactStructure &structure) {}

void B1_Calculator ::compute(const LocalEnvironment &env) {
  // Initialize single bond vectors.
  int nos = descriptor_settings[0];
  int N = descriptor_settings[1];
  int lmax = 0;
  int no_descriptors = nos * N;

  single_bond_vals = Eigen::VectorXd::Zero(no_descriptors);
  single_bond_force_dervs = Eigen::MatrixXd::Zero(env.noa * 3, no_descriptors);
  single_bond_stress_dervs = Eigen::MatrixXd::Zero(6, no_descriptors);

  // Compute single bond vector.
  single_bond_sum_env(single_bond_vals, single_bond_force_dervs,
                      single_bond_stress_dervs, radial_pointer, cutoff_pointer,
                      env, descriptor_index, N, lmax, radial_hyps, cutoff_hyps);

  // Set B1 values.
  descriptor_vals = single_bond_vals;
  descriptor_force_dervs = single_bond_force_dervs;
  descriptor_stress_dervs = single_bond_stress_dervs;
}

B2_Calculator ::B2_Calculator() {}

B2_Calculator ::B2_Calculator(const std::string &radial_basis,
                              const std::string &cutoff_function,
                              const std::vector<double> &radial_hyps,
                              const std::vector<double> &cutoff_hyps,
                              const std::vector<int> &descriptor_settings,
                              int descriptor_index)
    : DescriptorCalculator(radial_basis, cutoff_function, radial_hyps,
                           cutoff_hyps, descriptor_settings, descriptor_index) {
}

void B2_Calculator ::compute_struc(CompactStructure &structure) {
    // Assign descriptors and descriptor gradients to structure.
    // Organize by species.

    // Compute single bond values.
    Eigen::MatrixXd single_bond_vals, force_dervs, stress_dervs;
    Eigen::VectorXi unique_neighbor_count, cumulative_neighbor_count,
        descriptor_indices;

    single_bond_sum_struc(single_bond_vals, force_dervs, stress_dervs,
        unique_neighbor_count, cumulative_neighbor_count, descriptor_indices,
        structure, descriptor_index);
    
    // Compute descriptor values.
}

void B2_Calculator ::compute(const LocalEnvironment &env) {
  // Initialize single bond vectors.
  int nos = descriptor_settings[0];
  int N = descriptor_settings[1];
  int lmax = descriptor_settings[2];
  int no_radial = nos * N;

  int no_harmonics = (lmax + 1) * (lmax + 1);

  int no_bond = no_radial * no_harmonics;
  int no_descriptors = (no_radial * (no_radial + 1) / 2) * (lmax + 1);

  single_bond_vals = Eigen::VectorXd::Zero(no_bond);
  single_bond_force_dervs = Eigen::MatrixXd::Zero(env.noa * 3, no_bond);
  single_bond_stress_dervs = Eigen::MatrixXd::Zero(6, no_bond);

  // Compute single bond vector.
  single_bond_sum_env(single_bond_vals, single_bond_force_dervs,
                      single_bond_stress_dervs, radial_pointer, cutoff_pointer,
                      env, descriptor_index, N, lmax, radial_hyps, cutoff_hyps);

  // Initialize B2 vectors.
  descriptor_vals = Eigen::VectorXd::Zero(no_descriptors);
  // Note: Can reduce memory by only storing gradients of atoms in the
  // environment, rather than all atoms in the structure.
  descriptor_force_dervs = Eigen::MatrixXd::Zero(env.noa * 3, no_descriptors);
  descriptor_stress_dervs = Eigen::MatrixXd::Zero(6, no_descriptors);

  B2_descriptor(descriptor_vals, descriptor_force_dervs,
                descriptor_stress_dervs, single_bond_vals,
                single_bond_force_dervs, single_bond_stress_dervs, env, nos, N,
                lmax);
}
