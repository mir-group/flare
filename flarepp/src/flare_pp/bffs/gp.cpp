#include "gp.h"
#include <algorithm> // Random shuffle
#include <chrono>
#include <iostream>
#include <numeric> // Iota

GP ::GP() {}

GP ::GP(std::vector<Kernel *> kernels, double energy_noise, double force_noise,
        double stress_noise) {

  this->kernels = kernels;
  n_kernels = kernels.size();
  label_count = Eigen::VectorXd::Zero(1);

  // Count hyperparameters.
  int n_hyps = 0;
  for (int i = 0; i < kernels.size(); i++) {
    n_hyps += kernels[i]->kernel_hyperparameters.size();
  }

  // Set the kernel hyperparameters.
  hyperparameters = Eigen::VectorXd::Zero(n_hyps + 3);
  Eigen::VectorXd hyps_curr;
  int hyp_counter = 0;
  for (int i = 0; i < kernels.size(); i++) {
    hyps_curr = kernels[i]->kernel_hyperparameters;

    for (int j = 0; j < hyps_curr.size(); j++) {
      hyperparameters(hyp_counter) = hyps_curr(j);
      hyp_counter++;
    }
  }

  // Set the noise hyperparameters.
  hyperparameters(n_hyps) = energy_noise;
  hyperparameters(n_hyps + 1) = force_noise;
  hyperparameters(n_hyps + 2) = stress_noise;

  this->energy_noise = energy_noise;
  this->force_noise = force_noise;
  this->stress_noise = stress_noise;

  // Initialize kernel lists.
  Eigen::MatrixXd empty_matrix;
  for (int i = 0; i < kernels.size(); i++) {
    Kff_kernels.push_back(empty_matrix);
  }
}

// TODO: Finish implementing.
void GP ::add_training_structure(const Structure &structure) {

  int n_energy = structure.energy.size();
  int n_force = structure.forces.size();
  int n_stress = structure.stresses.size();
  int n_struc_labels = n_energy + n_force + n_stress;
  int n_atoms = structure.noa;

  // Update Kff kernels.
  for (int i = 0; i < n_kernels; i++) {
    // Resize Kff matrix.
    Kff_kernels[i].conservativeResize(n_labels + n_struc_labels,
                                      n_labels + n_struc_labels);

    // Compute self block.
    Eigen::MatrixXd self_kernels = kernels[i]->struc_struc(
        structure.descriptors[i], structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);

    assign_kernels(self_kernels, Kff_kernels[i], n_energy, n_force, n_stress,
                   n_atoms, n_energy, n_force, n_stress, n_atoms, n_labels,
                   n_labels);

    // // Compute kernels between new structure and previous structures.
    // TODO: Store cumulative label counts to parallelize this loop.
    // for (int j = 0; j < n_strucs; j++){
    //   Eigen::MatrixXd struc_kernels =
    //     kernels[i]->struc_struc(
    //       structure.descriptors[i], training_structures[j].descriptors[i],
    //       kernels[i]->kernel_hyperparameters);
    // }

    // struc_struc_kernels =
    //     kernels[i]->envs_struc(sparse_descriptors[i],
    //     structure.descriptors[i],
    //                            kernels[i]->kernel_hyperparameters);

    // Kff_kernels[i].block(0, n_labels, n_sparse, n_energy) =
    //     envs_struc_kernels.block(0, 0, n_sparse, n_energy);
    // Kff_kernels[i].block(0, n_labels + n_energy, n_sparse, n_force) =
    //     envs_struc_kernels.block(0, 1, n_sparse, n_force);
    // Kff_kernels[i].block(0, n_labels + n_energy + n_force, n_sparse,
    // n_stress) =
    //     envs_struc_kernels.block(0, 1 + n_atoms * 3, n_sparse, n_sparse);
  }

  //   // Update labels.
  //   label_count.conservativeResize(training_structures.size() + 2);
  //   label_count(training_structures.size() + 1) = n_labels + n_struc_labels;
  //   y.conservativeResize(n_labels + n_struc_labels);
  //   y.segment(n_labels, n_energy) = structure.energy;
  //   y.segment(n_labels + n_energy, n_force) = structure.forces;
  //   y.segment(n_labels + n_energy + n_force, n_stress) = structure.stresses;

  //   // Update noise.
  //   noise_vector.conservativeResize(n_labels + n_struc_labels);
  //   noise_vector.segment(n_labels, n_energy) =
  //       Eigen::VectorXd::Constant(n_energy, 1 / (energy_noise *
  //       energy_noise));
  //   noise_vector.segment(n_labels + n_energy, n_force) =
  //       Eigen::VectorXd::Constant(n_force, 1 / (force_noise * force_noise));
  //   noise_vector.segment(n_labels + n_energy + n_force, n_stress) =
  //       Eigen::VectorXd::Constant(n_stress, 1 / (stress_noise *
  //       stress_noise));

  //   // Update label count.
  //   n_energy_labels += n_energy;
  //   n_force_labels += n_force;
  //   n_stress_labels += n_stress;
  //   n_labels += n_struc_labels;

  //   // Store training structure.
  //   training_structures.push_back(structure);
  //   n_strucs += 1;

  //   // Update Kuf.
  //   stack_Kuf();
}

void assign_kernels(const Eigen::MatrixXd &efs_kernels,
                    Eigen::MatrixXd &Kff_kernels, int ne1, int nf1, int ns1,
                    int na1, int ne2, int nf2, int ns2, int na2, int row,
                    int col) {

  // EE, FF, SS kernels.
  Kff_kernels.block(row, col, ne1, ne2) = efs_kernels.block(0, 0, ne1, ne1);
  Kff_kernels.block(row + ne1, col + ne2, nf1, nf2) =
      efs_kernels.block(1, 1, nf1, nf2);
  Kff_kernels.block(row + ne1 + nf1, col + ne2 + nf2, ns1, ns2) =
      efs_kernels.block(1 + na1 * 3, 1 + na2 * 3, ns1, ns2);

  // EF, FE kernels.
  Kff_kernels.block(row, col + ne2, ne1, nf2) =
      efs_kernels.block(0, 1, ne1, nf2);
  Kff_kernels.block(row + ne1, col, nf1, ne2) =
      efs_kernels.block(1, 0, nf1, ne2);

  // ES, SE kernels.
  Kff_kernels.block(row, col + ne2 + nf2, ne1, ns2) =
      efs_kernels.block(0, 1 + 3 * na2, ne1, ns2);
  Kff_kernels.block(row + ne1 + nf1, col, ns1, ne2) =
      efs_kernels.block(1 + 3 * na1, 0, ns1, ne2);

  // FS, SF kernels.
  Kff_kernels.block(row + ne1, col + ne2 + nf2, nf1, ns2) =
      efs_kernels.block(1, 1 + 3 * na2, nf1, ns2);
  Kff_kernels.block(row + ne1 + nf1, col + ne2, ns1, nf2) =
      efs_kernels.block(1 + 3 * na1, 1, ns1, nf2);
}
