#include "sparse_gp.h"
#include <algorithm> // Random shuffle
#include <chrono>
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>
#include <numeric> // Iota
#include <assert.h> 

#define MAXLINE 1024

SparseGP ::SparseGP() {}

SparseGP ::SparseGP(std::vector<Kernel *> kernels, double energy_noise,
                    double force_noise, double stress_noise) {

  this->kernels = kernels;
  n_kernels = kernels.size();
  Kuu_jitter = 1e-8; // default value
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
    Kuu_kernels.push_back(empty_matrix);
    Kuf_kernels.push_back(empty_matrix);
  }
}

void SparseGP ::initialize_sparse_descriptors(const Structure &structure) {
  if (sparse_descriptors.size() != 0)
    return;

  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor empty_descriptor;
    empty_descriptor.initialize_cluster(structure.descriptors[i].n_types,
                                        structure.descriptors[i].n_descriptors);
    sparse_descriptors.push_back(empty_descriptor);
    std::vector<std::vector<int>> empty_indices;
    sparse_indices.push_back(empty_indices); // NOTE: the sparse_indices should be of size n_kernels
  }
};

std::vector<std::vector<int>>
SparseGP ::sort_clusters_by_uncertainty(const Structure &structure) {

  // Compute cluster uncertainties.
  std::vector<Eigen::VectorXd> variances =
      compute_cluster_uncertainties(structure);

  std::vector<std::vector<int>> sorted_indices;
  for (int i = 0; i < n_kernels; i++) {
    // Sort cluster indices by decreasing uncertainty.
    std::vector<int> indices(variances[i].size());
    iota(indices.begin(), indices.end(), 0);
    Eigen::VectorXd v = variances[i];
    stable_sort(indices.begin(), indices.end(),
                [&v](int i1, int i2) { return v(i1) > v(i2); });
    sorted_indices.push_back(indices);
  }

  return sorted_indices;
}

std::vector<Eigen::VectorXd>
SparseGP ::compute_cluster_uncertainties(const Structure &structure) {
  // TODO: this only computes the energy-energy variance, and the Sigma matrix is not considered?

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Compute cluster uncertainties.
  std::vector<Eigen::VectorXd> K_self, Q_self, variances;
  std::vector<Eigen::MatrixXd> sparse_kernels;
  int sparse_count = 0;
  for (int i = 0; i < n_kernels; i++) {
    K_self.push_back(
        (kernels[i]->envs_envs(cluster_descriptors[i], cluster_descriptors[i],
                               kernels[i]->kernel_hyperparameters))
            .diagonal());

    sparse_kernels.push_back(
        kernels[i]->envs_envs(cluster_descriptors[i], sparse_descriptors[i],
                              kernels[i]->kernel_hyperparameters));

    int n_clusters = sparse_descriptors[i].n_clusters;
    Eigen::MatrixXd L_inverse_block =
        L_inv.block(sparse_count, sparse_count, n_clusters, n_clusters);
    sparse_count += n_clusters;

    Eigen::MatrixXd Q1 = L_inverse_block * sparse_kernels[i].transpose();
    Q_self.push_back((Q1.transpose() * Q1).diagonal());

    variances.push_back(K_self[i] - Q_self[i]); // it is sorted by clusters, not the original atomic order 
    // TODO: If the environment is empty, the assigned uncertainty should be
    // set to zero.
  }

  return variances;
}

void SparseGP ::add_specific_environments(const Structure &structure,
                                          const std::vector<int> atoms) {

  initialize_sparse_descriptors(structure);

  // Gather clusters with central atom in the given list.
  std::vector<std::vector<std::vector<int>>> indices_1;
  for (int i = 0; i < n_kernels; i++){
    sparse_indices[i].push_back(atoms); // for each kernel the added atoms are the same

    int n_types = structure.descriptors[i].n_types;
    std::vector<std::vector<int>> indices_2;
    for (int j = 0; j < n_types; j++){
      int n_clusters = structure.descriptors[i].n_clusters_by_type[j];
      std::vector<int> indices_3;
      for (int k = 0; k < n_clusters; k++){
        int atom_index_1 = structure.descriptors[i].atom_indices[j](k);
        for (int l = 0; l < atoms.size(); l++){
          int atom_index_2 = atoms[l];
          if (atom_index_1 == atom_index_2){
            indices_3.push_back(k);
          }
        }
      }
      indices_2.push_back(indices_3);
    }
    indices_1.push_back(indices_2);
  }

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < n_kernels; i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i], indices_1[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Update Kuu and Kuf.
  update_Kuu(cluster_descriptors);
  update_Kuf(cluster_descriptors);
  stack_Kuu();
  stack_Kuf();

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_clusters_by_type(structure.descriptors[i],
                                               indices_1[i]);
  }
}

void SparseGP ::add_uncertain_environments(const Structure &structure,
                                           const std::vector<int> &n_added) {

  initialize_sparse_descriptors(structure);
  // Compute cluster uncertainties.
  std::vector<std::vector<int>> sorted_indices =
      sort_clusters_by_uncertainty(structure);

  std::vector<std::vector<int>> n_sorted_indices;
  for (int i = 0; i < n_kernels; i++) {
    // Take the first N indices.
    int n_curr = n_added[i];
    if (n_curr > sorted_indices[i].size())
      n_curr = sorted_indices[i].size();
    std::vector<int> n_indices(n_curr);
    for (int j = 0; j < n_curr; j++) {
      n_indices[j] = sorted_indices[i][j];
    }
    n_sorted_indices.push_back(n_indices);
  }

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < n_kernels; i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i], n_sorted_indices[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Update Kuu and Kuf.
  update_Kuu(cluster_descriptors);
  update_Kuf(cluster_descriptors);
  stack_Kuu();
  stack_Kuf();

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_clusters(structure.descriptors[i],
                                       n_sorted_indices[i]);

    // find the atom index of added sparse env
    std::vector<int> added_indices;
    for (int k = 0; k < n_sorted_indices[i].size(); k++) {
      int cluster_val = n_sorted_indices[i][k];
      int atom_index, val;
      for (int j = 0; j < structure.descriptors[i].n_types; j++) {
        int ccount = structure.descriptors[i].cumulative_type_count[j];
        int ccount_p1 = structure.descriptors[i].cumulative_type_count[j + 1];
        if ((cluster_val >= ccount) && (cluster_val < ccount_p1)) {
          val = cluster_val - ccount;
          atom_index = structure.descriptors[i].atom_indices[j][val];
          added_indices.push_back(atom_index);
          break;
        }
      }
    }

    sparse_indices[i].push_back(added_indices);
  }
}

void SparseGP ::add_random_environments(const Structure &structure,
                                        const std::vector<int> &n_added) {

  initialize_sparse_descriptors(structure);
  // Randomly select environments without replacement.
  std::vector<std::vector<int>> envs1;
  for (int i = 0; i < structure.descriptors.size(); i++) { // NOTE: n_kernels might be diff from descriptors number
    std::vector<int> envs2;
    int n_clusters = structure.descriptors[i].n_clusters;
    std::vector<int> clusters(n_clusters);
    std::iota(clusters.begin(), clusters.end(), 0);
    std::random_shuffle(clusters.begin(), clusters.end());
    int n_curr = n_added[i];
    if (n_curr > n_clusters)
      n_curr = n_clusters;
    for (int k = 0; k < n_curr; k++) {
      envs2.push_back(clusters[k]);
    }
    envs1.push_back(envs2);
  }

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i], envs1[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Update Kuu and Kuf.
  update_Kuu(cluster_descriptors);
  update_Kuf(cluster_descriptors);
  stack_Kuu();
  stack_Kuf();

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_clusters(structure.descriptors[i], envs1[i]);

    // find the atom index of added sparse env
    std::vector<int> added_indices;
    for (int k = 0; k < envs1[i].size(); k++) {
      int cluster_val = envs1[i][k];
      int atom_index, val;
      for (int j = 0; j < structure.descriptors[i].n_types; j++) {
        int ccount = structure.descriptors[i].cumulative_type_count[j];
        int ccount_p1 = structure.descriptors[i].cumulative_type_count[j + 1];
        if ((cluster_val >= ccount) && (cluster_val < ccount_p1)) {
          val = cluster_val - ccount;
          atom_index = structure.descriptors[i].atom_indices[j][val];
          added_indices.push_back(atom_index);
          break;
        }
      }
    }
    sparse_indices[i].push_back(added_indices);
  }
}

void SparseGP ::add_all_environments(const Structure &structure) {
  initialize_sparse_descriptors(structure);

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Update Kuu and Kuf.
  update_Kuu(cluster_descriptors);
  update_Kuf(cluster_descriptors);
  stack_Kuu();
  stack_Kuf();

  // Store sparse environments.
  std::vector<int> added_indices;
  for (int j = 0; j < structure.noa; j++) {
    added_indices.push_back(j);
  }
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_all_clusters(structure.descriptors[i]);
    sparse_indices[i].push_back(added_indices);
  }
}

void SparseGP ::update_Kuu(
    const std::vector<ClusterDescriptor> &cluster_descriptors) {

  // Update Kuu matrices.
  for (int i = 0; i < n_kernels; i++) {
    Eigen::MatrixXd prev_block =
        kernels[i]->envs_envs(sparse_descriptors[i], cluster_descriptors[i],
                              kernels[i]->kernel_hyperparameters);
    Eigen::MatrixXd self_block =
        kernels[i]->envs_envs(cluster_descriptors[i], cluster_descriptors[i],
                              kernels[i]->kernel_hyperparameters);

    int n_sparse = sparse_descriptors[i].n_clusters;
    int n_envs = cluster_descriptors[i].n_clusters;
    int n_types = cluster_descriptors[i].n_types;

    Eigen::MatrixXd kern_mat =
        Eigen::MatrixXd::Zero(n_sparse + n_envs, n_sparse + n_envs);

    int n1 = 0; // Sparse descriptor counter 1
    int n2 = 0; // Cluster descriptor counter 1

    // TODO: Generalize to allow comparisons across types.
    for (int j = 0; j < n_types; j++) {
      int n3 = 0; // Sparse descriptor counter 2
      int n4 = 0; // Cluster descriptor counter 2
      int n5 = sparse_descriptors[i].n_clusters_by_type[j];
      int n6 = cluster_descriptors[i].n_clusters_by_type[j];

      for (int k = 0; k < n_types; k++){
        int n7 = sparse_descriptors[i].n_clusters_by_type[k];
        int n8 = cluster_descriptors[i].n_clusters_by_type[k];

        Eigen::MatrixXd prev_vals_1 = prev_block.block(n1, n4, n5, n8);
        Eigen::MatrixXd prev_vals_2 = prev_block.block(n3, n2, n7, n6);
        Eigen::MatrixXd self_vals = self_block.block(n2, n4, n6, n8);

        kern_mat.block(n1 + n2, n3 + n4, n5, n7) =
          Kuu_kernels[i].block(n1, n3, n5, n7);
        kern_mat.block(n1 + n2, n3 + n4 + n7, n5, n8) =
          prev_vals_1;
        kern_mat.block(n1 + n2 + n5, n3 + n4, n6, n7) =
          prev_vals_2.transpose();
        kern_mat.block(n1 + n2 + n5, n3 + n4 + n7, n6, n8) =
          self_vals;

        n3 += n7;
        n4 += n8;
      }
      n1 += n5;
      n2 += n6;
    }
    Kuu_kernels[i] = kern_mat;

    // Update sparse count.
    this->n_sparse += n_envs;
    }
  }

void SparseGP ::update_Kuf(
    const std::vector<ClusterDescriptor> &cluster_descriptors) {

  // Compute kernels between new sparse environments and training structures.
  for (int i = 0; i < n_kernels; i++) {
    int n_sparse = sparse_descriptors[i].n_clusters;
    int n_envs = cluster_descriptors[i].n_clusters;
    int n_types = cluster_descriptors[i].n_types;

    // Precompute indices.
    Eigen::ArrayXi inds = Eigen::ArrayXi::Zero(n_types + 1);
    int counter = 0;
    for (int j = 0; j < n_types; j++) {
      int t1 = sparse_descriptors[i].n_clusters_by_type[j];
      int t2 = cluster_descriptors[i].n_clusters_by_type[j];
      counter += t1 + t2;
      inds(j + 1) = counter;
    }

    Eigen::MatrixXd kern_mat =
        Eigen::MatrixXd::Zero(n_sparse + n_envs, n_labels);

#pragma omp parallel for
    for (int j = 0; j < n_strucs; j++) {
      int n_atoms = training_structures[j].noa;
      Eigen::MatrixXd envs_struc_kernels = kernels[i]->envs_struc(
          cluster_descriptors[i], training_structures[j].descriptors[i],
          kernels[i]->kernel_hyperparameters);

      int n1 = 0; // Sparse descriptor count
      int n2 = 0; // Cluster descriptor count
      for (int k = 0; k < n_types; k++) {
        int current_count = 0;
        int u_ind = inds(k);
        int n3 = sparse_descriptors[i].n_clusters_by_type[k];
        int n4 = cluster_descriptors[i].n_clusters_by_type[k];

        if (training_structures[j].energy.size() != 0) {
          kern_mat.block(u_ind, label_count(j), n3, 1) =
              Kuf_kernels[i].block(n1, label_count(j), n3, 1);
          kern_mat.block(u_ind + n3, label_count(j), n4, 1) =
              envs_struc_kernels.block(n2, 0, n4, 1);

          current_count += 1;
        }

        if (training_structures[j].forces.size() != 0) {
          std::vector<int> atom_indices = training_atom_indices[j];
          for (int a = 0; a < atom_indices.size(); a++) {  // Allow adding a subset of force labels
            kern_mat.block(u_ind, label_count(j) + current_count, n3, 3) =
                Kuf_kernels[i].block(n1, label_count(j) + current_count, n3, 3);
            kern_mat.block(u_ind + n3, label_count(j) + current_count, n4, 3) =
                envs_struc_kernels.block(n2, 1 + atom_indices[a] * 3, n4, 3);
            current_count += 3;
          }
        }

        if (training_structures[j].stresses.size() != 0) {
          kern_mat.block(u_ind, label_count(j) + current_count, n3, 6) =
              Kuf_kernels[i].block(n1, label_count(j) + current_count, n3, 6);
          kern_mat.block(u_ind + n3, label_count(j) + current_count, n4, 6) =
              envs_struc_kernels.block(n2, 1 + n_atoms * 3, n4, 6);
        }

        n1 += n3;
        n2 += n4;
      }
    }
    Kuf_kernels[i] = kern_mat;
  }
}

void SparseGP ::add_training_structure(const Structure &structure,
                                       const std::vector<int> atom_indices, 
                                       double rel_e_noise,
                                       double rel_f_noise,
                                       double rel_s_noise) {
  // Allow adding a subset of force labels
  initialize_sparse_descriptors(structure);

  int n_atoms = structure.noa;
  int n_energy = structure.energy.size();
  int n_force = 0;
  std::vector<int> atoms;
  if (atom_indices[0] == -1) { // add all atoms
    n_force = structure.forces.size();
    for (int i = 0; i < n_atoms; i++) {
      atoms.push_back(i);
    }
  } else {
    atoms = atom_indices;
    n_force = atoms.size() * 3;
  }
  training_atom_indices.push_back(atoms);
  int n_stress = structure.stresses.size();
  int n_struc_labels = n_energy + n_force + n_stress;

  // Update labels.
  label_count.conservativeResize(training_structures.size() + 2);
  label_count(training_structures.size() + 1) = n_labels + n_struc_labels;
  y.conservativeResize(n_labels + n_struc_labels);
  y.segment(n_labels, n_energy) = structure.energy;
  y.segment(n_labels + n_energy + n_force, n_stress) = structure.stresses;
  for (int a = 0; a < atoms.size(); a++) {
    y.segment(n_labels + n_energy + a * 3, 3) = structure.forces.segment(atoms[a] * 3, 3);
  }

  // Update noise.
  noise_vector.conservativeResize(n_labels + n_struc_labels);
  noise_vector.segment(n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, 1 / (energy_noise * energy_noise * rel_e_noise * rel_e_noise));
  noise_vector.segment(n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, 1 / (force_noise * force_noise * rel_f_noise * rel_f_noise));
  noise_vector.segment(n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, 1 / (stress_noise * stress_noise * rel_s_noise * rel_s_noise));

  // Save "1" vector for energy, force and stress noise, for likelihood gradient calculation
  e_noise_one.conservativeResize(n_labels + n_struc_labels);
  f_noise_one.conservativeResize(n_labels + n_struc_labels);
  s_noise_one.conservativeResize(n_labels + n_struc_labels);

  e_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);
  f_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);
  s_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);

  e_noise_one.segment(n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, 1 / (rel_e_noise * rel_e_noise));
  f_noise_one.segment(n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, 1 / (rel_f_noise * rel_f_noise));
  s_noise_one.segment(n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, 1 / (rel_s_noise * rel_s_noise));

  inv_e_noise_one.conservativeResize(n_labels + n_struc_labels);
  inv_f_noise_one.conservativeResize(n_labels + n_struc_labels);
  inv_s_noise_one.conservativeResize(n_labels + n_struc_labels);

  inv_e_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);
  inv_f_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);
  inv_s_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);

  inv_e_noise_one.segment(n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, rel_e_noise * rel_e_noise);
  inv_f_noise_one.segment(n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, rel_f_noise * rel_f_noise);
  inv_s_noise_one.segment(n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, rel_s_noise * rel_s_noise);

  // Update Kuf kernels.
  Eigen::MatrixXd envs_struc_kernels;
  for (int i = 0; i < n_kernels; i++) {
    int n_sparse = sparse_descriptors[i].n_clusters;

    envs_struc_kernels = // contain all atoms
        kernels[i]->envs_struc(sparse_descriptors[i], structure.descriptors[i],
                               kernels[i]->kernel_hyperparameters);

    Kuf_kernels[i].conservativeResize(n_sparse, n_labels + n_struc_labels);
    Kuf_kernels[i].block(0, n_labels, n_sparse, n_energy) =
        envs_struc_kernels.block(0, 0, n_sparse, n_energy);
    Kuf_kernels[i].block(0, n_labels + n_energy + n_force, n_sparse, n_stress) =
        envs_struc_kernels.block(0, 1 + n_atoms * 3, n_sparse, n_stress);

    // Only add forces from `atoms`
    for (int a = 0; a < atoms.size(); a++) {
      Kuf_kernels[i].block(0, n_labels + n_energy + a * 3, n_sparse, 3) =
          envs_struc_kernels.block(0, 1 + atoms[a] * 3, n_sparse, 3); // if n_energy=0, we can not use n_energy but 1
    }
  }

  // Update label count.
  n_energy_labels += n_energy;
  n_force_labels += n_force;
  n_stress_labels += n_stress;
  n_labels += n_struc_labels;

  // Store training structure.
  training_structures.push_back(structure);
  n_strucs += 1;

  // Update Kuf.
  stack_Kuf();
}

void SparseGP ::stack_Kuu() {
  // Update Kuu.
  Kuu = Eigen::MatrixXd::Zero(n_sparse, n_sparse);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    Kuu.block(count, count, size, size) = Kuu_kernels[i];
    count += size;
  }
}

void SparseGP ::stack_Kuf() {
  // Update Kuf kernels.
  Kuf = Eigen::MatrixXd::Zero(n_sparse, n_labels);
  int count = 0;
  for (int i = 0; i < Kuf_kernels.size(); i++) {
    int size = Kuf_kernels[i].rows();
    Kuf.block(count, 0, size, n_labels) = Kuf_kernels[i];
    count += size;
  }
}

void SparseGP ::update_matrices_QR() {
  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = sqrt(noise_vector.array());

  // Cholesky decompose Kuu.
  Eigen::LLT<Eigen::MatrixXd> chol(
      Kuu + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()));

  // Get the inverse of Kuu from Cholesky decomposition.
  Eigen::MatrixXd Kuu_eye = Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
  L_inv = chol.matrixL().solve(Kuu_eye);
  L_diag = L_inv.diagonal();
  Kuu_inverse = L_inv.transpose() * L_inv;

  // Form A matrix.
  Eigen::MatrixXd A =
      Eigen::MatrixXd::Zero(Kuf.cols() + Kuu.cols(), Kuu.cols());
  A.block(0, 0, Kuf.cols(), Kuu.cols()) =
      noise_vector_sqrt.asDiagonal() * Kuf.transpose();
  A.block(Kuf.cols(), 0, Kuu.cols(), Kuu.cols()) = chol.matrixL().transpose();

  // Form b vector.
  Eigen::VectorXd b = Eigen::VectorXd::Zero(Kuf.cols() + Kuu.cols());
  b.segment(0, Kuf.cols()) = noise_vector_sqrt.asDiagonal() * y;

  // QR decompose A.
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
  Eigen::VectorXd Q_b = qr.householderQ().transpose() * b;
  R_inv = qr.matrixQR().block(0, 0, Kuu.cols(), Kuu.cols())
                       .triangularView<Eigen::Upper>()
                       .solve(Kuu_eye);
  R_inv_diag = R_inv.diagonal();
  alpha = R_inv * Q_b;
  Sigma = R_inv * R_inv.transpose();
}

void SparseGP ::predict_mean(Structure &test_structure) {

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;
}

void SparseGP ::predict_SOR(Structure &test_structure) {

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;
  Eigen::MatrixXd variance_sqrt = kernel_mat.transpose() * R_inv;
  test_structure.variance_efs =
      (variance_sqrt * variance_sqrt.transpose()).diagonal();
}

void SparseGP ::predict_DTC(Structure &test_structure) {

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;

  // Compute variances.
  Eigen::VectorXd V_SOR, Q_self, K_self = Eigen::VectorXd::Zero(n_out);

  for (int i = 0; i < n_kernels; i++) {
    K_self += kernels[i]->self_kernel_struc(test_structure.descriptors[i],
                                            kernels[i]->kernel_hyperparameters);
  }

  Q_self = (kernel_mat.transpose() * Kuu_inverse * kernel_mat).diagonal();
  V_SOR = (kernel_mat.transpose() * Sigma * kernel_mat).diagonal();

  test_structure.variance_efs = K_self - Q_self + V_SOR;
}

void SparseGP ::predict_local_uncertainties(Structure &test_structure) {
  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;

  std::vector<Eigen::VectorXd> local_uncertainties =
    compute_cluster_uncertainties(test_structure);
  test_structure.local_uncertainties = local_uncertainties;

}

void SparseGP ::compute_likelihood_stable() {
  // Compute inverse of Qff from Sigma.
  Eigen::MatrixXd noise_diag = noise_vector.asDiagonal();

  data_fit =
      -(1. / 2.) * y.transpose() * noise_diag * (y - Kuf.transpose() * alpha);
  constant_term = -(1. / 2.) * n_labels * log(2 * M_PI);

  // Compute complexity penalty.
  double noise_det = 0;
  for (int i = 0; i < noise_vector.size(); i++) {
    noise_det += log(noise_vector(i));
  }

  double Kuu_inv_det = 0;
  for (int i = 0; i < L_diag.size(); i++) {
    Kuu_inv_det -= 2 * log(abs(L_diag(i)));
  }

  double sigma_inv_det = 0;
  for (int i = 0; i < R_inv_diag.size(); i++) {
    sigma_inv_det += 2 * log(abs(R_inv_diag(i)));
  }

  complexity_penalty = (1. / 2.) * (noise_det + Kuu_inv_det + sigma_inv_det);
  std::cout << "like comp data " << complexity_penalty << " " << data_fit << std::endl;
  std::cout << "noise_det Kuu_inv_det sigma_inv_det " << noise_det << " " << Kuu_inv_det << " " << sigma_inv_det << std::endl;
  log_marginal_likelihood = complexity_penalty + data_fit + constant_term;
}

double SparseGP ::compute_likelihood_gradient_stable(bool precomputed_KnK) {

  // Compute training data fitting loss
  Eigen::VectorXd K_alpha = Kuf.transpose() * alpha;
  Eigen::VectorXd y_K_alpha = y - K_alpha;
  data_fit =
      -(1. / 2.) * y.transpose() * noise_vector.cwiseProduct(y_K_alpha);
  constant_term = -(1. / 2.) * n_labels * log(2 * M_PI);

  // Compute complexity penalty.
  double noise_det = 0;
  for (int i = 0; i < noise_vector.size(); i++) {
    noise_det += log(noise_vector(i));
  }

  assert(L_diag.size() == R_inv_diag.size());
  double Kuu_inv_det = 0;
  double sigma_inv_det = 0;
//#pragma omp parallel for
  for (int i = 0; i < L_diag.size(); i++) {
    Kuu_inv_det -= 2 * log(abs(L_diag(i)));
    sigma_inv_det += 2 * log(abs(R_inv_diag(i)));
  }

  complexity_penalty = (1. / 2.) * (noise_det + Kuu_inv_det + sigma_inv_det);
  log_marginal_likelihood = complexity_penalty + data_fit + constant_term;

  // Compute Kuu and Kuf matrices and gradients.
  int n_hyps_total = hyperparameters.size();

  //Eigen::MatrixXd Kuu_mat = Eigen::MatrixXd::Zero(n_sparse, n_sparse);
  //Eigen::MatrixXd Kuf_mat = Eigen::MatrixXd::Zero(n_sparse, n_labels);

  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad, Kuu_grads, Kuf_grads;

  int n_hyps, hyp_index = 0, grad_index = 0;
  Eigen::VectorXd hyps_curr;

  int count = 0;
  Eigen::VectorXd complexity_grad = Eigen::VectorXd::Zero(n_hyps_total);
  Eigen::VectorXd datafit_grad = Eigen::VectorXd::Zero(n_hyps_total);
  likelihood_gradient = Eigen::VectorXd::Zero(n_hyps_total);
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    hyps_curr = hyperparameters.segment(hyp_index, n_hyps);
    int size = Kuu_kernels[i].rows();

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu_kernels[i], hyps_curr);
    if (!precomputed_KnK) { 
      Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                      i, Kuf_kernels[i], hyps_curr);
    }

    //Kuu_mat.block(count, count, size, size) = Kuu_grad[0];
    //Kuf_mat.block(count, 0, size, n_labels) = Kuf_grad[0];
    Eigen::MatrixXd Kuu_i = Kuu_grad[0];

    for (int j = 0; j < n_hyps; j++) {
      Kuu_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_sparse));
      Kuu_grads[hyp_index + j].block(count, count, size, size) =
          Kuu_grad[j + 1];

      if (!precomputed_KnK) { 
        Kuf_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_labels));
        Kuf_grads[hyp_index + j].block(count, 0, size, n_labels) =
            Kuf_grad[j + 1];
      }

      // Compute Pi matrix and save as an intermediate variable for derivative of complexity
      Eigen::MatrixXd dK_noise_K;
      if (precomputed_KnK) {
        dK_noise_K = compute_dKnK(i);
      } else {
        Eigen::MatrixXd noise_diag = noise_vector.asDiagonal();
        dK_noise_K = Kuf_grads[hyp_index + j] * noise_diag * Kuf.transpose();
      }
      Eigen::MatrixXd Pi_mat = dK_noise_K + dK_noise_K.transpose() + Kuu_grads[hyp_index + j]; 

      // Derivative of complexity over sigma
      // TODO: the 2nd term is not very stable numerically, because dK_noise_K is very large, and Kuu_grads is small
      complexity_grad(hyp_index + j) += 1./2. * (Kuu_i.inverse() * Kuu_grad[j + 1]).trace() - 1./2. * (Pi_mat * Sigma).trace(); 

      // Derivative of data_fit over sigma
      Eigen::VectorXd dK_alpha;
      if (precomputed_KnK) {
        Eigen::MatrixXd dKuf = Eigen::MatrixXd::Zero(n_sparse, n_labels);
        dKuf.block(count, 0, size, n_labels) = Kuf.block(count, 0, size, n_labels);
        dK_alpha = (2. / hyps_curr(j)) * dKuf.transpose() * alpha;
      } else {
        dK_alpha = Kuf_grads[hyp_index + j].transpose() * alpha;
      }

      datafit_grad(hyp_index + j) += dK_alpha.transpose() * noise_vector.cwiseProduct(y_K_alpha);
      datafit_grad(hyp_index + j) += - 1./2. * alpha.transpose() * Kuu_grads[hyp_index + j] * alpha;

      likelihood_gradient(hyp_index + j) += complexity_grad(hyp_index + j) + datafit_grad(hyp_index + j); 
    }

    count += size;
    hyp_index += n_hyps;
  }

  // Derivative of complexity over noise
  double en3 = energy_noise * energy_noise * energy_noise;
  double fn3 = force_noise * force_noise * force_noise;
  double sn3 = stress_noise * stress_noise * stress_noise;
  
  compute_KnK(precomputed_KnK);
  complexity_grad(hyp_index + 0) = - n_energy_labels / energy_noise 
      + (KnK_e * Sigma).trace() / en3;
  complexity_grad(hyp_index + 1) = - n_force_labels / force_noise 
      + (KnK_f * Sigma).trace() / fn3;
  complexity_grad(hyp_index + 2) = - n_stress_labels / stress_noise 
      + (KnK_s * Sigma).trace() / sn3;

  // Derivative of data_fit over noise  
  datafit_grad(hyp_index + 0) = y_K_alpha.transpose() * e_noise_one.cwiseProduct(y_K_alpha);
  datafit_grad(hyp_index + 0) /= en3;
  datafit_grad(hyp_index + 1) = y_K_alpha.transpose() * f_noise_one.cwiseProduct(y_K_alpha);
  datafit_grad(hyp_index + 1) /= fn3;
  datafit_grad(hyp_index + 2) = y_K_alpha.transpose() * s_noise_one.cwiseProduct(y_K_alpha);
  datafit_grad(hyp_index + 2) /= sn3;

  likelihood_gradient(hyp_index + 0) += complexity_grad(hyp_index + 0) + datafit_grad(hyp_index + 0);
  likelihood_gradient(hyp_index + 1) += complexity_grad(hyp_index + 1) + datafit_grad(hyp_index + 1);
  likelihood_gradient(hyp_index + 2) += complexity_grad(hyp_index + 2) + datafit_grad(hyp_index + 2);

  return log_marginal_likelihood;

}

void SparseGP ::precompute_KnK() {
  // For NormalizedDotProduct kernel, since the signal variance is just a prefactor, we can
  // save some intermediate matrices without the prefactor. Here we save
  // Kuf * energy_noise_vector_one * Kfu / sig^4
  Kuf_e_noise_Kfu = {};
  Kuf_f_noise_Kfu = {};
  Kuf_s_noise_Kfu = {};
  for (int i = 0; i < n_kernels; i++) {
    Eigen::VectorXd hyps_i = kernels[i]->kernel_hyperparameters;
    assert(hyps_i.size() == 1);

    for (int j = 0; j < n_kernels; j++) {
      Eigen::VectorXd hyps_j = kernels[j]->kernel_hyperparameters;
      assert(hyps_j.size() == 1);
 
      double sig4 = hyps_i(0) * hyps_i(0) * hyps_j(0) * hyps_j(0);
  
      Kuf_e_noise_Kfu.push_back(Kuf_kernels[i] * e_noise_one.asDiagonal() * Kuf_kernels[j].transpose() / sig4);
      Kuf_f_noise_Kfu.push_back(Kuf_kernels[i] * f_noise_one.asDiagonal() * Kuf_kernels[j].transpose() / sig4);
      Kuf_s_noise_Kfu.push_back(Kuf_kernels[i] * s_noise_one.asDiagonal() * Kuf_kernels[j].transpose() / sig4);
    }
  }
}

void SparseGP ::compute_KnK(bool precomputed) {
  // Compute Kuf * noise_vector * Kfu sperately for energy, force and stress noises
  if (precomputed) {
    KnK_e = Eigen::MatrixXd::Zero(n_sparse, n_sparse); 
    KnK_f = Eigen::MatrixXd::Zero(n_sparse, n_sparse); 
    KnK_s = Eigen::MatrixXd::Zero(n_sparse, n_sparse); 

    int count_i = 0, count_ij = 0;
    for (int i = 0; i < n_kernels; i++) {
      Eigen::VectorXd hyps_i = kernels[i]->kernel_hyperparameters;
      assert(hyps_i.size() == 1);
      int size_i = Kuu_kernels[i].rows();
      int count_j = 0; 
      for (int j = 0; j < n_kernels; j++) {
        Eigen::VectorXd hyps_j = kernels[j]->kernel_hyperparameters;
        assert(hyps_j.size() == 1);
        int size_j = Kuu_kernels[j].rows();
   
        double sig4 = hyps_i(0) * hyps_i(0) * hyps_j(0) * hyps_j(0);
    
        KnK_e.block(count_i, count_j, size_i, size_j) += Kuf_e_noise_Kfu[count_ij] * sig4;
        KnK_f.block(count_i, count_j, size_i, size_j) += Kuf_f_noise_Kfu[count_ij] * sig4;
        KnK_s.block(count_i, count_j, size_i, size_j) += Kuf_s_noise_Kfu[count_ij] * sig4;

        count_ij += 1;
        count_j += size_j;
      }
      count_i += size_i;
    }
  } else {
    KnK_e = Kuf * e_noise_one.asDiagonal() * Kuf.transpose();
    KnK_f = Kuf * f_noise_one.asDiagonal() * Kuf.transpose();
    KnK_s = Kuf * s_noise_one.asDiagonal() * Kuf.transpose();
  }
}

Eigen::MatrixXd SparseGP ::compute_dKnK(int i) {
  // Compute Kuf_gra * noise_vector * Kfu sperately for energy, force and stress noises
  Eigen::MatrixXd dKnK = Eigen::MatrixXd::Zero(n_sparse, n_sparse);

  int count_ij = i * n_kernels;
  Eigen::VectorXd hyps_i = kernels[i]->kernel_hyperparameters;
  assert(hyps_i.size() == 1);

  int count_i = 0;
  for (int r = 0; r < i; r++) {
    count_i += Kuu_kernels[r].rows();
  }
  int size_i = Kuu_kernels[i].rows();

  int count_j = 0; 
  for (int j = 0; j < n_kernels; j++) {
    Eigen::VectorXd hyps_j = kernels[j]->kernel_hyperparameters;
    assert(hyps_j.size() == 1);
    int size_j = Kuu_kernels[j].rows();
  
    double sig3 = 2 * hyps_i(0) * hyps_j(0) * hyps_j(0);
    double sig3e = sig3 / (energy_noise * energy_noise);
    double sig3f = sig3 / (force_noise * force_noise);
    double sig3s = sig3 / (stress_noise * stress_noise);
  
    dKnK.block(count_i, count_j, size_i, size_j) += Kuf_e_noise_Kfu[count_ij] * sig3e;
    dKnK.block(count_i, count_j, size_i, size_j) += Kuf_f_noise_Kfu[count_ij] * sig3f;
    dKnK.block(count_i, count_j, size_i, size_j) += Kuf_s_noise_Kfu[count_ij] * sig3s;

    count_ij += 1;
    count_j += size_j;
  }
  return dKnK;
}

void SparseGP ::compute_likelihood() {
  if (n_labels == 0) {
    std::cout << "Warning: The likelihood is being computed without any "
                 "labels in the training set. The result won't be meaningful."
              << std::endl;
    return;
  }

  // Construct noise vector.
  Eigen::VectorXd noise = 1 / noise_vector.array();

  Eigen::MatrixXd Qff_plus_lambda =
      Kuf.transpose() * Kuu_inverse * Kuf +
      noise.asDiagonal() * Eigen::MatrixXd::Identity(n_labels, n_labels);

  // Decompose the matrix. Use QR decomposition instead of LLT/LDLT becaues Qff
  // becomes nonpositive when the training set is large.
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Qff_plus_lambda);
  Eigen::VectorXd Q_inv_y = qr.solve(y);
  Eigen::MatrixXd qr_mat = qr.matrixQR();
  // Compute the complexity penalty.
  complexity_penalty = 0;
  for (int i = 0; i < qr_mat.rows(); i++) {
    complexity_penalty += -log(abs(qr_mat(i, i)));
  }
  complexity_penalty /= 2;

  double half = 1.0 / 2.0;
  data_fit = -half * y.transpose() * Q_inv_y;
  constant_term = -half * n_labels * log(2 * M_PI);
  log_marginal_likelihood = complexity_penalty + data_fit + constant_term;
}

double
SparseGP ::compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters) {

  // Compute Kuu and Kuf matrices and gradients.
  int n_hyps_total = hyperparameters.size();

  Eigen::MatrixXd Kuu_mat = Eigen::MatrixXd::Zero(n_sparse, n_sparse);
  Eigen::MatrixXd Kuf_mat = Eigen::MatrixXd::Zero(n_sparse, n_labels);

  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad, Kuu_grads, Kuf_grads;

  int n_hyps, hyp_index = 0, grad_index = 0;
  Eigen::VectorXd hyps_curr;

  int count = 0;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    hyps_curr = hyperparameters.segment(hyp_index, n_hyps);
    int size = Kuu_kernels[i].rows();

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu_kernels[i], hyps_curr);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                    i, Kuf_kernels[i], hyps_curr);

    Kuu_mat.block(count, count, size, size) = Kuu_grad[0];
    Kuf_mat.block(count, 0, size, n_labels) = Kuf_grad[0];

    for (int j = 0; j < n_hyps; j++) {
      Kuu_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_sparse));
      Kuf_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_labels));

      Kuu_grads[hyp_index + j].block(count, count, size, size) =
          Kuu_grad[j + 1];
      Kuf_grads[hyp_index + j].block(count, 0, size, n_labels) =
          Kuf_grad[j + 1];
    }

    count += size;
    hyp_index += n_hyps;
  }

  Eigen::MatrixXd Kuu_inverse =
      (Kuu_mat + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()))
          .inverse();

  // Construct updated noise vector and gradients.
  double sigma_e = hyperparameters(hyp_index);
  double sigma_f = hyperparameters(hyp_index + 1);
  double sigma_s = hyperparameters(hyp_index + 2);

  Eigen::VectorXd noise_vec = sigma_e * sigma_e * inv_e_noise_one 
                            + sigma_f * sigma_f * inv_f_noise_one 
                            + sigma_s * sigma_s * inv_s_noise_one; 
  Eigen::VectorXd e_noise_grad = 2 * sigma_e * inv_e_noise_one;
  Eigen::VectorXd f_noise_grad = 2 * sigma_f * inv_f_noise_one;
  Eigen::VectorXd s_noise_grad = 2 * sigma_s * inv_s_noise_one;

  // Compute Qff and Qff grads.
  Eigen::MatrixXd Qff_plus_lambda =
      Kuf_mat.transpose() * Kuu_inverse * Kuf_mat +
      noise_vec.asDiagonal() * Eigen::MatrixXd::Identity(n_labels, n_labels);

  std::vector<Eigen::MatrixXd> Qff_grads;
  grad_index = 0;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    for (int j = 0; j < n_hyps; j++) {
      Qff_grads.push_back(
          Kuf_grads[grad_index].transpose() * Kuu_inverse * Kuf_mat -
          Kuf_mat.transpose() * Kuu_inverse * Kuu_grads[grad_index] *
              Kuu_inverse * Kuf_mat +
          Kuf_mat.transpose() * Kuu_inverse * Kuf_grads[grad_index]);

      grad_index++;
    }
  }

  Qff_grads.push_back(e_noise_grad.asDiagonal() *
                      Eigen::MatrixXd::Identity(n_labels, n_labels));
  Qff_grads.push_back(f_noise_grad.asDiagonal() *
                      Eigen::MatrixXd::Identity(n_labels, n_labels));
  Qff_grads.push_back(s_noise_grad.asDiagonal() *
                      Eigen::MatrixXd::Identity(n_labels, n_labels));

  // Perform LU decomposition inplace and compute the inverse.
  Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXd>> lu(Qff_plus_lambda);
  Eigen::MatrixXd Qff_inverse = lu.inverse();

  // Compute log determinant from the diagonal of U.
  complexity_penalty = 0;
#pragma omp parallel for reduction(+:complexity_penalty)
  for (int i = 0; i < Qff_plus_lambda.rows(); i++) {
    complexity_penalty += -log(abs(Qff_plus_lambda(i, i)));
  }
  complexity_penalty /= 2;

  // Compute log marginal likelihood.
  Eigen::VectorXd Q_inv_y = Qff_inverse * y;
  data_fit = -(1. / 2.) * y.transpose() * Q_inv_y;
  constant_term = -n_labels * log(2 * M_PI) / 2;
  log_marginal_likelihood = complexity_penalty + data_fit + constant_term;

  // Compute likelihood gradient.
  likelihood_gradient = Eigen::VectorXd::Zero(n_hyps_total);
  Eigen::MatrixXd Qff_inv_grad;
  for (int i = 0; i < n_hyps_total; i++) {
    Qff_inv_grad = Qff_inverse * Qff_grads[i];
    double complexity_grad = - Qff_inv_grad.trace();
    double datafit_grad = y.transpose() * Qff_inv_grad * Q_inv_y;
    likelihood_gradient(i) = (complexity_grad + datafit_grad) / 2.;
  }

  return log_marginal_likelihood;
}

void SparseGP ::set_hyperparameters(Eigen::VectorXd hyps) {
  // Reset Kuu and Kuf matrices.
  int n_hyps, hyp_index = 0;
  Eigen::VectorXd new_hyps;

  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    new_hyps = hyps.segment(hyp_index, n_hyps);

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu_kernels[i], new_hyps);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                    i, Kuf_kernels[i], new_hyps);

    Kuu_kernels[i] = Kuu_grad[0];
    Kuf_kernels[i] = Kuf_grad[0];

    kernels[i]->set_hyperparameters(new_hyps);
    hyp_index += n_hyps;
  }

  stack_Kuu();
  stack_Kuf();

  hyperparameters = hyps;
  energy_noise = hyps(hyp_index);
  force_noise = hyps(hyp_index + 1);
  stress_noise = hyps(hyp_index + 2);

  noise_vector = 1 / (energy_noise * energy_noise) * e_noise_one 
               + 1 / (force_noise * force_noise) * f_noise_one 
               + 1 / (stress_noise * stress_noise) * s_noise_one; 

  // Update remaining matrices.
  update_matrices_QR();
}

void SparseGP::write_mapping_coefficients(std::string file_name,
                                          std::string contributor,
                                          int kernel_index) {

  // Compute mapping coefficients.
  Eigen::MatrixXd mapping_coeffs =
      kernels[kernel_index]->compute_mapping_coefficients(*this, kernel_index);

  // Make beta file.
  std::ofstream coeff_file;
  coeff_file.open(file_name);

  // Record the date.
  time_t now = std::time(0);
  std::string t(ctime(&now));
  coeff_file << "DATE: ";
  coeff_file << t.substr(0, t.length() - 1) << " ";

  // Record the contributor.
  coeff_file << "CONTRIBUTOR: ";
  coeff_file << contributor << "\n";

  // Write the kernel power
  kernels[kernel_index]->write_info(coeff_file);

  // Write descriptor information to file.
  int coeff_size = mapping_coeffs.row(0).size();
  training_structures[0].descriptor_calculators[kernel_index]->write_to_file(
      coeff_file, coeff_size);

  // Write beta vectors to file.
  coeff_file << std::scientific << std::setprecision(16);

  int count = 0;
  for (int i = 0; i < mapping_coeffs.rows(); i++) {
    Eigen::VectorXd coeff_vals = mapping_coeffs.row(i);

    // Start a new line for each beta.
    if (count != 0) {
      coeff_file << "\n";
    }

    for (int j = 0; j < coeff_vals.size(); j++) {
      double coeff_val = coeff_vals[j];

      // Pad with 2 spaces if positive, 1 if negative.
      if (coeff_val > 0) {
        coeff_file << "  ";
      } else {
        coeff_file << " ";
      }

      coeff_file << coeff_vals[j];
      count++;

      // New line if 5 numbers have been added.
      if (count == 5) {
        count = 0;
        coeff_file << "\n";
      }
    }
  }

  coeff_file.close();
}

void SparseGP::write_varmap_coefficients(
  std::string file_name, std::string contributor, int kernel_index) {

  // TODO: merge this function with write_mapping_coeff, 
  // add an option in the function above for mapping "mean" or "var"

  // Compute mapping coefficients.
  //Eigen::MatrixXd varmap_coeffs =
  varmap_coeffs =
    kernels[kernel_index]->compute_varmap_coefficients(*this, kernel_index);

  // Make beta file.
  std::ofstream coeff_file;
  coeff_file.open(file_name);

  // Record the date.
  time_t now = std::time(0);
  std::string t(ctime(&now));
  coeff_file << "DATE: ";
  coeff_file << t.substr(0, t.length() - 1) << " ";

  // Record the contributor.
  coeff_file << "CONTRIBUTOR: ";
  coeff_file << contributor << "\n";

  // Record the hyps
  coeff_file << hyperparameters.size() << "\n";
  coeff_file << std::scientific << std::setprecision(16);
  for (int i = 0; i < hyperparameters.size(); i++) {      
    coeff_file << hyperparameters(i) << " ";
  }
  coeff_file << "\n" << kernels[kernel_index]->kernel_name << "\n";

  // Write descriptor information to file.
  int coeff_size = varmap_coeffs.row(0).size();
  training_structures[0].descriptor_calculators[kernel_index]->
    write_to_file(coeff_file, coeff_size);

  // Write beta vectors to file.
  coeff_file << std::scientific << std::setprecision(16);

  int count = 0;
  for (int i = 0; i < varmap_coeffs.rows(); i++) {
    Eigen::VectorXd coeff_vals = varmap_coeffs.row(i);

    // Start a new line for each beta.
    if (count != 0) {
      coeff_file << "\n";
      count = 0;
    }

    for (int j = 0; j < coeff_vals.size(); j++) {
      double coeff_val = coeff_vals[j];

      // Pad with 2 spaces if positive, 1 if negative.
      if (coeff_val > 0) {
        coeff_file << "  ";
      } else {
        coeff_file << " ";
      }

      coeff_file << coeff_vals[j];
      count++;

      // New line if 5 numbers have been added.
      if (count == 5) {
        count = 0;
        coeff_file << "\n";
      }
    }
  }

  coeff_file.close();
}

void SparseGP::write_L_inverse(
  std::string file_name, std::string contributor) {
  // Make beta file.
  std::ofstream coeff_file;
  coeff_file.open(file_name);

  // Record file name
  coeff_file << "L_inverse_block file ";

  // Record the date.
  time_t now = std::time(0);
  std::string t(ctime(&now));
  coeff_file << "DATE: ";
  coeff_file << t.substr(0, t.length() - 1) << " ";

  // Record the contributor.
  coeff_file << "CONTRIBUTOR: ";
  coeff_file << contributor << "\n";

  // Write the kernel power
  // TODO: support multiple kernels
  kernels[0]->write_info(coeff_file);

  // Record the hyps
  coeff_file << hyperparameters.size() << "\n";
  coeff_file << std::scientific << std::setprecision(16);
  for (int i = 0; i < hyperparameters.size(); i++) {      
    coeff_file << hyperparameters(i) << " ";
  }
  coeff_file << "\n";

  int sparse_count = 0;
  for (int i = 0; i < n_kernels; i++) {
    //  sparse_descriptors[i].descriptors[s];
    training_structures[0].descriptor_calculators[i]->
      write_to_file(coeff_file, n_kernels);

    coeff_file << std::scientific << std::setprecision(16);

    // write the lower triangular part of L_inv_block 
    int n_clusters = sparse_descriptors[i].n_clusters;
    Eigen::MatrixXd L_inverse_block =
        L_inv.block(sparse_count, sparse_count, n_clusters, n_clusters);
    sparse_count += n_clusters;

    coeff_file << n_clusters << "\n";
    int count = 1;
    for (int j = 0; j < n_clusters; j++) {
      for (int k = 0; k <= j; k++) {
        coeff_file << L_inverse_block(j, k) << " ";

        // Change line after writing 5 numbers
        if (count % 5 == 0) coeff_file << "\n";
        count++;
      }
    }
    if (count % 5 != 0) coeff_file << "\n";

  }

  coeff_file.close();
}

void SparseGP::write_sparse_descriptors(
  std::string file_name, std::string contributor) {
  double empty_thresh = 1e-8;

  // Make beta file.
  std::ofstream coeff_file;
  coeff_file.open(file_name);

  // Record file name
  coeff_file << "sparse_descriptors file ";

  // Record the date.
  time_t now = std::time(0);
  std::string t(ctime(&now));
  coeff_file << "DATE: ";
  coeff_file << t.substr(0, t.length() - 1) << " ";

  // Record the contributor.
  coeff_file << "CONTRIBUTOR: ";
  coeff_file << contributor << "\n";

  coeff_file << std::scientific << std::setprecision(16);

  // Record the number of kernels
  coeff_file << n_kernels << "\n";

  for (int i = 0; i < n_kernels; i++) {

    int n_types = sparse_descriptors[i].n_types;
    int n_clusters = sparse_descriptors[i].n_clusters;

    coeff_file << i << " " << n_clusters << " " << n_types << "\n";

    int count = 1;
    for (int s = 0; s < n_types; s++) {
      int n_clusters_by_type = sparse_descriptors[i].n_clusters_by_type[s];
      int n_descriptors = sparse_descriptors[i].n_descriptors;

      coeff_file << n_clusters_by_type << "\n";
      for (int j = 0; j < n_clusters_by_type; j++) {
        for (int k = 0; k < n_descriptors; k++) {
          if (sparse_descriptors[i].descriptor_norms[s](j) < empty_thresh) {
            coeff_file << 0.0 << " ";
          } else {
            if (kernels[i]->kernel_name.find("NormalizedDotProduct") != std::string::npos) {
              coeff_file << sparse_descriptors[i].descriptors[s](j, k) / sparse_descriptors[i].descriptor_norms[s](j) << " ";
            } else {
              coeff_file << sparse_descriptors[i].descriptors[s](j, k) << " ";
            }
          }

          // Change line after writing 5 numbers
          if (count % 5 == 0) coeff_file << "\n";
          count++;
        }
      }
      if (count % 5 != 1) coeff_file << "\n";
    }

  }

  coeff_file.close();
}

void SparseGP ::to_json(std::string file_name, const SparseGP & sgp){
  std::ofstream sgp_file(file_name);
  nlohmann::json j = sgp;
  sgp_file << j;
}

SparseGP SparseGP ::from_json(std::string file_name){
  std::ifstream sgp_file(file_name);
  nlohmann::json j;
  sgp_file >> j;
  return j;
}
