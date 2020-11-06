#include "compact_gp.h"

CompactGP ::CompactGP() {}

CompactGP ::CompactGP(std::vector<CompactKernel *> kernels,
                      double energy_noise, double force_noise,
                      double stress_noise){

  this->kernels = kernels;
  Kuu_jitter = 1e-8; // default value

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
    Kuf_energy.push_back(empty_matrix);
    Kuf_force.push_back(empty_matrix);
    Kuf_stress.push_back(empty_matrix);
    Kuu_kernels.push_back(empty_matrix);
  }
}

void CompactGP ::add_sparse_environments(const CompactStructure &structure) {

  int n_kernels = kernels.size();
  int n_labels = Kuf.cols();
  int n_strucs = training_structures.size();

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < structure.descriptors.size(); i++){
      ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i]);
      cluster_descriptors.push_back(cluster_descriptor);
  }

  // Update Kuu matrices.
  for (int i = 0; i < n_kernels; i++) {
    Eigen::MatrixXd prev_block = kernels[i]->envs_envs(
      sparse_descriptors[i], cluster_descriptors[i]);
    Eigen::MatrixXd self_block = kernels[i]->envs_envs(
      cluster_descriptors[i], cluster_descriptors[i]);

    int n_sparse = sparse_descriptors[i].n_clusters;
    int n_envs = cluster_descriptors[i].n_clusters;

    Kuu_kernels[i].conservativeResize(n_sparse + n_envs, n_sparse + n_envs);
    Kuu_kernels[i].block(0, n_sparse, n_sparse, n_envs) = prev_block;
    Kuu_kernels[i].block(n_sparse, 0, n_envs, n_sparse) =
        prev_block.transpose();
    Kuu_kernels[i].block(n_sparse, n_sparse, n_envs, n_envs) = self_block;
  }

  // Compute kernels between new sparse environments and training structures.
  Eigen::MatrixXd envs_struc_kernels;
  for (int i = 0; i < n_kernels; i++) {
    int n_sparse = sparse_descriptors[i].n_clusters;
    int n_envs = cluster_descriptors[i].n_clusters;

    Kuf_energy[i].conservativeResize(n_sparse + n_envs, n_energy_labels);
    Kuf_force[i].conservativeResize(n_sparse + n_envs, n_force_labels);
    Kuf_stress[i].conservativeResize(n_sparse + n_envs, n_stress_labels);

    int e_count = 0;
    int f_count = 0;
    int s_count = 0;

    for (int j = 0; j < n_strucs; j++){
      int n_atoms = training_structures[j].noa;
      envs_struc_kernels = 
        kernels[i]->envs_struc(sparse_descriptors[i], structure.descriptors[i]);

      if (training_structures[j].energy.size() != 0) {
          Kuf_energy[i].block(n_sparse, e_count, n_envs, 1) =
            envs_struc_kernels.block(0, 0, n_envs, 1);
          e_count += 1;
        }

      if (training_structures[j].forces.size() != 0) {
        Kuf_force[i].block(n_sparse, f_count, n_envs, n_atoms * 3) =
          envs_struc_kernels.block(0, 1, n_envs, n_atoms * 3);
          f_count += n_atoms * 3;
      }

      if (training_structures[j].stresses.size() != 0) {
        Kuf_stress[i].block(n_sparse, s_count, n_envs, 6) =
          envs_struc_kernels.block(0, 1 + n_atoms * 3, n_envs, 6);
        s_count += 6;
      }
    }
  }

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_cluster(structure.descriptors[i]);
  }
}
