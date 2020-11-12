#include "compact_gp.h"
#include <iostream>

CompactGP ::CompactGP() {}

CompactGP ::CompactGP(std::vector<CompactKernel *> kernels, double energy_noise,
                      double force_noise, double stress_noise) {

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
  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i]);
    cluster_descriptors.push_back(cluster_descriptor);

    if (sparse_descriptors.size() == 0) {
      ClusterDescriptor empty_descriptor;
      empty_descriptor.initialize_cluster(cluster_descriptor.n_types,
                                          cluster_descriptor.n_descriptors);
      sparse_descriptors.push_back(empty_descriptor);
    }
  }

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

    Kuu_kernels[i].conservativeResize(n_sparse + n_envs, n_sparse + n_envs);
    Kuu_kernels[i].block(0, n_sparse, n_sparse, n_envs) = prev_block;
    Kuu_kernels[i].block(n_sparse, 0, n_envs, n_sparse) =
        prev_block.transpose();
    Kuu_kernels[i].block(n_sparse, n_sparse, n_envs, n_envs) = self_block;

    // Update sparse count.
    this->n_sparse += n_envs;
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

    // TODO: Parellelize this loop.
    for (int j = 0; j < n_strucs; j++) {
      int n_atoms = training_structures[j].noa;
      envs_struc_kernels = kernels[i]->envs_struc(
          cluster_descriptors[i], structure.descriptors[i],
          kernels[i]->kernel_hyperparameters);

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

  // Update Kuu and Kuf.
  update_Kuu();
  update_Kuf();
}

void CompactGP ::add_training_structure(const CompactStructure &structure) {

  int n_energy = structure.energy.size();
  int n_force = structure.forces.size();
  int n_stress = structure.stresses.size();
  int n_labels = n_energy + n_force + n_stress;
  int n_atoms = structure.noa;
  int n_kernels = kernels.size();

  // Initialize sparse descriptors.
  if (sparse_descriptors.size() == 0) {
    for (int i = 0; i < structure.descriptors.size(); i++) {
      ClusterDescriptor empty_descriptor;
      empty_descriptor.initialize_cluster(
          structure.descriptors[i].n_types,
          structure.descriptors[i].n_descriptors);
      sparse_descriptors.push_back(empty_descriptor);
    }
  }

  // Update Kuf kernels.
  Eigen::MatrixXd envs_struc_kernels;
  for (int i = 0; i < n_kernels; i++) {
    int n_sparse = sparse_descriptors[i].n_clusters;

    envs_struc_kernels =
        kernels[i]->envs_struc(sparse_descriptors[i], structure.descriptors[i],
                               kernels[i]->kernel_hyperparameters);

    Kuf_energy[i].conservativeResize(n_sparse, n_energy_labels + n_energy);
    Kuf_force[i].conservativeResize(n_sparse, n_force_labels + n_force);
    Kuf_stress[i].conservativeResize(n_sparse, n_stress_labels + n_stress);

    Kuf_energy[i].block(0, n_energy_labels, n_sparse, n_energy) =
        envs_struc_kernels.block(0, 0, n_sparse, n_energy);
    Kuf_force[i].block(0, n_force_labels, n_sparse, n_force) =
        envs_struc_kernels.block(0, 1, n_sparse, n_force);
    Kuf_stress[i].block(0, n_stress_labels, n_sparse, n_stress) =
        envs_struc_kernels.block(0, 1 + n_atoms * 3, n_sparse, n_stress);
  }

  // Update label count.
  n_energy_labels += n_energy;
  n_force_labels += n_force;
  n_stress_labels += n_stress;
  this->n_labels += n_energy + n_force + n_stress;

  // Store training structure.
  training_structures.push_back(structure);

  // Update labels.
  energy_labels.conservativeResize(n_energy_labels);
  force_labels.conservativeResize(n_force_labels);
  stress_labels.conservativeResize(n_stress_labels);

  energy_labels.tail(n_energy) = structure.energy;
  force_labels.tail(n_force) = structure.forces;
  stress_labels.tail(n_stress) = structure.stresses;

  y.conservativeResize(n_energy_labels + n_force_labels + n_stress_labels);
  y.segment(0, n_energy_labels) = energy_labels;
  y.segment(n_energy_labels, n_force_labels) = force_labels;
  y.segment(n_energy_labels + n_force_labels, n_stress_labels) = stress_labels;

  // Update noise.
  noise_vector.conservativeResize(n_energy_labels + n_force_labels +
                                  n_stress_labels);
  noise_vector.segment(0, n_energy_labels) = Eigen::VectorXd::Constant(
      n_energy_labels, 1 / (energy_noise * energy_noise));
  noise_vector.segment(n_energy_labels, n_force_labels) =
      Eigen::VectorXd::Constant(n_force_labels,
                                1 / (force_noise * force_noise));
  noise_vector.segment(n_energy_labels + n_force_labels, n_stress_labels) =
      Eigen::VectorXd::Constant(n_stress_labels,
                                1 / (stress_noise * stress_noise));

  // Update Kuf.
  update_Kuf();
}

void CompactGP ::update_Kuu() {
  // Update Kuu.
  Kuu = Eigen::MatrixXd::Zero(n_sparse, n_sparse);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    Kuu.block(count, count, size, size) = Kuu_kernels[i];
    count += size;
  }
}

void CompactGP ::update_Kuf() {
  // Update Kuf kernels.
  std::vector<Eigen::MatrixXd> empty_matrices;
  Kuf_kernels = empty_matrices;
  Kuf = Eigen::MatrixXd::Zero(n_sparse, n_labels);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    Eigen::MatrixXd Kuf_mat = Eigen::MatrixXd::Zero(size, n_labels);
    Kuf_mat.block(0, 0, size, n_energy_labels) = Kuf_energy[i];
    Kuf_mat.block(0, n_energy_labels, size, n_force_labels) = Kuf_force[i];
    Kuf_mat.block(0, n_energy_labels + n_force_labels, size, n_stress_labels) =
        Kuf_stress[i];

    Kuf.block(count, 0, size, n_labels) = Kuf_mat;
    Kuf_kernels.push_back(Kuf_mat);
    count += size;
  }
}

void CompactGP ::update_matrices_QR() {
  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = Eigen::VectorXd::Zero(n_labels);
  for (int i = 0; i < noise_vector_sqrt.size(); i++) {
    noise_vector_sqrt(i) = sqrt(noise_vector(i));
  }

  // Cholesky decompose Kuu.
  Eigen::LLT<Eigen::MatrixXd> chol(
      Kuu + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()));

  // Get the inverse from Cholesky decomposition.
  // TODO: Check if this is actually faster than explicit inversion.
  Eigen::MatrixXd Kuu_eye = Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
  Kuu_inverse = chol.solve(Kuu_eye);

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
  Eigen::MatrixXd R_inv = qr.matrixQR()
                              .block(0, 0, Kuu.cols(), Kuu.cols())
                              .triangularView<Eigen::Upper>()
                              .solve(Kuu_eye);
  alpha = R_inv * Q_b;
  Sigma = R_inv * R_inv.transpose();
}

void CompactGP ::predict_on_structure(CompactStructure &test_structure) {

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;
  int n_kernels = kernels.size();

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

void CompactGP ::compute_likelihood() {
  if (n_labels == 0) {
    std::cout << "Warning: The likelihood is being computed withouot any "
                 "labels in the training set. The result won't be meaningful."
              << std::endl;
    return;
  }

  // Construct noise vector.
  Eigen::VectorXd noise = Eigen::VectorXd::Zero(n_labels);
  noise.segment(0, n_energy_labels) =
      Eigen::VectorXd::Constant(n_energy_labels, energy_noise * energy_noise);
  noise.segment(n_energy_labels, n_force_labels) =
      Eigen::VectorXd::Constant(n_force_labels, force_noise * force_noise);
  noise.segment(n_energy_labels + n_force_labels, n_stress_labels) =
      Eigen::VectorXd::Constant(n_stress_labels, stress_noise * stress_noise);

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
