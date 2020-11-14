#include "compact_gp.h"
#include <iostream>

CompactGP ::CompactGP() {}

CompactGP ::CompactGP(std::vector<CompactKernel *> kernels, double energy_noise,
                      double force_noise, double stress_noise) {

  this->kernels = kernels;
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

    Kuf_kernels[i].conservativeResize(n_sparse + n_envs, n_labels);

#pragma omp parallel for
    for (int j = 0; j < n_strucs; j++) {
      int n_atoms = training_structures[j].noa;
      envs_struc_kernels = kernels[i]->envs_struc(
          cluster_descriptors[i], structure.descriptors[i],
          kernels[i]->kernel_hyperparameters);

      int current_count = 0;
      if (training_structures[j].energy.size() != 0) {
        Kuf_kernels[i].block(n_sparse, label_count(j), n_envs, 1) =
            envs_struc_kernels.block(0, 0, n_envs, 1);
        current_count += 1;
      }

      if (training_structures[j].forces.size() != 0) {
        Kuf_kernels[i].block(n_sparse, label_count(j) + current_count, n_envs,
                             n_atoms * 3) =
            envs_struc_kernels.block(0, 1, n_envs, n_atoms * 3);
        current_count += n_atoms * 3;
      }

      if (training_structures[j].stresses.size() != 0) {
        Kuf_kernels[i].block(n_sparse, label_count(j) + current_count, n_envs,
                             6) =
            envs_struc_kernels.block(0, 1 + n_atoms * 3, n_envs, 6);
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
  int n_struc_labels = n_energy + n_force + n_stress;
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

    Kuf_kernels[i].conservativeResize(n_sparse, n_labels + n_struc_labels);
    Kuf_kernels[i].block(0, n_labels, n_sparse, n_energy) =
        envs_struc_kernels.block(0, 0, n_sparse, n_energy);
    Kuf_kernels[i].block(0, n_labels + n_energy, n_sparse, n_force) =
        envs_struc_kernels.block(0, 1, n_sparse, n_force);
    Kuf_kernels[i].block(0, n_labels + n_energy + n_force, n_sparse, n_stress) =
        envs_struc_kernels.block(0, 1 + n_atoms * 3, n_sparse, n_sparse);
  }

  // Update labels.
  label_count.conservativeResize(training_structures.size() + 2);
  label_count(training_structures.size() + 1) = n_labels + n_struc_labels;
  y.conservativeResize(n_labels + n_struc_labels);
  y.segment(n_labels, n_energy) = structure.energy;
  y.segment(n_labels + n_energy, n_force) = structure.forces;
  y.segment(n_labels + n_energy + n_force, n_stress) = structure.stresses;

  // Update noise.
  noise_vector.conservativeResize(n_labels + n_struc_labels);
  noise_vector.segment(n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, 1 / (energy_noise * energy_noise));
  noise_vector.segment(n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, 1 / (force_noise * force_noise));
  noise_vector.segment(n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, 1 / (stress_noise * stress_noise));

  // Update label count.
  n_energy_labels += n_energy;
  n_force_labels += n_force;
  n_stress_labels += n_stress;
  n_labels += n_struc_labels;

  // Store training structure.
  training_structures.push_back(structure);

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
  Kuf = Eigen::MatrixXd::Zero(n_sparse, n_labels);
  int count = 0;
  for (int i = 0; i < Kuf_kernels.size(); i++) {
    int size = Kuf_kernels[i].rows();
    Kuf.block(count, 0, size, n_labels) = Kuf_kernels[i];
    count += size;
  }
}

void CompactGP ::update_matrices_QR() {
  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = sqrt(noise_vector.array());

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

double CompactGP ::compute_likelihood_gradient(
    const Eigen::VectorXd &hyperparameters) {

  // Compute Kuu and Kuf matrices and gradients.
  int n_kernels = kernels.size();
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

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu, hyps_curr);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                    i, Kuf, hyps_curr);

    Kuu_mat.block(count, count, size, size) = Kuu_grad[0];
    Kuf_mat.block(count, 0, size, n_labels) = Kuf_grad[0];

    for (int j = 0; j < n_hyps; j++) {
      Kuu_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_sparse));
      Kuf_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_labels));

      Kuu_grads[j].block(count, count, size, size) = Kuu_grad[j + 1];
      Kuf_grads[j].block(count, 0, size, n_labels) = Kuf_grad[j + 1];
    }

    count += size;
    hyp_index += n_hyps;
  }

  Eigen::MatrixXd Kuu_inverse =
      (Kuu_mat + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()))
          .inverse();

  // Construct updated noise vector and gradients.
  Eigen::VectorXd noise_vec = Eigen::VectorXd::Zero(n_labels);
  Eigen::VectorXd e_noise_grad = Eigen::VectorXd::Zero(n_labels);
  Eigen::VectorXd f_noise_grad = Eigen::VectorXd::Zero(n_labels);
  Eigen::VectorXd s_noise_grad = Eigen::VectorXd::Zero(n_labels);

  double sigma_e = hyperparameters(hyp_index);
  double sigma_f = hyperparameters(hyp_index + 1);
  double sigma_s = hyperparameters(hyp_index + 2);

  int current_count = 0;
  for (int i = 0; i < training_structures.size(); i++){
    int n_atoms = training_structures[i].noa;

    if (training_structures[i].energy.size() != 0) {
      noise_vec(current_count) = sigma_e * sigma_e;
      e_noise_grad(current_count) = 2 * sigma_e;
      current_count += 1;
    }

    if (training_structures[i].forces.size() != 0) {
      noise_vec.segment(current_count, n_atoms * 3) =
        Eigen::VectorXd::Constant(n_atoms * 3, sigma_f * sigma_f);
      f_noise_grad.segment(current_count, n_atoms * 3) =
        Eigen::VectorXd::Constant(n_atoms * 3, 2 * sigma_f);
      current_count += n_atoms * 3;
    }

    if (training_structures[i].stresses.size() != 0){
      noise_vec.segment(current_count, 6) =
        Eigen::VectorXd::Constant(6, sigma_s * sigma_s);
      s_noise_grad.segment(current_count, 6) = 
        Eigen::VectorXd::Constant(6, 2 * sigma_s);
      current_count += 6;
    }
  }

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
  double complexity_penalty = 0;
  for (int i = 0; i < Qff_plus_lambda.rows(); i++) {
    complexity_penalty += -log(abs(Qff_plus_lambda(i, i)));
  }
  complexity_penalty /= 2;

  // Compute log marginal likelihood.
  Eigen::VectorXd Q_inv_y = Qff_inverse * y;
  double data_fit = -(1. / 2.) * y.transpose() * Q_inv_y;
  double constant_term = -n_labels * log(2 * M_PI) / 2;
  double log_marginal_likelihood =
      complexity_penalty + data_fit + constant_term;

  // Compute likelihood gradient.
  likelihood_gradient = Eigen::VectorXd::Zero(n_hyps_total);
  Eigen::MatrixXd Qff_inv_grad;
  for (int i = 0; i < n_hyps_total; i++) {
    Qff_inv_grad = Qff_inverse * Qff_grads[i];
    likelihood_gradient(i) =
        -Qff_inv_grad.trace() + y.transpose() * Qff_inv_grad * Q_inv_y;
    likelihood_gradient(i) /= 2;
  }

  return log_marginal_likelihood;
}

void CompactGP ::set_hyperparameters(Eigen::VectorXd hyps){
  // Reset Kuu and Kuf matrices.
  int n_kernels = kernels.size();
  int n_hyps, hyp_index = 0;
  Eigen::VectorXd new_hyps;

  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    new_hyps = hyps.segment(hyp_index, n_hyps);

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu, new_hyps);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                    i, Kuf, new_hyps);

    Kuu_kernels[i] = Kuu_grad[0];
    Kuf_kernels[i] = Kuf_grad[0];

    kernels[i]->set_hyperparameters(new_hyps);
    hyp_index += n_hyps;
  }

  update_Kuu();
  update_Kuf();

  hyperparameters = hyps;
  energy_noise = hyps(hyp_index);
  force_noise = hyps(hyp_index + 1);
  stress_noise = hyps(hyp_index + 2);

  int current_count = 0;
  for (int i = 0; i < training_structures.size(); i++){
    int n_atoms = training_structures[i].noa;

    if (training_structures[i].energy.size() != 0) {
      noise_vector(current_count) = 1 / (energy_noise * energy_noise);
      current_count += 1;
    }

    if (training_structures[i].forces.size() != 0) {
      noise_vector.segment(current_count, n_atoms * 3) =
        Eigen::VectorXd::Constant(n_atoms * 3, 1 / (force_noise * force_noise));
      current_count += n_atoms * 3;
    }

    if (training_structures[i].stresses.size() != 0){
      noise_vector.segment(current_count, 6) =
        Eigen::VectorXd::Constant(6, 1 / (stress_noise * stress_noise));
      current_count += 6;
    }
  }

  // Update remaining matrices.
  update_matrices_QR();
}
