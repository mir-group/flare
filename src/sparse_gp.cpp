#include "sparse_gp.h"
#include <cmath>
#include <iostream>

SparseGP :: SparseGP(){}

SparseGP :: SparseGP(std::vector<Kernel *> kernels, double sigma_e,
    double sigma_f, double sigma_s){

    this->kernels = kernels;

    // Count hyperparameters.
    int n_hyps = 0;
    for (int i = 0; i < kernels.size(); i ++){
        n_hyps += kernels[i]->kernel_hyperparameters.size();
    }

    // Set the kernel hyperparameters.
    hyperparameters = Eigen::VectorXd::Zero(n_hyps + 3);
    std::vector<double> hyps_curr;
    int hyp_counter = 0;
    for (int i = 0; i < kernels.size(); i ++){
        hyps_curr = kernels[i]->kernel_hyperparameters;

        for (int j = 0; j < hyps_curr.size(); j ++){
            hyperparameters(hyp_counter) = hyps_curr[j];
            hyp_counter ++;
        }
    }

    // Set the noise hyperparameters.
    hyperparameters(n_hyps) = sigma_e;
    hyperparameters(n_hyps+1) = sigma_f;
    hyperparameters(n_hyps+2) = sigma_s;

    this->sigma_e = sigma_e;
    this->sigma_f = sigma_f;
    this->sigma_s = sigma_s;
}

void SparseGP :: add_sparse_environment(LocalEnvironment env){
    // Compute kernels between new environment and previous sparse
    // environments.
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();

    Eigen::VectorXd uu_vector = Eigen::VectorXd::Zero(n_sparse + 1);
    #pragma omp parallel for
    for (int i = 0; i < n_sparse; i ++){
        for (int j = 0; j < n_kernels; j ++){
            uu_vector(i) += kernels[j] -> env_env(sparse_environments[i], env);
        }
    }

    // Compute self kernel.
    double self_kernel = 0;
    for (int j = 0; j < n_kernels; j ++){
        self_kernel += kernels[j] -> env_env(env, env);
    }

    // Update Kuu matrix.
    Kuu.conservativeResize(Kuu.rows()+1, Kuu.cols()+1);
    Kuu.col(Kuu.cols()-1) = uu_vector;
    Kuu.row(Kuu.rows()-1) = uu_vector;
    Kuu(n_sparse, n_sparse) = self_kernel;

    // Compute kernels between new environment and training structures.
    int n_labels = Kuf.cols();
    int n_strucs = training_structures.size();
    Eigen::VectorXd uf_vector = Eigen::VectorXd::Zero(n_labels);
    Eigen::VectorXd kernel_vector;
    int label_count = 0;
    int n_atoms;
    #pragma omp parallel for
    for (int i = 0; i < n_strucs; i ++){
        n_atoms = training_structures[i].noa;
        kernel_vector = Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector += kernels[j] -> env_struc(env,
                training_structures[i]);
        }

        if (training_structures[i].energy.size() != 0){
            uf_vector(label_count) = kernel_vector(0);
            label_count += 1;
        }

        if (training_structures[i].forces.size() != 0){
            uf_vector.segment(label_count, n_atoms * 3) =
                kernel_vector.segment(1, n_atoms * 3);
            label_count += n_atoms * 3;
        }

        if (training_structures[i].stresses.size() != 0){
            uf_vector.segment(label_count, 6) =
                kernel_vector.tail(6);
            label_count += 6;
        }
    }

    // Update Kuf matrix.
    Kuf.conservativeResize(Kuf.rows()+1, Kuf.cols());
    Kuf.row(Kuf.rows()-1) = uf_vector;

    // Store sparse environment.
    sparse_environments.push_back(env);
}

void SparseGP :: add_sparse_environment_serial(LocalEnvironment env){
    // Compute kernels between new environment and previous sparse
    // environments.
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();

    Eigen::VectorXd uu_vector = Eigen::VectorXd::Zero(n_sparse + 1);
    for (int i = 0; i < n_sparse; i ++){
        for (int j = 0; j < n_kernels; j ++){
            uu_vector(i) += kernels[j] -> env_env(sparse_environments[i], env);
        }
    }

    // Compute self kernel.
    double self_kernel = 0;
    for (int j = 0; j < n_kernels; j ++){
        self_kernel += kernels[j] -> env_env(env, env);
    }

    // Update Kuu matrix.
    Kuu.conservativeResize(Kuu.rows()+1, Kuu.cols()+1);
    Kuu.col(Kuu.cols()-1) = uu_vector;
    Kuu.row(Kuu.rows()-1) = uu_vector;
    Kuu(n_sparse, n_sparse) = self_kernel;

    // Compute kernels between new environment and training structures.
    int n_labels = Kuf.cols();
    int n_strucs = training_structures.size();
    Eigen::VectorXd uf_vector = Eigen::VectorXd::Zero(n_labels);
    Eigen::VectorXd kernel_vector;
    int label_count = 0;
    int n_atoms;
    for (int i = 0; i < n_strucs; i ++){
        n_atoms = training_structures[i].noa;
        kernel_vector = Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector += kernels[j] -> env_struc(env,
                training_structures[i]);
        }

        if (training_structures[i].energy.size() != 0){
            uf_vector(label_count) = kernel_vector(0);
            label_count += 1;
        }

        if (training_structures[i].forces.size() != 0){
            uf_vector.segment(label_count, n_atoms * 3) =
                kernel_vector.segment(1, n_atoms * 3);
            label_count += n_atoms * 3;
        }

        if (training_structures[i].stresses.size() != 0){
            uf_vector.segment(label_count, 6) =
                kernel_vector.tail(6);
            label_count += 6;
        }
    }

    // Update Kuf matrix.
    Kuf.conservativeResize(Kuf.rows()+1, Kuf.cols());
    Kuf.row(Kuf.rows()-1) = uf_vector;

    // Store sparse environment.
    sparse_environments.push_back(env);
}

void SparseGP :: add_training_structure_serial(StructureDescriptor
    training_structure){

    int n_labels = training_structure.energy.size() +
        training_structure.forces.size() + training_structure.stresses.size();
    int n_atoms = training_structure.noa;
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();

    // Calculate kernels between sparse environments and training structure.
    Eigen::MatrixXd kernel_block = Eigen::MatrixXd::Zero(n_sparse, n_labels);
    LocalEnvironment sparse_env;
    Eigen::VectorXd kernel_vector;
    int label_count;

    for (int i = 0; i < n_sparse; i ++){
        kernel_vector = Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector += kernels[j] -> env_struc(sparse_environments[i],
                training_structure);
        }

        // Update kernel block.
        label_count = 0;
        if (training_structure.energy.size() != 0){
            kernel_block(i, 0) = kernel_vector(0);
            label_count += 1;
        }

        if (training_structure.forces.size() != 0){
            kernel_block.row(i).segment(label_count, n_atoms * 3) =
                kernel_vector.segment(1, n_atoms * 3);
        }

        if (training_structure.stresses.size() != 0){
            kernel_block.row(i).tail(6) = kernel_vector.tail(6);
        }
    }

    // Add kernel block to Kuf.
    int prev_cols = Kuf.cols();
    Kuf.conservativeResize(n_sparse, prev_cols + n_labels);
    Kuf.block(0, prev_cols, n_sparse, n_labels) = kernel_block;

    // Store training structure.
    training_structures.push_back(training_structure);

    // Update y vector and noise matrix.
    Eigen::VectorXd labels = Eigen::VectorXd::Zero(n_labels);
    Eigen::VectorXd noise_vector = Eigen::VectorXd::Zero(n_labels);

    label_count = 0;
    if (training_structure.energy.size() != 0){
        labels.head(1) = training_structure.energy;
        noise_vector(0) = 1 / (sigma_e * sigma_e);
        label_count ++;
    }

    if (training_structure.forces.size() != 0){
        labels.segment(label_count, n_atoms * 3) = training_structure.forces;
        noise_vector.segment(label_count, n_atoms * 3) =
            Eigen::VectorXd::Constant(n_atoms * 3, 1 / (sigma_f * sigma_f));
    }

    if (training_structure.stresses.size() != 0){
        labels.tail(6) = training_structure.stresses;
        noise_vector.tail(6) =
            Eigen::VectorXd::Constant(6, 1 / (sigma_s * sigma_s));
    }

    y.conservativeResize(y.size() + n_labels);
    y.tail(n_labels) = labels;

    noise.conservativeResize(prev_cols + n_labels);
    noise.tail(n_labels) = noise_vector;
    noise_matrix = noise.asDiagonal();
}

void SparseGP :: add_training_structure(StructureDescriptor training_structure){

    int n_labels = training_structure.energy.size() +
        training_structure.forces.size() + training_structure.stresses.size();
    int n_atoms = training_structure.noa;
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();

    // Calculate kernels between sparse environments and training structure.
    Eigen::MatrixXd kernel_block = Eigen::MatrixXd::Zero(n_sparse, n_labels);
    LocalEnvironment sparse_env;
    Eigen::VectorXd kernel_vector;
    int label_count;

    #pragma omp parallel for
    for (int i = 0; i < n_sparse; i ++){
        kernel_vector = Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector += kernels[j] -> env_struc(sparse_environments[i],
                training_structure);
        }

        // Update kernel block.
        label_count = 0;
        if (training_structure.energy.size() != 0){
            kernel_block(i, 0) = kernel_vector(0);
            label_count += 1;
        }

        if (training_structure.forces.size() != 0){
            kernel_block.row(i).segment(label_count, n_atoms * 3) =
                kernel_vector.segment(1, n_atoms * 3);
        }

        if (training_structure.stresses.size() != 0){
            kernel_block.row(i).tail(6) = kernel_vector.tail(6);
        }
    }

    // Add kernel block to Kuf.
    int prev_cols = Kuf.cols();
    Kuf.conservativeResize(n_sparse, prev_cols + n_labels);
    Kuf.block(0, prev_cols, n_sparse, n_labels) = kernel_block;

    // Store training structure.
    training_structures.push_back(training_structure);

    // Update y vector and noise matrix.
    Eigen::VectorXd labels = Eigen::VectorXd::Zero(n_labels);
    Eigen::VectorXd noise_vector = Eigen::VectorXd::Zero(n_labels);

    label_count = 0;
    if (training_structure.energy.size() != 0){
        labels.head(1) = training_structure.energy;
        noise_vector(0) = 1 / (sigma_e * sigma_e);
        label_count ++;
    }

    if (training_structure.forces.size() != 0){
        labels.segment(label_count, n_atoms * 3) = training_structure.forces;
        noise_vector.segment(label_count, n_atoms * 3) =
            Eigen::VectorXd::Constant(n_atoms * 3, 1 / (sigma_f * sigma_f));
    }

    if (training_structure.stresses.size() != 0){
        labels.tail(6) = training_structure.stresses;
        noise_vector.tail(6) =
            Eigen::VectorXd::Constant(6, 1 / (sigma_s * sigma_s));
    }

    y.conservativeResize(y.size() + n_labels);
    y.tail(n_labels) = labels;

    noise.conservativeResize(prev_cols + n_labels);
    noise.tail(n_labels) = noise_vector;
    noise_matrix = noise.asDiagonal();
}

void SparseGP::update_alpha(){
    Eigen::MatrixXd sigma_inv = Kuu + Kuf * noise_matrix * Kuf.transpose();
    Sigma = sigma_inv.inverse();
    alpha = Sigma * Kuf * noise_matrix * y;
}

Eigen::VectorXd SparseGP::predict(StructureDescriptor test_structure){
    int n_atoms = test_structure.noa;
    int n_out = 1 + 3 * n_atoms + 6;
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();
    Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(n_out, n_sparse);

    LocalEnvironment sparse_env;
    Eigen::VectorXd kernel_vector;

    // Compute the kernel between the test structure and each sparse
    // environment, parallelizing over environments.
    #pragma omp parallel for
    for (int i = 0; i < n_sparse; i ++){
        for (int j = 0; j < n_kernels; j ++){
            kern_mat.col(i) +=
                kernels[j] -> env_struc(sparse_environments[i], test_structure);
        }
    }

    return kern_mat * alpha;
}

Eigen::VectorXd SparseGP::predict_serial(StructureDescriptor test_structure){
    int n_atoms = test_structure.noa;
    int n_out = 1 + 3 * n_atoms + 6;
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();
    Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(n_out, n_sparse);

    LocalEnvironment sparse_env;
    Eigen::VectorXd kernel_vector;

    for (int i = 0; i < n_sparse; i ++){
        sparse_env = sparse_environments[i];
        kernel_vector = Eigen::VectorXd::Zero(n_out);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector +=
                kernels[j] -> env_struc(sparse_env, test_structure);
        }
        kern_mat.col(i) = kernel_vector;
    }

    return kern_mat * alpha;
}