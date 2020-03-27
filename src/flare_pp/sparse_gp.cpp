#include "sparse_gp.h"
#include <cmath>
#include <iostream>

static const double Pi = 3.14159265358979323846;

SparseGP :: SparseGP(){}

SparseGP :: SparseGP(std::vector<Kernel *> kernels, double sigma_e,
    double sigma_f, double sigma_s){

    this->kernels = kernels;
    Kuu_jitter = 1e-8;  // default value

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
    int n_labels = Kuf_struc.cols();
    int n_strucs = training_structures.size();
    Eigen::VectorXd uf_vector = Eigen::VectorXd::Zero(n_labels);

    #pragma omp parallel for
    for (int i = 0; i < n_strucs; i ++){
        int initial_index, index;

        // Get initial index.
        if (i == 0){
            initial_index = 0;
        }
        else{
            initial_index = label_count[i - 1];
        }
        index = initial_index;

        int n_atoms = training_structures[i].noa;
        Eigen::VectorXd kernel_vector =\
            Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector += kernels[j] -> env_struc(env,
                training_structures[i]);
        }

        if (training_structures[i].energy.size() != 0){
            uf_vector(index) = kernel_vector(0);
            index += 1;
        }

        if (training_structures[i].forces.size() != 0){
            uf_vector.segment(index, n_atoms * 3) =
                kernel_vector.segment(1, n_atoms * 3);
            index += n_atoms * 3;
        }

        if (training_structures[i].stresses.size() != 0){
            uf_vector.segment(index, 6) = kernel_vector.tail(6);
        }
    }

    // Update Kuf_struc matrix.
    Kuf_struc.conservativeResize(Kuf_struc.rows()+1, Kuf_struc.cols());
    Kuf_struc.row(Kuf_struc.rows()-1) = uf_vector;

    // Compute kernels between new environment and training environments.

    // Store sparse environment.
    sparse_environments.push_back(env);
}

void SparseGP :: add_training_structure(StructureDescriptor training_structure){

    int n_labels = training_structure.energy.size() +
        training_structure.forces.size() + training_structure.stresses.size();
    int n_atoms = training_structure.noa;
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();

    // Update label counts.
    int prev_count;
    int curr_size = label_count.size();
    if (label_count.size() == 0){
        label_count.push_back(n_labels);
    }
    else{
        prev_count = label_count[curr_size-1];
        label_count.push_back(n_labels + prev_count);
    }

    // Calculate kernels between sparse environments and training structure.
    Eigen::MatrixXd kernel_block = Eigen::MatrixXd::Zero(n_sparse, n_labels);

    #pragma omp parallel for
    for (int i = 0; i < n_sparse; i ++){
        Eigen::VectorXd kernel_vector =
            Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector += kernels[j] -> env_struc(sparse_environments[i],
                training_structure);
        }

        // Update kernel block.
        int count = 0;
        if (training_structure.energy.size() != 0){
            kernel_block(i, 0) = kernel_vector(0);
            count += 1;
        }

        if (training_structure.forces.size() != 0){
            kernel_block.row(i).segment(count, n_atoms * 3) =
                kernel_vector.segment(1, n_atoms * 3);
        }

        if (training_structure.stresses.size() != 0){
            kernel_block.row(i).tail(6) = kernel_vector.tail(6);
        }
    }

    // Add kernel block to Kuf_struc.
    int prev_cols = Kuf_struc.cols();
    Kuf_struc.conservativeResize(n_sparse, prev_cols + n_labels);
    Kuf_struc.block(0, prev_cols, n_sparse, n_labels) = kernel_block;

    // Store training structure.
    training_structures.push_back(training_structure);

    // Update y vector and noise matrix.
    Eigen::VectorXd labels = Eigen::VectorXd::Zero(n_labels);
    Eigen::VectorXd noise_vector = Eigen::VectorXd::Zero(n_labels);

    int count = 0;
    if (training_structure.energy.size() != 0){
        labels.head(1) = training_structure.energy;
        noise_vector(0) = 1 / (sigma_e * sigma_e);
        count ++;
    }

    if (training_structure.forces.size() != 0){
        labels.segment(count, n_atoms * 3) = training_structure.forces;
        noise_vector.segment(count, n_atoms * 3) =
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

// TODO: decouple triplets of different types.
void SparseGP :: three_body_grid(double min_dist, double max_dist,
    double cutoff, int n_species, int n_dist, int n_angle){

    double dist1, dist2, dist3, angle;
    double dist_step = (max_dist - min_dist) / (n_dist - 1);
    double angle_step = Pi / (n_angle - 1);

    // Create cell. Both max_dist and cutoff should be no greater than half the cell size to avoid periodic images.
    double max_val;
    if (max_dist > cutoff){
        max_val = max_dist;
    }
    else{
        max_val = cutoff;
    }

    double struc_factor = 10;
    Eigen::MatrixXd cell{3, 3};
    cell << max_val * struc_factor, 0, 0,
            0, max_val * struc_factor, 0,
            0, 0, max_val * struc_factor;
    std::vector<int> species;
    Eigen::MatrixXd positions = Eigen::MatrixXd::Zero(3, 3);
    std::vector<double> n_body_cutoffs {cutoff, cutoff};
    StructureDescriptor struc;
    LocalEnvironment env;
    int counter = 0;

    // Loop over species.
    for (int s1 = 0; s1 < n_species; s1 ++){
        for (int s2 = s1; s2 < n_species; s2 ++){
            for (int s3 = s2; s3 < n_species; s3 ++){
                species = {s1, s2, s3};

                // Loop over distances.
                for (int d1 = 0; d1 < n_dist; d1 ++){
                    dist1 = min_dist + d1 * dist_step;

                    for (int d2 = 0; d2 < n_dist; d2 ++){
                        dist2 = min_dist + d2 * dist_step;

                        if ((s2 == s3) && (dist2 < dist1)){
                            continue;
                        }
                        else{
                            // Loop over angles ranging from 0 to Pi.
                            for (int a1 = 0; a1 < n_angle; a1 ++){
                                angle = a1 * angle_step;
                                dist3 = sqrt(dist1 * dist1 + dist2 * dist2 -
                                    2 * dist1 * dist2 * cos(angle));

                                // Skip over triplets for which dist3 exceeds the cutoff.
                                if (dist3 > cutoff){
                                    continue;
                                }
                                // If all three species are the same, neglect the case where dist3 < dist2.
                                else if ((s1 == s2) && (s2 == s3) && (dist3 < dist2)){
                                    continue;
                                }
                                else{
                                    // Create structure of 3 atoms.
                                    positions(1, 0) = dist1;
                                    positions(2, 0) = dist2 * cos(angle);
                                    positions(2, 1) = dist2 * sin(angle);
                                    struc = StructureDescriptor(cell, species,
                                        positions, cutoff, n_body_cutoffs);

                                    // Create environment.
                                    env = struc.local_environments[0];

                                    // Add to the training set
                                    this->add_sparse_environment(env);

                                    counter ++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void SparseGP::update_alpha(){
    Eigen::MatrixXd sigma_inv = Kuu + Kuf_struc * noise_matrix * Kuf_struc.transpose() +
        Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
    // TODO: Use Woodbury identity to perform inversion once.
    Sigma = sigma_inv.inverse();
    Kuu_inverse = Kuu.inverse();
    alpha = Sigma * Kuf_struc * noise_matrix * y;
}

Eigen::VectorXd SparseGP::predict(StructureDescriptor test_structure){
    int n_atoms = test_structure.noa;
    int n_out = 1 + 3 * n_atoms + 6;
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();
    Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(n_out, n_sparse);

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
