#include <cmath>
#include "ace.h"

DescriptorCalculator::DescriptorCalculator(){}

DescriptorCalculator::DescriptorCalculator(
        const std::string & radial_basis, const std::string & cutoff_function,
        const std::vector<double> & radial_hyps,
        const std::vector<double> & cutoff_hyps,
        const std::vector<int> & descriptor_settings){

    this->radial_basis = radial_basis;
    this->cutoff_function = cutoff_function;
    this->radial_hyps = radial_hyps;
    this->cutoff_hyps = cutoff_hyps;
    this->descriptor_settings = descriptor_settings;

    if (radial_basis == "chebyshev"){
        this->radial_pointer = chebyshev;
    }

    if (cutoff_function == "quadratic"){
        this->cutoff_pointer = quadratic_cutoff;
    }
    else if (cutoff_function == "hard"){
        this->cutoff_pointer = hard_cutoff;
    }
    else if (cutoff_function == "cosine"){
        this->cutoff_pointer = cos_cutoff;
    }

}

void DescriptorCalculator::compute_B1(const LocalEnvironment & env){

    // Initialize single bond vectors.
    int nos = descriptor_settings[0];
    int N = descriptor_settings[1];
    int lmax = 0;
    int no_descriptors = nos * N;

    single_bond_vals = std::vector<double> (no_descriptors, 0);
    single_bond_force_dervs =
        Eigen::MatrixXd::Zero(env.noa * 3, no_descriptors);
    single_bond_stress_dervs =
        Eigen::MatrixXd::Zero(6, no_descriptors);

    // Compute single bond vector.
    single_bond_sum_env(single_bond_vals, single_bond_force_dervs,
                        single_bond_stress_dervs, radial_pointer,
                        cutoff_pointer, env, env.cutoff, N,
                        lmax, radial_hyps, cutoff_hyps);

    // Set B1 values.
    descriptor_vals = single_bond_vals;
    descriptor_force_dervs = single_bond_force_dervs;
    descriptor_stress_dervs = single_bond_stress_dervs;
}

void DescriptorCalculator::compute_B2(const LocalEnvironment & env){

    // Initialize single bond vectors.
    int nos = descriptor_settings[0];
    int N = descriptor_settings[1];
    int lmax = descriptor_settings[2];
    int no_radial = nos * N;

    int no_harmonics = (lmax + 1) * (lmax + 1);
    int no_bond = nos * N * no_harmonics; 
    int no_descriptors = no_radial * (no_radial + 1) * (lmax + 1) / 2;

    single_bond_vals = std::vector<double> (no_bond, 0);
    single_bond_force_dervs =
        Eigen::MatrixXd::Zero(env.noa * 3, no_bond);
    single_bond_stress_dervs =
        Eigen::MatrixXd::Zero(6, no_bond);

    // Compute single bond vector.
    single_bond_sum_env(single_bond_vals, single_bond_force_dervs,
                        single_bond_stress_dervs, radial_pointer,
                        cutoff_pointer, env, env.cutoff, N,
                        lmax, radial_hyps, cutoff_hyps);

    // Initialize B2 vectors.
    descriptor_vals = std::vector<double> (no_descriptors, 0);
    descriptor_force_dervs =
        Eigen::MatrixXd::Zero(env.noa * 3, no_descriptors);
    descriptor_stress_dervs =
        Eigen::MatrixXd::Zero(6, no_descriptors);
    
    B2_descriptor(descriptor_vals, descriptor_force_dervs,
                  descriptor_stress_dervs,
                  single_bond_vals, single_bond_force_dervs,
                  single_bond_stress_dervs,
                  env, nos, N, lmax);

}

void B1_descriptor(
std::vector<double> & B1_vals,
Eigen::MatrixXd & B1_force_dervs,
Eigen::MatrixXd & B1_stress_dervs,
const std::vector<double> & single_bond_vals,
const Eigen::MatrixXd & single_bond_force_dervs,
const Eigen::MatrixXd & single_bond_stress_dervs,
const LocalEnvironment & env, int nos, int N, int lmax){

int neigh_size = env.neighbor_list.size();
int no_elements = nos * N;
int no_harmonics = (lmax + 1) * (lmax + 1);
int s, n, s_ind, n_ind, ind_curr, env_ind;
int cent_ind = env.central_index;
int counter = 0;

for (s = 0; s < nos; s ++){
    s_ind = s * N * no_harmonics;

    for (n = 0; n < N; n ++){
        n_ind = n * no_harmonics;
        ind_curr = s_ind + n_ind;

        // Store B1 value.
        B1_vals[counter] = single_bond_vals[ind_curr];

        // Store force derivatives.
        for (int atom_index = 0; atom_index < neigh_size; atom_index ++){
            env_ind = env.neighbor_list[atom_index];
            B1_force_dervs(env_ind * 3, counter) =
                single_bond_force_dervs(env_ind * 3, ind_curr);
            B1_force_dervs(env_ind * 3 + 1, counter) =
                single_bond_force_dervs(env_ind * 3 + 1, ind_curr);
            B1_force_dervs(env_ind * 3 + 2, counter) =
                single_bond_force_dervs(env_ind * 3 + 2, ind_curr);

        }

        // Store stress derivatives.
        for (int p = 0; p < 6; p ++){
            B1_stress_dervs(p, counter) =
                single_bond_stress_dervs(p, ind_curr);
        }

        counter ++;
    }
}
}

void B2_descriptor(
std::vector<double> & B2_vals,
Eigen::MatrixXd & B2_force_dervs,
Eigen::MatrixXd & B2_stress_dervs,
const std::vector<double> & single_bond_vals,
const Eigen::MatrixXd & single_bond_force_dervs,
const Eigen::MatrixXd & single_bond_stress_dervs,
const LocalEnvironment & env, int nos, int N, int lmax){

int neigh_size = env.neighbor_list.size();
int cent_ind = env.central_index;
int no_radial = nos * N;
int no_harmonics = (lmax + 1) * (lmax + 1);
int n1_ind, n2_ind, l_ind, n1_l, n2_l, env_ind;
int counter = 0;

for (int n1 = 0; n1 < no_radial; n1 ++){
    n1_ind = n1 * no_harmonics;
    for (int n2 = n1; n2 < no_radial; n2 ++){
        n2_ind = n2 * no_harmonics;
        l_ind = 0;
        for (int l = 0; l < (lmax + 1); l ++){
            for (int m = 0; m < (2 * l + 1); m ++){
                n1_l = n1_ind + l_ind;
                n2_l = n2_ind + l_ind;

                // Store B2 value.
                B2_vals[counter] +=
                    single_bond_vals[n1_l] *
                    single_bond_vals[n2_l];

                // Store force derivatives.
                for (int atom_index = 0; atom_index < neigh_size;
                     atom_index ++){
                    env_ind = env.neighbor_list[atom_index];
                    for (int comp = 0; comp < 3; comp ++){
                        B2_force_dervs(env_ind * 3 + comp, counter) +=
                            single_bond_vals[n1_l] *
                            single_bond_force_dervs(env_ind * 3 + comp, n2_l)+
                            single_bond_force_dervs(env_ind * 3 + comp, n1_l)*
                            single_bond_vals[n2_l];
                    }
                 }

                // Store stress derivatives.
                for (int p = 0; p < 6; p ++){
                    B2_stress_dervs(p, counter) +=
                        single_bond_vals[n1_l] *
                        single_bond_stress_dervs(p, n2_l) +
                        single_bond_stress_dervs(p, n1_l) *
                        single_bond_vals[n2_l];
                }
                l_ind ++;
            }
            counter ++;
        }
    }
}
};
