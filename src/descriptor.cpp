#include "descriptor.h"
#include "radial.h"
#include "cutoffs.h"
#include "local_environment.h"
#include "single_bond.h"
#include <cmath>
#include <iostream>

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


void B2_descriptor(
Eigen::VectorXd & B2_vals,
Eigen::MatrixXd & B2_force_dervs,
Eigen::MatrixXd & B2_stress_dervs,
const Eigen::VectorXd & single_bond_vals,
const Eigen::MatrixXd & single_bond_force_dervs,
const Eigen::MatrixXd & single_bond_stress_dervs,
const LocalEnvironment & env, int nos, int N, int lmax){

int neigh_size = env.neighbor_list.size();
int cent_ind = env.central_index;
int no_radial = nos * N;
int no_harmonics = (lmax + 1) * (lmax + 1);
int l_ind, n1_l, n2_l, env_ind;
int counter = 0;
int temp_counter;

for (int n1 = 0; n1 < no_radial * no_harmonics; n1 += no_harmonics){
    for (int n2 = n1; n2 < no_radial * no_harmonics; n2 += no_harmonics){
        for (int l = 0; l < (lmax + 1); l ++){
            // counter = l + n2 * (lmax + 1) + n1
            for (int m = 0; m < (2 * l + 1); m ++){
                n1_l = n1 + (l * l + m);
                n2_l = n2 + (l * l + m);

                // Store B2 value.
                B2_vals(counter) +=
                    single_bond_vals(n1_l) *
                    single_bond_vals(n2_l);

                // Store force derivatives.
                for (int atom_index = 0; atom_index < neigh_size;
                     atom_index ++){
                    env_ind = env.neighbor_list[atom_index];
                    for (int comp = 0; comp < 3; comp ++){
                        B2_force_dervs(env_ind * 3 + comp, counter) +=
                            single_bond_vals(n1_l) *
                            single_bond_force_dervs(env_ind * 3 + comp, n2_l)+
                            single_bond_force_dervs(env_ind * 3 + comp, n1_l)*
                            single_bond_vals(n2_l);
                    }
                 }

                // Store stress derivatives.
                for (int p = 0; p < 6; p ++){
                    B2_stress_dervs(p, counter) +=
                        single_bond_vals(n1_l) *
                        single_bond_stress_dervs(p, n2_l) +
                        single_bond_stress_dervs(p, n1_l) *
                        single_bond_vals(n2_l);
                }
            }


            temp_counter = (l + (n2*(lmax+1))); //+ (n1 * (no_radial * no_harmonics - n1) * (n1 + no_radial * no_harmonics - 1) * (lmax+1)));
            // if (temp_counter != counter) {
                std::cout << counter << " " << temp_counter << std::endl;
            // } else {

            // }

            counter ++;
        }
    }
    assert(false);
    
}
};

B1_Calculator :: B1_Calculator(){}

B1_Calculator :: B1_Calculator(const std::string & radial_basis,
    const std::string & cutoff_function,
    const std::vector<double> & radial_hyps,
    const std::vector<double> & cutoff_hyps,
    const std::vector<int> & descriptor_settings)
    : DescriptorCalculator(radial_basis, cutoff_function, radial_hyps,
        cutoff_hyps, descriptor_settings){}

void B1_Calculator :: compute(const LocalEnvironment & env){
    // Initialize single bond vectors.
    int nos = descriptor_settings[0];
    int N = descriptor_settings[1];
    int lmax = 0;
    int no_descriptors = nos * N;

    single_bond_vals = Eigen::VectorXd::Zero(no_descriptors);
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

B2_Calculator :: B2_Calculator(){}

B2_Calculator :: B2_Calculator(const std::string & radial_basis,
    const std::string & cutoff_function,
    const std::vector<double> & radial_hyps,
    const std::vector<double> & cutoff_hyps,
    const std::vector<int> & descriptor_settings)
    : DescriptorCalculator(radial_basis, cutoff_function, radial_hyps,
        cutoff_hyps, descriptor_settings){}

void B2_Calculator :: compute(const LocalEnvironment & env){
    // Initialize single bond vectors.
    int nos = descriptor_settings[0];
    int N = descriptor_settings[1];
    int lmax = descriptor_settings[2];
    int no_radial = nos * N;

    int no_harmonics = (lmax + 1) * (lmax + 1);

    int no_bond = no_radial * no_harmonics;
    int no_descriptors = (no_radial * (no_radial + 1) / 2) * (lmax + 1);

    single_bond_vals = Eigen::VectorXd::Zero(no_bond);
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
    descriptor_vals = Eigen::VectorXd::Zero(no_descriptors);
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
