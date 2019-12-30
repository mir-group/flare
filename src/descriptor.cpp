#include <cmath>
#include "ace.h"

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

        // Store central atom force derivatives.
        B1_force_dervs(cent_ind * 3, counter) =
            single_bond_force_dervs(cent_ind * 3, ind_curr);
        B1_force_dervs(cent_ind * 3 + 1, counter) =
            single_bond_force_dervs(cent_ind * 3 + 1, ind_curr);
        B1_force_dervs(cent_ind * 3 + 2, counter) =
            single_bond_force_dervs(cent_ind * 3 + 2, ind_curr);

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
double * descriptor_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, std::vector<double>),
void (*cutoff_function)(double *, double, double, std::vector<double>),
double * xs, double * ys, double * zs, double * rs, int * species,
int nos, int noa, double rcut, int N, int lmax,
std::vector<double> radial_hyps, std::vector<double> cutoff_hyps){

// Calculate atomic base.
int number_of_harmonics = (lmax + 1) * (lmax + 1);
int no_basis_vals = N * number_of_harmonics;

double * atomic_base = new double[nos * N * number_of_harmonics]();
double * base_env_dervs = 
    new double[nos * noa * N * number_of_harmonics * 3]();
double * base_cent_dervs =
    new double[nos * N * number_of_harmonics * 3]();

single_bond_sum(atomic_base, base_env_dervs, base_cent_dervs,
                basis_function, cutoff_function, xs, ys, zs, rs,
                species, noa, rcut, N, lmax,
                radial_hyps, cutoff_hyps);

// Calculate rotational invariants.
int s1, s2, n1, n2, l, m, s1_ind, s2_ind, n1_ind, n2_ind;
int l_ind;
int counter = 0;
for (s1 = 0; s1 < nos; s1 ++){
    s1_ind = s1 * no_basis_vals;

    for (s2 = s1; s2 < nos; s2 ++){
        s2_ind = s2 * no_basis_vals;

        // Note that there is redundancy when s1 = s2.
        for (n1 = 0; n1 < N; n1 ++){
            n1_ind = n1 * number_of_harmonics;

            for (n2 = 0; n2 < N; n2 ++){
                n2_ind = n2 * number_of_harmonics;

                l_ind = 0;
                for (l = 0; l < (lmax + 1); l ++){

                    for (m = 0; m < (2 * l + 1); m ++){
                        descriptor_vals[counter] +=
                            atomic_base[s1_ind + n1_ind + l_ind] *
                            atomic_base[s2_ind + n2_ind + l_ind];

                        l_ind += 1;
                    }
                    counter += 1;
                }
            }
        }      
    }
}

delete [] atomic_base;
delete [] base_env_dervs;
delete [] base_cent_dervs;
}
