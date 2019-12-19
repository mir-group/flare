#include <cmath>
#include "ace.h"

void B1_descriptor(
double * descriptor_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, double *),
void (*cutoff_function)(double *, double, double, double *),
double * xs, double * ys, double * zs, double * rs, int * species,
int nos, int noa, double rcut, int N,
std::vector<double> radial_hyps, std::vector<double> cutoff_hyps){

// TODO: implement this!

}

void B2_descriptor(
double * descriptor_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, double *),
void (*cutoff_function)(double *, double, double, double *),
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
