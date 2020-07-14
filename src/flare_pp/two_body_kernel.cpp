#include "two_body_kernel.h"
#include "cutoffs.h"
#include "local_environment.h"
#include <cmath>
#include <iostream>

TwoBodyKernel :: TwoBodyKernel() {};

TwoBodyKernel :: TwoBodyKernel(double sigma, double ls,
    const std::string & cutoff_function,
    std::vector<double> cutoff_hyps){

    this->sigma = sigma;
    sig2 = sigma * sigma;

    this->ls = ls;
    ls1 = 1 / (2 * ls * ls);
    ls2 = 1 / (ls * ls);
    ls3 = ls2 * ls2;
    this->cutoff_hyps = cutoff_hyps;

    kernel_hyperparameters = std::vector<double> {sigma, ls};

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

double TwoBodyKernel :: env_env(const LocalEnvironment & env1,
                                const LocalEnvironment & env2){
    double kern = 0;
    double ri, rj, fi, fj, rdiff;
    int e1, e2, ind1, ind2;

    double cut1 = env1.n_body_cutoffs[0];
    double cut2 = env2.n_body_cutoffs[0];
    double rcut_vals_1[2];
    double rcut_vals_2[2];
    int c1 = env1.central_species;
    int c2 = env2.central_species;
    std::vector<int> inds1 = env1.n_body_indices[0];
    std::vector<int> inds2 = env2.n_body_indices[0];

    for (int m = 0; m < inds1.size(); m ++){
        ind1 = inds1[m];
        ri = env1.rs[ind1];
        e1 = env1.environment_species[ind1];
        (*cutoff_pointer)(rcut_vals_1, ri, cut1, cutoff_hyps);
        fi = rcut_vals_1[0];
        for (int n = 0; n < inds2.size(); n ++){
            ind2 = inds2[n];
            e2 = env2.environment_species[ind2];

            // Proceed only if the pairs match.
            if ((c1 == c2 && e1 == e2) || (c1 == e2 && c2 == e1)){
                rj = env2.rs[ind2];
                (*cutoff_pointer)(rcut_vals_2, rj, cut2, cutoff_hyps);
                fj = rcut_vals_2[0];
                rdiff = ri - rj;
                kern += fi * fj * exp(-rdiff * rdiff * ls1);
            }
        }
    }
    return sig2 * kern / 4;
}

Eigen::VectorXd TwoBodyKernel :: env_env_force(const LocalEnvironment & env1,
    const LocalEnvironment & env2){

    // TODO: Implement.
    Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(3);
    return kernel_vector;
}

Eigen::VectorXd TwoBodyKernel :: self_kernel_env(
    const StructureDescriptor & struc1, int atom){

    LocalEnvironment env_curr = struc1.local_environments[atom];
    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);
    double ri, fi, fj, fdj, c1, c2, c3, c4, en_kern,
        xval1, yval1, zval1, xrel1, yrel1, zrel1,
        xval2, yval2, zval2, xrel2, yrel2, zrel2,
        diff_sq, fdi, fx, fy, fz, rj, rdiff;

    int e1, ind1, e2, ind2;
    int cent = env_curr.central_species;

    double cut = env_curr.n_body_cutoffs[0];
    double rcut_vals_1[2];
    double rcut_vals_2[2];

    double vol_inv = 1 / struc1.volume;
    double vol_inv_sq = vol_inv * vol_inv;

    std::vector<int> inds = env_curr.n_body_indices[0];

    for (int m = 0; m < inds.size(); m ++){
        ind1 = inds[m];
        ri = env_curr.rs[ind1];
        (*cutoff_pointer)(rcut_vals_1, ri, cut, cutoff_hyps);
        fi = rcut_vals_1[0];
        fdi = rcut_vals_1[1];
        e1 = env_curr.environment_species[ind1];
    
        xval1 = env_curr.xs[ind1];
        yval1 = env_curr.ys[ind1];
        zval1 = env_curr.zs[ind1];
        xrel1 = env_curr.xrel[ind1];
        yrel1 = env_curr.yrel[ind1];
        zrel1 = env_curr.zrel[ind1];

        for (int n = 0; n < inds.size(); n ++){
            ind2 = inds[n];
            e2 = env_curr.environment_species[ind2];

            // Proceed only if the pairs match.
            if (e1 == e2){
                rj = env_curr.rs[ind2];
                rdiff = ri - rj;

                xval2 = env_curr.xs[ind2];
                yval2 = env_curr.ys[ind2];
                zval2 = env_curr.zs[ind2];
                xrel2 = env_curr.xrel[ind2];
                yrel2 = env_curr.yrel[ind2];
                zrel2 = env_curr.zrel[ind2];

                (*cutoff_pointer)(rcut_vals_2, rj, cut, cutoff_hyps);
                fj = rcut_vals_2[0];
                fdj = rcut_vals_2[1];

                // energy kernel
                c1 = rdiff * rdiff;
                c2 = exp(-c1 * ls1);
                kernel_vector(0) += sig2 * fi * fj * c2 / 4;

                // fx + exx, exy, exz stress component
                fx = force_helper(xrel1 * xrel2, rdiff * xrel1, rdiff * xrel2,
                    c1, fi, fj, -fdi * xrel1, -fdj * xrel2, ls1, ls2, ls3,
                    sig2);

                kernel_vector(1 + 3 * atom) += fx;
                kernel_vector(no_elements - 6) +=
                    fx * xval1 * xval2 * vol_inv_sq / 4;
                kernel_vector(no_elements - 5) +=
                    fx * yval1 * yval2 * vol_inv_sq / 4;
                kernel_vector(no_elements - 4) +=
                    fx * zval1 * zval2 * vol_inv_sq / 4;

                // fy + eyy, eyz stress component
                fy = force_helper(yrel1 * yrel2, rdiff * yrel1, rdiff * yrel2,
                    c1, fi, fj, -fdi * yrel1, -fdj * yrel2, ls1, ls2, ls3,
                    sig2);

                kernel_vector(2 + 3 * atom) += fy;
                kernel_vector(no_elements - 3) +=
                    fy * yval1 * yval2 * vol_inv_sq / 4;
                kernel_vector(no_elements - 2) +=
                    fy * zval1 * zval2 * vol_inv_sq / 4;

                // fz + ezz stress component
                fz = force_helper(zrel1 * zrel2, rdiff * zrel1, rdiff * zrel2,
                    c1, fi, fj, -fdi * zrel1, -fdj * zrel2, ls1, ls2, ls3,
                    sig2);

                kernel_vector(3 + 3 * atom) += fz;
                kernel_vector(no_elements - 1) +=
                    fz * zval1 * zval2 * vol_inv_sq / 4;
            }
        }
    }

    return kernel_vector;
}

Eigen::VectorXd TwoBodyKernel :: self_kernel_struc(
    const StructureDescriptor & struc){

    int no_elements = 1 + 3 * struc.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);
    double ri, fi, fj, fdj, c1, c2, c3, c4, en_kern,
        xval1, yval1, zval1, xrel1, yrel1, zrel1,
        xval2, yval2, zval2, xrel2, yrel2, zrel2,
        diff_sq, fdi, fx, fy, fz, rj, rdiff,
        cut1, cut2, mult_fac;

    int e1, ind1, e2, ind2, cent1, cent2;

    std::vector<int> inds1, inds2;

    double rcut_vals_1[2], rcut_vals_2[2];

    double vol_inv = 1 / struc.volume;
    double vol_inv_sq = vol_inv * vol_inv;

    // Double loop over environments.
    LocalEnvironment env1, env2;
    for (int i = 0; i < struc.noa; i ++){
        env1 = struc.local_environments[i];
        cut1 = env1.n_body_cutoffs[0];
        cent1 = env1.central_species;
        inds1 = env1.n_body_indices[0];
        for (int j = i; j < struc.noa; j ++){
            env2 = struc.local_environments[j];
            cut2 = env2.n_body_cutoffs[0];
            cent2 = env2.central_species;
            inds2 = env2.n_body_indices[0];

            if (i == j){
                mult_fac = 1;
            }
            else{
                mult_fac = 2;
            }

            // Loop over neighbors.
            for (int m = 0; m < inds1.size(); m ++){
                ind1 = inds1[m];
                ri = env1.rs[ind1];
                (*cutoff_pointer)(rcut_vals_1, ri, cut1, cutoff_hyps);
                fi = rcut_vals_1[0];
                fdi = rcut_vals_1[1];
                e1 = env1.environment_species[ind1];
            
                xval1 = env1.xs[ind1];
                yval1 = env1.ys[ind1];
                zval1 = env1.zs[ind1];
                xrel1 = env1.xrel[ind1];
                yrel1 = env1.yrel[ind1];
                zrel1 = env1.zrel[ind1];

                for (int n = 0; n < inds2.size(); n ++){
                    ind2 = inds2[n];
                    e2 = env2.environment_species[ind2];

                    // Proceed only if the pairs match.
                    if ((cent1 == cent2 && e1 == e2) ||
                        (cent1 == e2 && cent2 == e1)){

                        rj = env2.rs[ind2];
                        rdiff = ri - rj;

                        xval2 = env2.xs[ind2];
                        yval2 = env2.ys[ind2];
                        zval2 = env2.zs[ind2];
                        xrel2 = env2.xrel[ind2];
                        yrel2 = env2.yrel[ind2];
                        zrel2 = env2.zrel[ind2];

                        (*cutoff_pointer)(rcut_vals_2, rj, cut2, cutoff_hyps);
                        fj = rcut_vals_2[0];
                        fdj = rcut_vals_2[1];

                        // energy kernel
                        c1 = rdiff * rdiff;
                        c2 = exp(-c1 * ls1);
                        kernel_vector(0) += mult_fac * sig2 * fi * fj * c2 / 4;

                        // fx + exx, exy, exz stress component
                        fx = force_helper(
                            xrel1 * xrel2, rdiff * xrel1, rdiff * xrel2,
                            c1, fi, fj, -fdi * xrel1, -fdj * xrel2, ls1, ls2,
                            ls3, sig2);

                        kernel_vector(no_elements - 6) +=
                            mult_fac * fx * xval1 * xval2 * vol_inv_sq / 4;
                        kernel_vector(no_elements - 5) +=
                            mult_fac * fx * yval1 * yval2 * vol_inv_sq / 4;
                        kernel_vector(no_elements - 4) +=
                            mult_fac * fx * zval1 * zval2 * vol_inv_sq / 4;

                        // fy + eyy, eyz stress component
                        fy = force_helper(
                            yrel1 * yrel2, rdiff * yrel1, rdiff * yrel2,
                            c1, fi, fj, -fdi * yrel1, -fdj * yrel2, ls1, ls2,
                            ls3, sig2);

                        kernel_vector(no_elements - 3) +=
                            mult_fac * fy * yval1 * yval2 * vol_inv_sq / 4;
                        kernel_vector(no_elements - 2) +=
                            mult_fac * fy * zval1 * zval2 * vol_inv_sq / 4;

                        // fz + ezz stress component
                        fz = force_helper(
                            zrel1 * zrel2, rdiff * zrel1, rdiff * zrel2,
                            c1, fi, fj, -fdi * zrel1, -fdj * zrel2, ls1, ls2,
                            ls3, sig2);

                        kernel_vector(no_elements - 1) +=
                            mult_fac * fz * zval1 * zval2 * vol_inv_sq / 4;

                        if (i == j){
                            kernel_vector(1 + 3 * i) += fx;
                            kernel_vector(2 + 3 * i) += fy;
                            kernel_vector(3 + 3 * i) += fz;
                        }
                    }
                }
            }
        }
    }

    return kernel_vector;
}

Eigen::VectorXd TwoBodyKernel :: env_struc_partial(
    const LocalEnvironment & env1, const StructureDescriptor & struc1,
    int atom){

    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);
    double ri, rj, fi, fj, fdj, rdiff, c1, c2, c3, c4, fx, fy, fz, xval, yval,
        zval, xrel, yrel, zrel, en_kern;

    int e1, e2, cent2, ind1, ind2;
    int cent1 = env1.central_species;

    double cut1 = env1.n_body_cutoffs[0];
    double cut2 = struc1.n_body_cutoffs[0];
    double rcut_vals_1[2];
    double rcut_vals_2[2];

    double vol_inv = 1 / struc1.volume;

    LocalEnvironment env_curr = struc1.local_environments[atom];
    std::vector<int> inds1 = env1.n_body_indices[0];
    std::vector<int> inds2 = env_curr.n_body_indices[0];
    cent2 = env_curr.central_species;

    for (int m = 0; m < inds1.size(); m ++){
        ind1 = inds1[m];
        ri = env1.rs[ind1];
        (*cutoff_pointer)(rcut_vals_1, ri, cut1, cutoff_hyps);
        fi = rcut_vals_1[0];
        e1 = env1.environment_species[ind1];

        for (int n = 0; n < inds2.size(); n ++){
            ind2 = inds2[n];
            e2 = env_curr.environment_species[ind2];

            // Proceed only if the pairs match.
            if ((cent1 == cent2 && e1 == e2) || (cent1 == e2 && 
                cent2 == e1)){
                rj = env_curr.rs[ind2];
                rdiff = ri - rj;

                xval = env_curr.xs[ind2];
                yval = env_curr.ys[ind2];
                zval = env_curr.zs[ind2];
                xrel = env_curr.xrel[ind2];
                yrel = env_curr.yrel[ind2];
                zrel = env_curr.zrel[ind2];

                (*cutoff_pointer)(rcut_vals_2, rj, cut2, cutoff_hyps);
                fj = rcut_vals_2[0];
                fdj = rcut_vals_2[1];

                // energy kernel
                // Note on fractional factors: the local energy of an atomic environment is defined as the sum of pair energies divided by 2 to avoid overcounting.
                c1 = rdiff * rdiff;
                c2 = exp(-c1 * ls1);
                kernel_vector(0) += sig2 * fi * fj * c2 / 4;

                // helper constants
                c3 = c2 * ls2 * fi * fj * rdiff;
                c4 = c2 * fi * fdj;

                // fx + exx, exy, exz stress components
                fx = xrel * c3 + c4 * xrel;
                kernel_vector(1 + 3 * atom) += sig2 * fx / 2;
                kernel_vector(no_elements - 6) -=
                    sig2 * fx * xval * vol_inv / 4;
                kernel_vector(no_elements - 5) -=
                    sig2 * fx * yval * vol_inv / 4;
                kernel_vector(no_elements - 4) -=
                    sig2 * fx * zval * vol_inv / 4;

                // fy + eyy, eyz stress components
                fy = yrel * c3 + c4 * yrel;
                kernel_vector(2 + 3 * atom) += sig2 * fy / 2;
                kernel_vector(no_elements - 3) -=
                    sig2 * fy * yval * vol_inv / 4;
                kernel_vector(no_elements - 2) -=
                    sig2 * fy * zval * vol_inv / 4;

                // fz + ezz stress component
                fz = zrel * c3 + c4 * zrel;
                kernel_vector(3 + 3 * atom) += sig2 * fz / 2;
                kernel_vector(no_elements - 1) -=
                    sig2 * fz * zval * vol_inv / 4;
            }
        }
    } 

    return kernel_vector;
}

Eigen::VectorXd TwoBodyKernel :: env_struc(const LocalEnvironment & env1,
    const StructureDescriptor & struc1){
    
    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);

    for (int i = 0; i < struc1.noa; i ++){
        kernel_vector += env_struc_partial(env1, struc1, i);
    }

    return kernel_vector;
}

double force_helper(double rel1_rel2, double diff_rel1, double diff_rel2,
    double diff_sq, double fi, double fj, double fdi, double fdj, double l1,
    double l2, double l3, double s2){

    double A, B, C, D, E, F, G, H, I;

    A = exp(-diff_sq * l1);
    B = A * diff_rel1 * l2;
    C = -A * diff_rel2 * l2;
    D = rel1_rel2 * A * l2 - diff_rel1 * diff_rel2 * A * l3;
    E = A * fdi * fdj;
    F = B * fi * fdj;
    G = C * fdi * fj;
    H = D * fi * fj;
    I = s2 * (E + F + G + H);

    return I;
}
