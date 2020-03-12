#include "kernels.h"
#include "cutoffs.h"
#include "local_environment.h"
#include <cmath>
#include <iostream>

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

Kernel :: Kernel() {};
Kernel :: Kernel(std::vector<double> kernel_hyperparameters){
    this->kernel_hyperparameters = kernel_hyperparameters;
};

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

// TODO: test env_env method
// factor of 1/4 missing?
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

ThreeBodyKernel :: ThreeBodyKernel() {};

ThreeBodyKernel :: ThreeBodyKernel(double sigma, double ls,
    const std::string & cutoff_function, std::vector<double> cutoff_hyps){

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

double ThreeBodyKernel :: env_env(const LocalEnvironment & env1,
                                  const LocalEnvironment & env2){
    double kern = 0;
    double ri1, ri2, ri3, rj1, rj2, rj3, fi1, fi2, fi3, fj1, fj2, fj3,
        fdi1, fdi2, fdi3, fdj1, fdj2, fdj3, fi, fj, r11, r12, r13, r21,
        r22, r23, r31, r32, r33, p1, p2, p3, p4, p5, p6;
    int i1, i2, j1, j2, ei1, ei2, ej1, ej2;

    double cut1 = env1.n_body_cutoffs[1];
    double cut2 = env2.n_body_cutoffs[1];
    double rcut_vals_i1[2], rcut_vals_i2[2], rcut_vals_i3[2],
        rcut_vals_j1[2], rcut_vals_j2[2], rcut_vals_j3[2];
    int c1 = env1.central_species;
    int c2 = env2.central_species;

    for (int m = 0; m < env1.three_body_indices.size(); m ++){
        i1 = env1.three_body_indices[m][0];
        i2 = env1.three_body_indices[m][1];

        ri1 = env1.rs[i1];
        ri2 = env1.rs[i2];
        ri3 = env1.cross_bond_dists[m];

        ei1 = env1.environment_species[i1];
        ei2 = env1.environment_species[i2];

        (*cutoff_pointer)(rcut_vals_i1, ri1, cut1, cutoff_hyps);
        (*cutoff_pointer)(rcut_vals_i2, ri2, cut1, cutoff_hyps);
        (*cutoff_pointer)(rcut_vals_i3, ri3, cut1, cutoff_hyps);

        fi1 = rcut_vals_i1[0];
        fi2 = rcut_vals_i2[0];
        fi3 = rcut_vals_i3[0];
        fi = fi1 * fi2 * fi3;

        fdi1 = rcut_vals_i1[1];
        fdi2 = rcut_vals_i2[1];

        for (int n = 0; n < env2.three_body_indices.size(); n ++){
            j1 = env2.three_body_indices[n][0];
            j2 = env2.three_body_indices[n][1];

            rj1 = env2.rs[j1];
            rj2 = env2.rs[j2];
            rj3 = env2.cross_bond_dists[n];

            ej1 = env2.environment_species[j1];
            ej2 = env2.environment_species[j2];

            (*cutoff_pointer)(rcut_vals_j1, rj1, cut2, cutoff_hyps);
            (*cutoff_pointer)(rcut_vals_j2, rj2, cut2, cutoff_hyps);
            (*cutoff_pointer)(rcut_vals_j3, rj3, cut1, cutoff_hyps);

            fj1 = rcut_vals_j1[0];
            fj2 = rcut_vals_j2[0];
            fj3 = rcut_vals_j3[0];
            fj = fj1 * fj2 * fj3;

            fdj1 = rcut_vals_j1[1];
            fdj2 = rcut_vals_j2[1];

            r11 = ri1 - rj1;
            r12 = ri1 - rj2;
            r13 = ri1 - rj3;
            r21 = ri2 - rj1;
            r22 = ri2 - rj2;
            r23 = ri2 - rj3;
            r31 = ri3 - rj1;
            r32 = ri3 - rj2;
            r33 = ri3 - rj3;

            // Sum over six permutations.
            if (c1 == c2){
                if (ei1 == ej1 && ei2 == ej2){
                    p1 = r11 * r11 + r22 * r22 + r33 * r33;
                    kern += exp(-p1 * ls2) * fi * fj;
                }
                if (ei1 == ej2 && ei2 == ej1){
                    p2 = r12 * r12 + r21 * r21 + r33 * r33;
                    kern += exp(-p2 * ls2) * fi * fj;
                }
            }

            if (c1 == ej1){
                if (ei1 == ej2 && ei2 == c2){
                    p3 = r13 * r13 + r21 * r21 + r32 * r32;
                    kern += exp(-p3 * ls2) * fi * fj;
                }
                if (ei1 == c2 && ei2 == ej2){
                    p4 = r11 * r11 + r23 * r23 + r32 * r32;
                    kern += exp(-p4 * ls2) * fi * fj;
                }
            }

            if (c1 == ej2){
                if (ei1 == ej1 && ei2 == c2){
                    p5 = r13 * r13 + r22 * r22 + r31 * r31;
                    kern += exp(-p5 * ls2) * fi * fj;
                }
                if (ei1 == c2 && ei2 == ej1){
                    p6 = r12*r12+r23*r23+r31*r31;
                    kern += exp(-p6 * ls2) * fi * fj;
                }
            }
        }
    }

    return sig2 * kern / 9;
}

Eigen::VectorXd ThreeBodyKernel :: self_kernel_env(
    const StructureDescriptor & struc1, int atom){

    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);
    
    // TODO: implement the rest

    return kernel_vector;
}

Eigen::VectorXd ThreeBodyKernel :: env_struc_partial(
    const LocalEnvironment & env1, const StructureDescriptor & struc1,
    int atom){

    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);

    double ri1, ri2, ri3, rj1, rj2, rj3, fi1, fi2, fi3, fj1, fj2, fj3,
        fdj1, fdj2, fi, fj, r11, r12, r13, r21, r22, r23, r31, r32, r33, p1,
        p2, p3, p4, xval1, xval2, xrel1, xrel2, yval1, yval2, yrel1,
        yrel2, zval1, zval2, zrel1, zrel2, fdjx1, fdjx2, fdjy1, fdjy2, fdjz1,
        fdjz2, fx1, fx2, fy1, fy2, fz1, fz2;
    int i1, i2, j1, j2, ei1, ei2, ej1, ej2;

    LocalEnvironment env2 = struc1.local_environments[atom];
    double vol_inv = 1 / struc1.volume;

    double cut1 = env1.n_body_cutoffs[1];
    double cut2 = struc1.n_body_cutoffs[1];
    double rcut_vals_i1[2], rcut_vals_i2[2], rcut_vals_i3[2],
        rcut_vals_j1[2], rcut_vals_j2[2], rcut_vals_j3[2];
    int c1 = env1.central_species;
    int c2 = env2.central_species;
        
    for (int m = 0; m < env1.three_body_indices.size(); m ++){
        i1 = env1.three_body_indices[m][0];
        i2 = env1.three_body_indices[m][1];

        ri1 = env1.rs[i1];
        ri2 = env1.rs[i2];
        ri3 = env1.cross_bond_dists[m];

        ei1 = env1.environment_species[i1];
        ei2 = env1.environment_species[i2];

        (*cutoff_pointer)(rcut_vals_i1, ri1, cut1, cutoff_hyps);
        (*cutoff_pointer)(rcut_vals_i2, ri2, cut1, cutoff_hyps);
        (*cutoff_pointer)(rcut_vals_i3, ri3, cut1, cutoff_hyps);

        fi1 = rcut_vals_i1[0];
        fi2 = rcut_vals_i2[0];
        fi3 = rcut_vals_i3[0];
        fi = fi1 * fi2 * fi3;

        for (int n = 0; n < env2.cross_bond_dists.size(); n ++){
            j1 = env2.three_body_indices[n][0];
            j2 = env2.three_body_indices[n][1];

            rj1 = env2.rs[j1];
            rj2 = env2.rs[j2];
            rj3 = env2.cross_bond_dists[n];

            ej1 = env2.environment_species[j1];
            ej2 = env2.environment_species[j2];

            xval1 = env2.xs[j1];
            yval1 = env2.ys[j1];
            zval1 = env2.zs[j1];
            xrel1 = env2.xrel[j1];
            yrel1 = env2.yrel[j1];
            zrel1 = env2.zrel[j1];

            xval2 = env2.xs[j2];
            yval2 = env2.ys[j2];
            zval2 = env2.zs[j2];
            xrel2 = env2.xrel[j2];
            yrel2 = env2.yrel[j2];
            zrel2 = env2.zrel[j2];

            (*cutoff_pointer)(rcut_vals_j1, rj1, cut2, cutoff_hyps);
            (*cutoff_pointer)(rcut_vals_j2, rj2, cut2, cutoff_hyps);
            (*cutoff_pointer)(rcut_vals_j3, rj3, cut2, cutoff_hyps);

            fj1 = rcut_vals_j1[0];
            fdj1 = rcut_vals_j1[1];
            fj2 = rcut_vals_j2[0];
            fdj2 = rcut_vals_j2[1];
            fj3 = rcut_vals_j3[0];
            fj = fj1 * fj2 * fj3;

            fdjx1 = xrel1 * fdj1 * fj2 * fj3;
            fdjy1 = yrel1 * fdj1 * fj2 * fj3;
            fdjz1 = zrel1 * fdj1 * fj2 * fj3;
            fdjx2 = xrel2 * fj1 * fdj2 * fj3;
            fdjy2 = yrel2 * fj1 * fdj2 * fj3;
            fdjz2 = zrel2 * fj1 * fdj2 * fj3;

            r11 = ri1 - rj1;
            r12 = ri1 - rj2;
            r13 = ri1 - rj3;
            r21 = ri2 - rj1;
            r22 = ri2 - rj2;
            r23 = ri2 - rj3;
            r31 = ri3 - rj1;
            r32 = ri3 - rj2;
            r33 = ri3 - rj3;

            // Sum over six permutations.
            if (c1 == c2){
                if (ei1 == ej1 && ei2 == ej2){
                    update_kernel_vector(kernel_vector, no_elements, atom,
                        vol_inv, r11, r22, r33, fi, fj,
                        fdjx1, fdjx2, fdjy1, fdjy2, fdjz1, fdjz2,
                        xrel1, xval1, xrel2, xval2, yrel1, yval1, yrel2,
                        yval2, zrel1, zval1, zrel2, zval2);
                }
                if (ei1 == ej2 && ei2 == ej1){
                    update_kernel_vector(kernel_vector, no_elements, atom,
                        vol_inv, r21, r12, r33, fi, fj,
                        fdjx1, fdjx2, fdjy1, fdjy2, fdjz1, fdjz2,
                        xrel1, xval1, xrel2, xval2, yrel1, yval1, yrel2,
                        yval2, zrel1, zval1, zrel2, zval2);
                }
            }

            if (c1 == ej1){
                if (ei1 == ej2 && ei2 == c2){
                    update_kernel_vector(kernel_vector, no_elements, atom,
                        vol_inv, r21, r32, r13, fi, fj,
                        fdjx1, fdjx2, fdjy1, fdjy2, fdjz1, fdjz2,
                        xrel1, xval1, xrel2, xval2, yrel1, yval1, yrel2,
                        yval2, zrel1, zval1, zrel2, zval2);
                }
                if (ei1 == c2 && ei2 == ej2){
                    update_kernel_vector(kernel_vector, no_elements, atom,
                        vol_inv, r11, r32, r23, fi, fj,
                        fdjx1, fdjx2, fdjy1, fdjy2, fdjz1, fdjz2,
                        xrel1, xval1, xrel2, xval2, yrel1, yval1, yrel2,
                        yval2, zrel1, zval1, zrel2, zval2);
                }
            }

            if (c1 == ej2){
                if (ei1 == ej1 && ei2 == c2){
                    update_kernel_vector(kernel_vector, no_elements, atom,
                        vol_inv, r31, r22, r13, fi, fj,
                        fdjx1, fdjx2, fdjy1, fdjy2, fdjz1, fdjz2,
                        xrel1, xval1, xrel2, xval2, yrel1, yval1, yrel2,
                        yval2, zrel1, zval1, zrel2, zval2);
                }
                if (ei1 == c2 && ei2 == ej1){
                    update_kernel_vector(kernel_vector, no_elements, atom,
                        vol_inv, r31, r12, r23, fi, fj,
                        fdjx1, fdjx2, fdjy1, fdjy2, fdjz1, fdjz2,
                        xrel1, xval1, xrel2, xval2, yrel1, yval1, yrel2,
                        yval2, zrel1, zval1, zrel2, zval2);
                }
            }
        }
    }

    return kernel_vector;
}

Eigen::VectorXd ThreeBodyKernel :: env_struc(const LocalEnvironment & env1,
    const StructureDescriptor & struc1){

    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);

    for (int i = 0; i < struc1.noa; i ++){
       kernel_vector += env_struc_partial(env1, struc1, i);
    }

    return kernel_vector;
}

void ThreeBodyKernel :: update_kernel_vector(Eigen::VectorXd & kernel_vector,
            int no_elements, int i, double vol_inv,
            double r11, double r22, double r33,
            double fi, double fj, double fdjx1, double fdjx2,
            double fdjy1, double fdjy2, double fdjz1, double fdjz2,
            double xrel1, double xval1, double xrel2, double xval2,
            double yrel1, double yval1, double yrel2, double yval2,
            double zrel1, double zval1, double zrel2, double zval2){

    double p1 = r11 * r11 + r22 * r22 + r33 * r33;
    double p2 = exp(-p1 * ls1);
    kernel_vector(0) += sig2 * p2 * fi * fj / 9;

    double p3 = p2 * ls2 * fi * fj;
    double p4 = p2 * fi;

    double fx1 = p3 * r11 * xrel1 + p4 * fdjx1;
    double fx2 = p3 * r22 * xrel2 + p4 * fdjx2;
    kernel_vector(1 + 3 * i) += sig2 * (fx1 + fx2) / 3;
    kernel_vector(no_elements - 6) -=
        sig2 * (fx1 * xval1 + fx2 * xval2) * vol_inv / 6;
    kernel_vector(no_elements - 5) -=
        sig2 * (fx1 * yval1 + fx2 * yval2) * vol_inv / 6;
    kernel_vector(no_elements - 4) -=
        sig2 * (fx1 * zval1 + fx2 * zval2) * vol_inv / 6;

    double fy1 = p3 * r11 * yrel1  + p4 * fdjy1;
    double fy2 = p3 * r22 * yrel2 + p4 * fdjy2;
    kernel_vector(2 + 3 * i) += sig2 * (fy1 + fy2) / 3;
    kernel_vector(no_elements - 3) -=
        sig2 * (fy1 * yval1 + fy2 * yval2) * vol_inv / 6;
    kernel_vector(no_elements - 2) -=
        sig2 * (fy1 * zval1 + fy2 * zval2) * vol_inv / 6;

    double fz1 = p3 * r11 * zrel1 + p4 * fdjz1;
    double fz2 = p3 * r22 * zrel2 + p4 * fdjz2;
    kernel_vector(3 + 3 * i) += sig2 * (fz1 + fz2) / 3;
    kernel_vector(no_elements - 1) -=
        sig2 * (fz1 * zval1 + fz2 * zval2) * vol_inv / 6;
}

DotProductKernel :: DotProductKernel() {};

DotProductKernel :: DotProductKernel(double sigma, double power,
    int descriptor_index){

    this->sigma = sigma;
    sig2 = sigma * sigma;
    this->power = power;
    this -> descriptor_index = descriptor_index;

    kernel_hyperparameters = std::vector<double> {sigma, power};
}

double DotProductKernel :: env_env(const LocalEnvironment & env1,
                                   const LocalEnvironment & env2){
    // Central species must be the same to give a nonzero kernel.
    if (env1.central_species != env2.central_species) return 0;

    double dot = env1.descriptor_vals[descriptor_index]
        .dot(env2.descriptor_vals[descriptor_index]);
    double d1 = env1.descriptor_norm[descriptor_index];
    double d2 = env2.descriptor_norm[descriptor_index];

    return sig2 * pow(dot / (d1 * d2), power);
}

Eigen::VectorXd DotProductKernel :: self_kernel_env(
    const StructureDescriptor & struc1, int atom){

    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);
    
    // TODO: implement the rest

    return kernel_vector;
}

Eigen::VectorXd DotProductKernel :: env_struc_partial(
    const LocalEnvironment & env1, const StructureDescriptor & struc1,
    int atom){

    Eigen::VectorXd kern_vec = Eigen::VectorXd::Zero(1 + struc1.noa * 3 + 6);

    // Account for edge case where d1 is zero.
    double empty_thresh = 1e-8;
    double d1 = env1.descriptor_norm[descriptor_index];
    if (d1 < empty_thresh) return kern_vec;

    double en_kern = 0;
    Eigen::VectorXd force_kern = Eigen::VectorXd::Zero(struc1.noa * 3);
    Eigen::VectorXd stress_kern = Eigen::VectorXd::Zero(6);

    Eigen::VectorXd force_dot, stress_dot, f1, s1;
    const double vol_inv = 1 / struc1.volume;
    double dot_val, d2, norm_dot, dval, d2_cubed;
    LocalEnvironment env_curr = struc1.local_environments[atom];

    // Check that the environments have the same central species.
    if (env1.central_species != env_curr.central_species){
        return kern_vec;
    };

    // Check that d2 is nonzero.
    d2 = env_curr.descriptor_norm[descriptor_index];
    if (d2 < empty_thresh){
        return kern_vec;
    };
    d2_cubed = d2 * d2 * d2;

    // Energy kernel
    dot_val = env1.descriptor_vals[descriptor_index]
        .dot(env_curr.descriptor_vals[descriptor_index]);
    norm_dot = dot_val / (d1 * d2);
    en_kern += pow(norm_dot, power);

    // Force kernel
    force_dot = env_curr.descriptor_force_dervs[descriptor_index] *
        env1.descriptor_vals[descriptor_index];
    f1 = (force_dot / (d1 * d2)) -
        (dot_val * env_curr.force_dot[descriptor_index] / (d2_cubed * d1));
    dval = power * pow(norm_dot, power - 1);
    force_kern += dval * f1;

    // Stress kernel
    stress_dot = env_curr.descriptor_stress_dervs[descriptor_index] *
        env1.descriptor_vals[descriptor_index];
    s1 = (stress_dot / (d1 * d2)) -
        (dot_val * env_curr.stress_dot[descriptor_index] /(d2_cubed * d1));
    stress_kern += dval * s1;

    kern_vec(0) = sig2 * en_kern;
    kern_vec.segment(1, struc1.noa * 3) = -sig2 * force_kern;
    kern_vec.tail(6) = -sig2 * stress_kern * vol_inv;
    return kern_vec;
}

Eigen::VectorXd DotProductKernel :: env_struc(const LocalEnvironment & env1,
    const StructureDescriptor & struc1){

    Eigen::VectorXd kern_vec = Eigen::VectorXd::Zero(1 + struc1.noa * 3 + 6);

    for (int i = 0; i < struc1.noa; i ++){
        kern_vec += env_struc_partial(env1, struc1, i);
    }

    return kern_vec;
}
