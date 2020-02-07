#include "ace.h"
#include <cmath>
#include <iostream>

TwoBodyKernel :: TwoBodyKernel() {};

TwoBodyKernel :: TwoBodyKernel(double ls, const std::string & cutoff_function,
    std::vector<double> cutoff_hyps){

    this->ls = ls;
    ls1 = 1 / (2 * ls * ls);
    ls2 = 1 / (ls * ls);
    this->cutoff_hyps = cutoff_hyps;

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

// TODO: finish implementing two body kernel
double TwoBodyKernel :: env_env(const LocalEnvironment & env1,
                                const LocalEnvironment & env2){
    double kern = 0;
    double ri, rj, fi, fj, rdiff;

    double cut1 = env1.cutoff;
    double cut2 = env2.cutoff;
    double rcut_vals_1[2];
    double rcut_vals_2[2];

    for (int m = 0; m < env1.rs.size(); m ++){
        ri = env1.rs[m];
        (*cutoff_pointer)(rcut_vals_1, ri, cut1, cutoff_hyps);
        fi = rcut_vals_1[0];
        for (int n = 0; n < env2.rs.size(); n ++){
            rj = env2.rs[n];
            (*cutoff_pointer)(rcut_vals_2, rj, cut2, cutoff_hyps);
            fj = rcut_vals_2[0];
            rdiff = ri - rj;
            kern += fi * fj * exp(-rdiff * rdiff * ls1);
        }
    }
    return kern;
}

Eigen::VectorXd TwoBodyKernel :: env_struc(const LocalEnvironment & env1,
    const StructureDescriptor & struc1){
    
    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);
    double ri, rj, fi, fj, fdj, rdiff, c1, c2, c3, c4, c5, c6, c7, c8, fx,
        fy, fz, xval, yval, zval, xrel, yrel, zrel, en_kern;

    double cut1 = env1.cutoff;
    double cut2 = struc1.cutoff;
    double rcut_vals_1[2];
    double rcut_vals_2[2];

    double vol_inv = 1 / struc1.volume;

    LocalEnvironment env_curr;
    for (int i = 0; i < struc1.noa; i ++){
       env_curr = struc1.environment_descriptors[i];

       for (int m = 0; m < env1.rs.size(); m ++){
           ri = env1.rs[m];
           (*cutoff_pointer)(rcut_vals_1, ri, cut1, cutoff_hyps);
           fi = rcut_vals_1[0];

           for (int n = 0; n < env_curr.rs.size(); n ++){
                rj = env_curr.rs[n];
                rdiff = ri - rj;

                xval = env_curr.xs[n];
                yval = env_curr.ys[n];
                zval = env_curr.zs[n];
                xrel = env_curr.xrel[n];
                yrel = env_curr.yrel[n];
                zrel = env_curr.zrel[n];

                (*cutoff_pointer)(rcut_vals_2, rj, cut2, cutoff_hyps);
                fj = rcut_vals_2[0];
                fdj = rcut_vals_2[1];

                // energy kernel
                c1 = rdiff * rdiff;
                c2 = exp(-c1 * ls1);
                kernel_vector(0) += fi * fj * c2 / 2;

                // helper constants
                c6 = c2 * ls2 * fi * fj * rdiff;
                c7 = c2 * fi * fdj;

                // fx + exx, exy, exz stress components
                fx = xrel * c6 + c7 * xrel;
                kernel_vector(1 + 3 * i) += fx;
                kernel_vector(no_elements - 6) -= fx * xval * vol_inv / 2;
                kernel_vector(no_elements - 5) -= fx * yval * vol_inv / 2;
                kernel_vector(no_elements - 4) -= fx * zval * vol_inv / 2;
                
                // fy + eyy, eyz stress components
                fy = yrel * c6 + c7 * yrel;
                kernel_vector(2 + 3 * i) += fy;
                kernel_vector(no_elements - 3) -= fy * yval * vol_inv / 2;
                kernel_vector(no_elements - 2) -= fy * zval * vol_inv / 2;

                // fz + ezz stress component
                fz = zrel * c6 + c7 * zrel;
                kernel_vector(3 + 3 * i) += fz;
                kernel_vector(no_elements - 1) -= fz * zval * vol_inv / 2;
           }
       } 
    }

    return kernel_vector;
}

DotProductKernel :: DotProductKernel() {};

DotProductKernel :: DotProductKernel(double signal_variance, double power){
    this->signal_variance = signal_variance;
    sig2 = signal_variance * signal_variance;
    this->power = power;
}

double DotProductKernel :: env_env(const LocalEnvironmentDescriptor & env1,
                                   const LocalEnvironmentDescriptor & env2){
    // Central species must be the same to give a nonzero kernel.
    if (env1.central_species != env2.central_species) return 0;

    double dot = env1.descriptor_vals.dot(env2.descriptor_vals);
    double d1 = env1.descriptor_norm;
    double d2 = env2.descriptor_norm;

    return pow(dot / (d1 * d2), power);
}

Eigen::VectorXd DotProductKernel
    :: env_struc(const LocalEnvironmentDescriptor & env1,
                 const StructureDescriptor & struc1){

    Eigen::VectorXd kern_vec = Eigen::VectorXd::Zero(1 + struc1.noa * 3 + 6);

    // Account for edge case where d1 is zero.
    double empty_thresh = 1e-8;
    double d1 = env1.descriptor_norm;
    if (d1 < empty_thresh) return kern_vec;

    double en_kern = 0;
    Eigen::VectorXd force_kern = Eigen::VectorXd::Zero(struc1.noa * 3);
    Eigen::VectorXd stress_kern = Eigen::VectorXd::Zero(6);

    Eigen::VectorXd force_dot, stress_dot, f1, s1;
    const double vol_inv = 1 / struc1.volume;
    double dot_val, d2, norm_dot, dval, d2_cubed;
    LocalEnvironmentDescriptor env_curr;

    for (int i = 0; i < struc1.noa; i ++){
        env_curr = struc1.environment_descriptors[i];

        // Check that the environments have the same central species.
        if (env1.central_species != env_curr.central_species) continue;

        // Check that d2 is nonzero.
        d2 = env_curr.descriptor_norm;
        if (d2 < empty_thresh) continue;
        d2_cubed = d2 * d2 * d2;

        // Energy kernel
        dot_val = env1.descriptor_vals.dot(env_curr.descriptor_vals);
        norm_dot = dot_val / (d1 * d2);
        en_kern += pow(norm_dot, power);

        // Force kernel
        force_dot = env_curr.descriptor_force_dervs * env1.descriptor_vals;
        f1 = (force_dot / (d1 * d2)) -
            (dot_val * env_curr.force_dot / (d2_cubed * d1));
        dval = power * pow(norm_dot, power - 1);
        force_kern += dval * f1;

        // Stress kernel
        stress_dot = env_curr.descriptor_stress_dervs * env1.descriptor_vals;
        s1 = (stress_dot / (d1 * d2)) -
            (dot_val * env_curr.stress_dot /(d2_cubed * d1));
        stress_kern += dval * s1;
    }

    kern_vec(0) = en_kern;
    kern_vec.segment(1, struc1.noa * 3) = -force_kern;
    kern_vec.tail(6) = -stress_kern * vol_inv;
    return kern_vec;
}
