#include "ace.h"
#include <cmath>
#include <iostream>

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

    return sig2 * pow(dot / (d1 * d2), power);
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

    kern_vec(0) = sig2 * en_kern;
    kern_vec.segment(1, struc1.noa * 3) = -sig2 * force_kern;
    kern_vec.tail(6) = -sig2 * stress_kern * vol_inv;
    return kern_vec;
}
