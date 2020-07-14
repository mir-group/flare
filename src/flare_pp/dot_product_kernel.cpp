#include "dot_product_kernel.h"
#include "cutoffs.h"
#include "local_environment.h"
#include <cmath>
#include <iostream>

DotProductKernel :: DotProductKernel(){};

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

Eigen::VectorXd DotProductKernel :: env_env_force(
    const LocalEnvironment & env1, const LocalEnvironment & env2){

    // Note that env2 is assumed to have neighbor descriptors stored.

    Eigen::VectorXd kern_vec = Eigen::VectorXd::Zero(3);
    double d2, dot_val, norm_dot, dval, d2_cubed;
    Eigen::VectorXd force_dot, f1;
    double empty_thresh = 1e-8;
    int env_spec;

    int n_neighbors = env2.neighbor_list.size();
    double d1 = env1.descriptor_norm[descriptor_index];
    if (d1 < empty_thresh){
        return kern_vec;
    }

    // Compute force kernel (3-element vector)
    for (int n = 0; n < n_neighbors; n ++){
        env_spec = env2.structure.species[env2.neighbor_list[n]];
        if (env_spec != env1.central_species){
            continue;
        }

        d2 = env2.neighbor_descriptor_norms[n][descriptor_index];
        if (d2 < empty_thresh){
            continue;
        }
        d2_cubed = d2 * d2 * d2;
        dot_val = env1.descriptor_vals[descriptor_index]
            .dot(env2.neighbor_descriptors[n][descriptor_index]);
        norm_dot = dot_val / (d1 * d2);
        force_dot = env2.neighbor_force_dervs[n][descriptor_index] *
            env1.descriptor_vals[descriptor_index];
        f1 = (force_dot / (d1 * d2)) -
            (dot_val * env2.neighbor_force_dots[n][descriptor_index] / (d2_cubed * d1));
        dval = power * pow(norm_dot, power - 1);
        kern_vec += dval * f1;
    }

    return -sig2 * kern_vec;
}

Eigen::VectorXd DotProductKernel :: self_kernel_env(
    const StructureDescriptor & struc1, int atom){

    int no_elements = 1 + 3 * struc1.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);
    
    // TODO: implement the rest

    return kernel_vector;
}

Eigen::VectorXd DotProductKernel :: self_kernel_struc(
    const StructureDescriptor & struc){

    int no_elements = 1 + 3 * struc.noa + 6;
    Eigen::VectorXd kernel_vector =
        Eigen::VectorXd::Zero(no_elements);
    double empty_thresh = 1e-8;
    double vol_inv = 1 / struc.volume;
    double vol_inv_sq = vol_inv * vol_inv;
    
    LocalEnvironment env1, env2;
    for (int m = 0; m < struc.noa; m ++){
        env1 = struc.local_environments[m];

        // Check that d1 is nonzero.
        double d1 = env1.descriptor_norm[descriptor_index];
        if (d1 < empty_thresh){
            continue;
        }
        double d1_cubed = d1 * d1 * d1;

        for (int n = m; n < struc.noa; n ++){
            env2 = struc.local_environments[n];

            // Check that the environments have the same central species.
            if (env1.central_species != env2.central_species){
                continue;
            };

            // Check that d2 is nonzero.
            double d2 = env2.descriptor_norm[descriptor_index];
            if (d2 < empty_thresh){
                continue;
            };

            double mult_fac;
            if (m == n){
                mult_fac = 1;
            }
            else{
                mult_fac = 2;
            }

            double d2_cubed = d2 * d2 * d2;

            // Energy kernel
            double dot_val = env1.descriptor_vals[descriptor_index]
                .dot(env2.descriptor_vals[descriptor_index]);
            double norm_dot = dot_val / (d1 * d2);
            kernel_vector(0) += sig2 * mult_fac * pow(norm_dot, power);

            // Force kernel
            Eigen::MatrixXd p1_d2, d1_p2, p1_p2;

            p1_d2 = env1.descriptor_force_dervs[descriptor_index] *
                env2.descriptor_vals[descriptor_index];
            d1_p2 = env2.descriptor_force_dervs[descriptor_index] *
                env1.descriptor_vals[descriptor_index];
            p1_p2 = (env1.descriptor_force_dervs[descriptor_index].array() *
                env2.descriptor_force_dervs[descriptor_index].array())
                .rowwise().sum();

            double c1, c2;
            Eigen::MatrixXd v1, v2, v3, v4, v5, v6, force_kern;

            c1 = (power - 1) * power * pow(norm_dot, power-2);
            v1 = p1_d2 / (d1 * d2) - norm_dot * env1.force_dot[descriptor_index] / (d1 * d1);
            v2 = d1_p2 / (d1 * d2) - norm_dot * env2.force_dot[descriptor_index] / (d2 * d2);

            c2 = power * pow(norm_dot, power-1);
            v3 = p1_p2 / (d1 * d2);
            v4 = env1.force_dot[descriptor_index].array() * d1_p2.array() / (d1 * d1 * d1 * d2);
            v5 = env2.force_dot[descriptor_index].array() *  p1_d2.array() / (d1 * d2 * d2 * d2);
            v6 = env1.force_dot[descriptor_index].array() * env2.force_dot[descriptor_index].array() * norm_dot / (d1 * d1 * d2 * d2);

            force_kern = c1 * (v1.array() * v2.array()).matrix() + c2 * (v3 - v4 - v5 + v6);
            kernel_vector.segment(1, struc.noa * 3) +=
                sig2 * mult_fac * force_kern;

            // Stress kernel
            Eigen::MatrixXd p1_d2_s, d1_p2_s, p1_p2_s;

            p1_d2_s = env1.descriptor_stress_dervs[descriptor_index] *
                env2.descriptor_vals[descriptor_index];
            d1_p2_s = env2.descriptor_stress_dervs[descriptor_index] *
                env1.descriptor_vals[descriptor_index];
            p1_p2_s = (env1.descriptor_stress_dervs[descriptor_index].array() *
                env2.descriptor_stress_dervs[descriptor_index].array())
                .rowwise().sum();

            double c1_s, c2_s;
            Eigen::MatrixXd v1_s, v2_s, v3_s, v4_s, v5_s, v6_s, stress_kern;

            c1_s = (power - 1) * power * pow(norm_dot, power-2);
            v1_s = p1_d2_s / (d1 * d2) - norm_dot * env1.stress_dot[descriptor_index] / (d1 * d1);
            v2_s = d1_p2_s / (d1 * d2) - norm_dot * env2.stress_dot[descriptor_index] / (d2 * d2);

            c2_s = power * pow(norm_dot, power-1);
            v3_s = p1_p2_s / (d1 * d2);
            v4_s = env1.stress_dot[descriptor_index].array() * d1_p2_s.array() / (d1 * d1 * d1 * d2);
            v5_s = env2.stress_dot[descriptor_index].array() *  p1_d2_s.array() / (d1 * d2 * d2 * d2);
            v6_s = env1.stress_dot[descriptor_index].array() * env2.stress_dot[descriptor_index].array() * norm_dot / (d1 * d1 * d2 * d2);

            stress_kern = c1_s * (v1_s.array() * v2_s.array()).matrix() + c2_s * (v3_s - v4_s - v5_s + v6_s);
            kernel_vector.tail(6) += sig2 * mult_fac * vol_inv_sq * stress_kern;
        }
    }

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
