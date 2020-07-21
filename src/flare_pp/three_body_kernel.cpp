#include "three_body_kernel.h"
#include "cutoffs.h"
#include "local_environment.h"
#include <cmath>
#include <iostream>

ThreeBodyKernel ::ThreeBodyKernel(){};

ThreeBodyKernel ::ThreeBodyKernel(double sigma, double ls,
                                  const std::string &cutoff_function,
                                  std::vector<double> cutoff_hyps) {

  this->sigma = sigma;
  sig2 = sigma * sigma;

  this->ls = ls;
  ls1 = 1 / (2 * ls * ls);
  ls2 = 1 / (ls * ls);
  ls3 = ls2 * ls2;
  this->cutoff_hyps = cutoff_hyps;

  // Set kernel hyperparameters.
  Eigen::VectorXd hyps(2);
  hyps << sigma, ls;
  kernel_hyperparameters = hyps;

  if (cutoff_function == "quadratic") {
    this->cutoff_pointer = quadratic_cutoff;
  } else if (cutoff_function == "hard") {
    this->cutoff_pointer = hard_cutoff;
  } else if (cutoff_function == "cosine") {
    this->cutoff_pointer = cos_cutoff;
  }
}

double ThreeBodyKernel ::env_env(const LocalEnvironment &env1,
                                 const LocalEnvironment &env2) {
  double kern = 0;
  double ri1, ri2, ri3, rj1, rj2, rj3, fi1, fi2, fi3, fj1, fj2, fj3, fdi1, fdi2,
      fdi3, fdj1, fdj2, fdj3, fi, fj, r11, r12, r13, r21, r22, r23, r31, r32,
      r33, p1, p2, p3, p4, p5, p6;
  int i1, i2, j1, j2, ei1, ei2, ej1, ej2;

  double cut1 = env1.n_body_cutoffs[1];
  double cut2 = env2.n_body_cutoffs[1];
  double rcut_vals_i1[2], rcut_vals_i2[2], rcut_vals_i3[2], rcut_vals_j1[2],
      rcut_vals_j2[2], rcut_vals_j3[2];
  int c1 = env1.central_species;
  int c2 = env2.central_species;

  for (int m = 0; m < env1.three_body_indices.size(); m++) {
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

    for (int n = 0; n < env2.three_body_indices.size(); n++) {
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
      if (c1 == c2) {
        if (ei1 == ej1 && ei2 == ej2) {
          p1 = r11 * r11 + r22 * r22 + r33 * r33;
          kern += exp(-p1 * ls1) * fi * fj;
        }
        if (ei1 == ej2 && ei2 == ej1) {
          p2 = r12 * r12 + r21 * r21 + r33 * r33;
          kern += exp(-p2 * ls1) * fi * fj;
        }
      }

      if (c1 == ej1) {
        if (ei1 == ej2 && ei2 == c2) {
          p3 = r13 * r13 + r21 * r21 + r32 * r32;
          kern += exp(-p3 * ls1) * fi * fj;
        }
        if (ei1 == c2 && ei2 == ej2) {
          p4 = r11 * r11 + r23 * r23 + r32 * r32;
          kern += exp(-p4 * ls1) * fi * fj;
        }
      }

      if (c1 == ej2) {
        if (ei1 == ej1 && ei2 == c2) {
          p5 = r13 * r13 + r22 * r22 + r31 * r31;
          kern += exp(-p5 * ls1) * fi * fj;
        }
        if (ei1 == c2 && ei2 == ej1) {
          p6 = r12 * r12 + r23 * r23 + r31 * r31;
          kern += exp(-p6 * ls1) * fi * fj;
        }
      }
    }
  }

  return sig2 * kern / 9;
}

Eigen::VectorXd ThreeBodyKernel ::env_env_force(const LocalEnvironment &env1,
                                                const LocalEnvironment &env2) {

  // TODO: Implement.
  Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(3);
  return kernel_vector;
}

Eigen::VectorXd
ThreeBodyKernel ::self_kernel_env(const StructureDescriptor &struc1, int atom) {

  LocalEnvironment env_curr = struc1.local_environments[atom];
  int no_elements = 1 + 3 * struc1.noa + 6;
  Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(no_elements);

  double ri1, ri2, ri3, rj1, rj2, rj3, r11, r12, r13, r21, r22, r23, r31, r32,
      r33, p1, p2, p3, p4, xval_i1, xval_i2, xrel_i1, xrel_i2, yval_i1, yval_i2,
      yrel_i1, yrel_i2, zval_i1, zval_i2, zrel_i1, zrel_i2, xval_j1, xval_j2,
      xrel_j1, xrel_j2, yval_j1, yval_j2, yrel_j1, yrel_j2, zval_j1, zval_j2,
      zrel_j1, zrel_j2, fi1, fi2, fi3, fj1, fj2, fj3, fdi1, fdi2, fdj1, fdj2,
      fi, fj, fdix1, fdix2, fdiy1, fdiy2, fdiz1, fdiz2, fdjx1, fdjx2, fdjy1,
      fdjy2, fdjz1, fdjz2, fx1, fx2, fy1, fy2, fz1, fz2;

  int i1, i2, j1, j2, ei1, ei2, ej1, ej2;

  double vol_inv = 1 / struc1.volume;
  double vol_inv_sq = vol_inv * vol_inv;

  double cut = env_curr.n_body_cutoffs[1];
  double rcut_vals_i1[2], rcut_vals_i2[2], rcut_vals_i3[2], rcut_vals_j1[2],
      rcut_vals_j2[2], rcut_vals_j3[2];
  int c1 = env_curr.central_species;
  int c2 = c1;

  for (int m = 0; m < env_curr.three_body_indices.size(); m++) {
    i1 = env_curr.three_body_indices[m][0];
    i2 = env_curr.three_body_indices[m][1];

    ri1 = env_curr.rs[i1];
    ri2 = env_curr.rs[i2];
    ri3 = env_curr.cross_bond_dists[m];

    ei1 = env_curr.environment_species[i1];
    ei2 = env_curr.environment_species[i2];

    xval_i1 = env_curr.xs[i1];
    yval_i1 = env_curr.ys[i1];
    zval_i1 = env_curr.zs[i1];
    xrel_i1 = env_curr.xrel[i1];
    yrel_i1 = env_curr.yrel[i1];
    zrel_i1 = env_curr.zrel[i1];

    xval_i2 = env_curr.xs[i2];
    yval_i2 = env_curr.ys[i2];
    zval_i2 = env_curr.zs[i2];
    xrel_i2 = env_curr.xrel[i2];
    yrel_i2 = env_curr.yrel[i2];
    zrel_i2 = env_curr.zrel[i2];

    (*cutoff_pointer)(rcut_vals_i1, ri1, cut, cutoff_hyps);
    (*cutoff_pointer)(rcut_vals_i2, ri2, cut, cutoff_hyps);
    (*cutoff_pointer)(rcut_vals_i3, ri3, cut, cutoff_hyps);

    fi1 = rcut_vals_i1[0];
    fdi1 = rcut_vals_i1[1];
    fi2 = rcut_vals_i2[0];
    fdi2 = rcut_vals_i2[1];
    fi3 = rcut_vals_i3[0];
    fi = fi1 * fi2 * fi3;

    fdix1 = xrel_i1 * fdi1 * fi2 * fi3;
    fdiy1 = yrel_i1 * fdi1 * fi2 * fi3;
    fdiz1 = zrel_i1 * fdi1 * fi2 * fi3;
    fdix2 = xrel_i2 * fi1 * fdi2 * fi3;
    fdiy2 = yrel_i2 * fi1 * fdi2 * fi3;
    fdiz2 = zrel_i2 * fi1 * fdi2 * fi3;

    for (int n = 0; n < env_curr.three_body_indices.size(); n++) {
      j1 = env_curr.three_body_indices[n][0];
      j2 = env_curr.three_body_indices[n][1];

      rj1 = env_curr.rs[j1];
      rj2 = env_curr.rs[j2];
      rj3 = env_curr.cross_bond_dists[n];

      ej1 = env_curr.environment_species[j1];
      ej2 = env_curr.environment_species[j2];

      xval_j1 = env_curr.xs[j1];
      yval_j1 = env_curr.ys[j1];
      zval_j1 = env_curr.zs[j1];
      xrel_j1 = env_curr.xrel[j1];
      yrel_j1 = env_curr.yrel[j1];
      zrel_j1 = env_curr.zrel[j1];

      xval_j2 = env_curr.xs[j2];
      yval_j2 = env_curr.ys[j2];
      zval_j2 = env_curr.zs[j2];
      xrel_j2 = env_curr.xrel[j2];
      yrel_j2 = env_curr.yrel[j2];
      zrel_j2 = env_curr.zrel[j2];

      (*cutoff_pointer)(rcut_vals_j1, rj1, cut, cutoff_hyps);
      (*cutoff_pointer)(rcut_vals_j2, rj2, cut, cutoff_hyps);
      (*cutoff_pointer)(rcut_vals_j3, rj3, cut, cutoff_hyps);

      fj1 = rcut_vals_j1[0];
      fdj1 = rcut_vals_j1[1];
      fj2 = rcut_vals_j2[0];
      fdj2 = rcut_vals_j2[1];
      fj3 = rcut_vals_j3[0];
      fj = fj1 * fj2 * fj3;

      fdjx1 = xrel_j1 * fdj1 * fj2 * fj3;
      fdjy1 = yrel_j1 * fdj1 * fj2 * fj3;
      fdjz1 = zrel_j1 * fdj1 * fj2 * fj3;
      fdjx2 = xrel_j2 * fj1 * fdj2 * fj3;
      fdjy2 = yrel_j2 * fj1 * fdj2 * fj3;
      fdjz2 = zrel_j2 * fj1 * fdj2 * fj3;

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
      if (c1 == c2) {
        if (ei1 == ej1 && ei2 == ej2) {
          env_env_update_1(
              kernel_vector, no_elements, atom, vol_inv, r11, r22, r33, fi, fj,
              fdix1, fdix2, fdjx1, fdjx2, fdiy1, fdiy2, fdjy1, fdjy2, fdiz1,
              fdiz2, fdjz1, fdjz2, xrel_i1, xval_i1, xrel_i2, xval_i2, yrel_i1,
              yval_i1, yrel_i2, yval_i2, zrel_i1, zval_i1, zrel_i2, zval_i2,
              xrel_j1, xval_j1, xrel_j2, xval_j2, yrel_j1, yval_j1, yrel_j2,
              yval_j2, zrel_j1, zval_j1, zrel_j2, zval_j2);
        }
        if (ei1 == ej2 && ei2 == ej1) {
          env_env_update_1(
              kernel_vector, no_elements, atom, vol_inv, r12, r21, r33, fi, fj,
              fdix1, fdix2, fdjx2, fdjx1, fdiy1, fdiy2, fdjy2, fdjy1, fdiz1,
              fdiz2, fdjz2, fdjz1, xrel_i1, xval_i1, xrel_i2, xval_i2, yrel_i1,
              yval_i1, yrel_i2, yval_i2, zrel_i1, zval_i1, zrel_i2, zval_i2,
              xrel_j2, xval_j2, xrel_j1, xval_j1, yrel_j2, yval_j2, yrel_j1,
              yval_j1, zrel_j2, zval_j2, zrel_j1, zval_j1);
        }
      }

      if (c1 == ej1) {
        if (ei1 == ej2 && ei2 == c2) {
          env_env_update_2(
              kernel_vector, no_elements, atom, vol_inv, r21, r13, r32, fi, fj,
              fdix2, fdix1, fdjx2, fdjx1, fdiy2, fdiy1, fdjy2, fdjy1, fdiz2,
              fdiz1, fdjz2, fdjz1, xrel_i2, xval_i2, xrel_i1, xval_i1, yrel_i2,
              yval_i2, yrel_i1, yval_i1, zrel_i2, zval_i2, zrel_i1, zval_i1,
              xrel_j2, xval_j2, xrel_j1, xval_j1, yrel_j2, yval_j2, yrel_j1,
              yval_j1, zrel_j2, zval_j2, zrel_j1, zval_j1);
        }
        if (ei1 == c2 && ei2 == ej2) {
          env_env_update_2(
              kernel_vector, no_elements, atom, vol_inv, r11, r23, r32, fi, fj,
              fdix1, fdix2, fdjx2, fdjx1, fdiy1, fdiy2, fdjy2, fdjy1, fdiz1,
              fdiz2, fdjz2, fdjz1, xrel_i1, xval_i1, xrel_i2, xval_i2, yrel_i1,
              yval_i1, yrel_i2, yval_i2, zrel_i1, zval_i1, zrel_i2, zval_i2,
              xrel_j2, xval_j2, xrel_j1, xval_j1, yrel_j2, yval_j2, yrel_j1,
              yval_j1, zrel_j2, zval_j2, zrel_j1, zval_j1);
        }
      }

      if (c1 == ej2) {
        if (ei1 == ej1 && ei2 == c2) {
          env_env_update_2(
              kernel_vector, no_elements, atom, vol_inv, r22, r13, r31, fi, fj,
              fdix2, fdix1, fdjx1, fdjx2, fdiy2, fdiy1, fdjy1, fdjy2, fdiz2,
              fdiz1, fdjz1, fdjz2, xrel_i2, xval_i2, xrel_i1, xval_i1, yrel_i2,
              yval_i2, yrel_i1, yval_i1, zrel_i2, zval_i2, zrel_i1, zval_i1,
              xrel_j1, xval_j1, xrel_j2, xval_j2, yrel_j1, yval_j1, yrel_j2,
              yval_j2, zrel_j1, zval_j1, zrel_j2, zval_j2);
        }
        if (ei1 == c2 && ei2 == ej1) {
          env_env_update_2(
              kernel_vector, no_elements, atom, vol_inv, r12, r23, r31, fi, fj,
              fdix1, fdix2, fdjx1, fdjx2, fdiy1, fdiy2, fdjy1, fdjy2, fdiz1,
              fdiz2, fdjz1, fdjz2, xrel_i1, xval_i1, xrel_i2, xval_i2, yrel_i1,
              yval_i1, yrel_i2, yval_i2, zrel_i1, zval_i1, zrel_i2, zval_i2,
              xrel_j1, xval_j1, xrel_j2, xval_j2, yrel_j1, yval_j1, yrel_j2,
              yval_j2, zrel_j1, zval_j1, zrel_j2, zval_j2);
        }
      }
    }
  }

  return kernel_vector;
}

Eigen::VectorXd
ThreeBodyKernel ::self_kernel_struc(const StructureDescriptor &struc) {

  int no_elements = 1 + 3 * struc.noa + 6;
  Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(no_elements);

  double ri1, ri2, ri3, rj1, rj2, rj3, r11, r12, r13, r21, r22, r23, r31, r32,
      r33, p1, p2, p3, p4, xval_i1, xval_i2, xrel_i1, xrel_i2, yval_i1, yval_i2,
      yrel_i1, yrel_i2, zval_i1, zval_i2, zrel_i1, zrel_i2, xval_j1, xval_j2,
      xrel_j1, xrel_j2, yval_j1, yval_j2, yrel_j1, yrel_j2, zval_j1, zval_j2,
      zrel_j1, zrel_j2, fi1, fi2, fi3, fj1, fj2, fj3, fdi1, fdi2, fdj1, fdj2,
      fi, fj, fdix1, fdix2, fdiy1, fdiy2, fdiz1, fdiz2, fdjx1, fdjx2, fdjy1,
      fdjy2, fdjz1, fdjz2, fx1, fx2, fy1, fy2, fz1, fz2, cut1, cut2, mult_fac;
  ;

  int i1, i2, j1, j2, ei1, ei2, ej1, ej2, c1, c2;

  double vol_inv = 1 / struc.volume;
  double vol_inv_sq = vol_inv * vol_inv;

  double rcut_vals_i1[2], rcut_vals_i2[2], rcut_vals_i3[2], rcut_vals_j1[2],
      rcut_vals_j2[2], rcut_vals_j3[2];

  std::vector<int> inds1, inds2;

  // Double loop over environments.
  LocalEnvironment env1, env2;
  for (int i = 0; i < struc.noa; i++) {
    env1 = struc.local_environments[i];
    cut1 = env1.n_body_cutoffs[1];
    c1 = env1.central_species;
    inds1 = env1.n_body_indices[1];

    for (int j = i; j < struc.noa; j++) {
      env2 = struc.local_environments[j];
      cut2 = env2.n_body_cutoffs[1];
      c2 = env2.central_species;
      inds2 = env2.n_body_indices[1];

      if (i == j) {
        mult_fac = 1;
      } else {
        mult_fac = 2;
      }

      for (int m = 0; m < env1.three_body_indices.size(); m++) {
        i1 = env1.three_body_indices[m][0];
        i2 = env1.three_body_indices[m][1];

        ri1 = env1.rs[i1];
        ri2 = env1.rs[i2];
        ri3 = env1.cross_bond_dists[m];

        ei1 = env1.environment_species[i1];
        ei2 = env1.environment_species[i2];

        xval_i1 = env1.xs[i1];
        yval_i1 = env1.ys[i1];
        zval_i1 = env1.zs[i1];
        xrel_i1 = env1.xrel[i1];
        yrel_i1 = env1.yrel[i1];
        zrel_i1 = env1.zrel[i1];

        xval_i2 = env1.xs[i2];
        yval_i2 = env1.ys[i2];
        zval_i2 = env1.zs[i2];
        xrel_i2 = env1.xrel[i2];
        yrel_i2 = env1.yrel[i2];
        zrel_i2 = env1.zrel[i2];

        (*cutoff_pointer)(rcut_vals_i1, ri1, cut1, cutoff_hyps);
        (*cutoff_pointer)(rcut_vals_i2, ri2, cut1, cutoff_hyps);
        (*cutoff_pointer)(rcut_vals_i3, ri3, cut1, cutoff_hyps);

        fi1 = rcut_vals_i1[0];
        fdi1 = rcut_vals_i1[1];
        fi2 = rcut_vals_i2[0];
        fdi2 = rcut_vals_i2[1];
        fi3 = rcut_vals_i3[0];
        fi = fi1 * fi2 * fi3;

        fdix1 = xrel_i1 * fdi1 * fi2 * fi3;
        fdiy1 = yrel_i1 * fdi1 * fi2 * fi3;
        fdiz1 = zrel_i1 * fdi1 * fi2 * fi3;
        fdix2 = xrel_i2 * fi1 * fdi2 * fi3;
        fdiy2 = yrel_i2 * fi1 * fdi2 * fi3;
        fdiz2 = zrel_i2 * fi1 * fdi2 * fi3;

        for (int n = 0; n < env2.three_body_indices.size(); n++) {
          j1 = env2.three_body_indices[n][0];
          j2 = env2.three_body_indices[n][1];

          rj1 = env2.rs[j1];
          rj2 = env2.rs[j2];
          rj3 = env2.cross_bond_dists[n];

          ej1 = env2.environment_species[j1];
          ej2 = env2.environment_species[j2];

          xval_j1 = env2.xs[j1];
          yval_j1 = env2.ys[j1];
          zval_j1 = env2.zs[j1];
          xrel_j1 = env2.xrel[j1];
          yrel_j1 = env2.yrel[j1];
          zrel_j1 = env2.zrel[j1];

          xval_j2 = env2.xs[j2];
          yval_j2 = env2.ys[j2];
          zval_j2 = env2.zs[j2];
          xrel_j2 = env2.xrel[j2];
          yrel_j2 = env2.yrel[j2];
          zrel_j2 = env2.zrel[j2];

          (*cutoff_pointer)(rcut_vals_j1, rj1, cut2, cutoff_hyps);
          (*cutoff_pointer)(rcut_vals_j2, rj2, cut2, cutoff_hyps);
          (*cutoff_pointer)(rcut_vals_j3, rj3, cut2, cutoff_hyps);

          fj1 = rcut_vals_j1[0];
          fdj1 = rcut_vals_j1[1];
          fj2 = rcut_vals_j2[0];
          fdj2 = rcut_vals_j2[1];
          fj3 = rcut_vals_j3[0];
          fj = fj1 * fj2 * fj3;

          fdjx1 = xrel_j1 * fdj1 * fj2 * fj3;
          fdjy1 = yrel_j1 * fdj1 * fj2 * fj3;
          fdjz1 = zrel_j1 * fdj1 * fj2 * fj3;
          fdjx2 = xrel_j2 * fj1 * fdj2 * fj3;
          fdjy2 = yrel_j2 * fj1 * fdj2 * fj3;
          fdjz2 = zrel_j2 * fj1 * fdj2 * fj3;

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
          if (c1 == c2) {
            if (ei1 == ej1 && ei2 == ej2) {
              struc_struc_update_1(
                  kernel_vector, no_elements, i, j, mult_fac, vol_inv, r11, r22,
                  r33, fi, fj, fdix1, fdix2, fdjx1, fdjx2, fdiy1, fdiy2, fdjy1,
                  fdjy2, fdiz1, fdiz2, fdjz1, fdjz2, xrel_i1, xval_i1, xrel_i2,
                  xval_i2, yrel_i1, yval_i1, yrel_i2, yval_i2, zrel_i1, zval_i1,
                  zrel_i2, zval_i2, xrel_j1, xval_j1, xrel_j2, xval_j2, yrel_j1,
                  yval_j1, yrel_j2, yval_j2, zrel_j1, zval_j1, zrel_j2,
                  zval_j2);
            }
            if (ei1 == ej2 && ei2 == ej1) {
              struc_struc_update_1(
                  kernel_vector, no_elements, i, j, mult_fac, vol_inv, r12, r21,
                  r33, fi, fj, fdix1, fdix2, fdjx2, fdjx1, fdiy1, fdiy2, fdjy2,
                  fdjy1, fdiz1, fdiz2, fdjz2, fdjz1, xrel_i1, xval_i1, xrel_i2,
                  xval_i2, yrel_i1, yval_i1, yrel_i2, yval_i2, zrel_i1, zval_i1,
                  zrel_i2, zval_i2, xrel_j2, xval_j2, xrel_j1, xval_j1, yrel_j2,
                  yval_j2, yrel_j1, yval_j1, zrel_j2, zval_j2, zrel_j1,
                  zval_j1);
            }
          }

          if (c1 == ej1) {
            if (ei1 == ej2 && ei2 == c2) {
              struc_struc_update_2(
                  kernel_vector, no_elements, i, j, mult_fac, vol_inv, r21, r13,
                  r32, fi, fj, fdix2, fdix1, fdjx2, fdjx1, fdiy2, fdiy1, fdjy2,
                  fdjy1, fdiz2, fdiz1, fdjz2, fdjz1, xrel_i2, xval_i2, xrel_i1,
                  xval_i1, yrel_i2, yval_i2, yrel_i1, yval_i1, zrel_i2, zval_i2,
                  zrel_i1, zval_i1, xrel_j2, xval_j2, xrel_j1, xval_j1, yrel_j2,
                  yval_j2, yrel_j1, yval_j1, zrel_j2, zval_j2, zrel_j1,
                  zval_j1);
            }
            if (ei1 == c2 && ei2 == ej2) {
              struc_struc_update_2(
                  kernel_vector, no_elements, i, j, mult_fac, vol_inv, r11, r23,
                  r32, fi, fj, fdix1, fdix2, fdjx2, fdjx1, fdiy1, fdiy2, fdjy2,
                  fdjy1, fdiz1, fdiz2, fdjz2, fdjz1, xrel_i1, xval_i1, xrel_i2,
                  xval_i2, yrel_i1, yval_i1, yrel_i2, yval_i2, zrel_i1, zval_i1,
                  zrel_i2, zval_i2, xrel_j2, xval_j2, xrel_j1, xval_j1, yrel_j2,
                  yval_j2, yrel_j1, yval_j1, zrel_j2, zval_j2, zrel_j1,
                  zval_j1);
            }
          }

          if (c1 == ej2) {
            if (ei1 == ej1 && ei2 == c2) {
              struc_struc_update_2(
                  kernel_vector, no_elements, i, j, mult_fac, vol_inv, r22, r13,
                  r31, fi, fj, fdix2, fdix1, fdjx1, fdjx2, fdiy2, fdiy1, fdjy1,
                  fdjy2, fdiz2, fdiz1, fdjz1, fdjz2, xrel_i2, xval_i2, xrel_i1,
                  xval_i1, yrel_i2, yval_i2, yrel_i1, yval_i1, zrel_i2, zval_i2,
                  zrel_i1, zval_i1, xrel_j1, xval_j1, xrel_j2, xval_j2, yrel_j1,
                  yval_j1, yrel_j2, yval_j2, zrel_j1, zval_j1, zrel_j2,
                  zval_j2);
            }
            if (ei1 == c2 && ei2 == ej1) {
              struc_struc_update_2(
                  kernel_vector, no_elements, i, j, mult_fac, vol_inv, r12, r23,
                  r31, fi, fj, fdix1, fdix2, fdjx1, fdjx2, fdiy1, fdiy2, fdjy1,
                  fdjy2, fdiz1, fdiz2, fdjz1, fdjz2, xrel_i1, xval_i1, xrel_i2,
                  xval_i2, yrel_i1, yval_i1, yrel_i2, yval_i2, zrel_i1, zval_i1,
                  zrel_i2, zval_i2, xrel_j1, xval_j1, xrel_j2, xval_j2, yrel_j1,
                  yval_j1, yrel_j2, yval_j2, zrel_j1, zval_j1, zrel_j2,
                  zval_j2);
            }
          }
        }
      }
    }
  }

  return kernel_vector;
}

Eigen::VectorXd ThreeBodyKernel ::env_struc_partial(
    const LocalEnvironment &env1, const StructureDescriptor &struc1, int atom) {

  int no_elements = 1 + 3 * struc1.noa + 6;
  Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(no_elements);

  double ri1, ri2, ri3, rj1, rj2, rj3, fi1, fi2, fi3, fj1, fj2, fj3, fdj1, fdj2,
      fi, fj, r11, r12, r13, r21, r22, r23, r31, r32, r33, p1, p2, p3, p4,
      xval1, xval2, xrel1, xrel2, yval1, yval2, yrel1, yrel2, zval1, zval2,
      zrel1, zrel2, fdjx1, fdjx2, fdjy1, fdjy2, fdjz1, fdjz2, fx1, fx2, fy1,
      fy2, fz1, fz2;
  int i1, i2, j1, j2, ei1, ei2, ej1, ej2;

  LocalEnvironment env2 = struc1.local_environments[atom];
  double vol_inv = 1 / struc1.volume;

  double cut1 = env1.n_body_cutoffs[1];
  double cut2 = struc1.n_body_cutoffs[1];
  double rcut_vals_i1[2], rcut_vals_i2[2], rcut_vals_i3[2], rcut_vals_j1[2],
      rcut_vals_j2[2], rcut_vals_j3[2];
  int c1 = env1.central_species;
  int c2 = env2.central_species;

  for (int m = 0; m < env1.three_body_indices.size(); m++) {
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

    for (int n = 0; n < env2.cross_bond_dists.size(); n++) {
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
      if (c1 == c2) {
        if (ei1 == ej1 && ei2 == ej2) {
          env_struc_update(kernel_vector, no_elements, atom, vol_inv, r11, r22,
                           r33, fi, fj, fdjx1, fdjx2, fdjy1, fdjy2, fdjz1,
                           fdjz2, xrel1, xval1, xrel2, xval2, yrel1, yval1,
                           yrel2, yval2, zrel1, zval1, zrel2, zval2);
        }
        if (ei1 == ej2 && ei2 == ej1) {
          env_struc_update(kernel_vector, no_elements, atom, vol_inv, r21, r12,
                           r33, fi, fj, fdjx1, fdjx2, fdjy1, fdjy2, fdjz1,
                           fdjz2, xrel1, xval1, xrel2, xval2, yrel1, yval1,
                           yrel2, yval2, zrel1, zval1, zrel2, zval2);
        }
      }

      if (c1 == ej1) {
        if (ei1 == ej2 && ei2 == c2) {
          env_struc_update(kernel_vector, no_elements, atom, vol_inv, r21, r32,
                           r13, fi, fj, fdjx1, fdjx2, fdjy1, fdjy2, fdjz1,
                           fdjz2, xrel1, xval1, xrel2, xval2, yrel1, yval1,
                           yrel2, yval2, zrel1, zval1, zrel2, zval2);
        }
        if (ei1 == c2 && ei2 == ej2) {
          env_struc_update(kernel_vector, no_elements, atom, vol_inv, r11, r32,
                           r23, fi, fj, fdjx1, fdjx2, fdjy1, fdjy2, fdjz1,
                           fdjz2, xrel1, xval1, xrel2, xval2, yrel1, yval1,
                           yrel2, yval2, zrel1, zval1, zrel2, zval2);
        }
      }

      if (c1 == ej2) {
        if (ei1 == ej1 && ei2 == c2) {
          env_struc_update(kernel_vector, no_elements, atom, vol_inv, r31, r22,
                           r13, fi, fj, fdjx1, fdjx2, fdjy1, fdjy2, fdjz1,
                           fdjz2, xrel1, xval1, xrel2, xval2, yrel1, yval1,
                           yrel2, yval2, zrel1, zval1, zrel2, zval2);
        }
        if (ei1 == c2 && ei2 == ej1) {
          env_struc_update(kernel_vector, no_elements, atom, vol_inv, r31, r12,
                           r23, fi, fj, fdjx1, fdjx2, fdjy1, fdjy2, fdjz1,
                           fdjz2, xrel1, xval1, xrel2, xval2, yrel1, yval1,
                           yrel2, yval2, zrel1, zval1, zrel2, zval2);
        }
      }
    }
  }

  return kernel_vector;
}

Eigen::VectorXd ThreeBodyKernel ::env_struc(const LocalEnvironment &env1,
                                            const StructureDescriptor &struc1) {

  int no_elements = 1 + 3 * struc1.noa + 6;
  Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(no_elements);

  for (int i = 0; i < struc1.noa; i++) {
    kernel_vector += env_struc_partial(env1, struc1, i);
  }

  return kernel_vector;
}

void ThreeBodyKernel ::env_struc_update(
    Eigen::VectorXd &kernel_vector, int no_elements, int i, double vol_inv,
    double r11, double r22, double r33, double fi, double fj, double fdjx1,
    double fdjx2, double fdjy1, double fdjy2, double fdjz1, double fdjz2,
    double xrel1, double xval1, double xrel2, double xval2, double yrel1,
    double yval1, double yrel2, double yval2, double zrel1, double zval1,
    double zrel2, double zval2) {

  // energy/energy
  double p1 = r11 * r11 + r22 * r22 + r33 * r33;
  double p2 = exp(-p1 * ls1);
  kernel_vector(0) += sig2 * p2 * fi * fj / 9;

  // fx, exx, exy, exz
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

  // fy, eyy, eyz
  double fy1 = p3 * r11 * yrel1 + p4 * fdjy1;
  double fy2 = p3 * r22 * yrel2 + p4 * fdjy2;
  kernel_vector(2 + 3 * i) += sig2 * (fy1 + fy2) / 3;
  kernel_vector(no_elements - 3) -=
      sig2 * (fy1 * yval1 + fy2 * yval2) * vol_inv / 6;
  kernel_vector(no_elements - 2) -=
      sig2 * (fy1 * zval1 + fy2 * zval2) * vol_inv / 6;

  // fz, ezz
  double fz1 = p3 * r11 * zrel1 + p4 * fdjz1;
  double fz2 = p3 * r22 * zrel2 + p4 * fdjz2;
  kernel_vector(3 + 3 * i) += sig2 * (fz1 + fz2) / 3;
  kernel_vector(no_elements - 1) -=
      sig2 * (fz1 * zval1 + fz2 * zval2) * vol_inv / 6;
}

void ThreeBodyKernel ::env_env_update_1(
    Eigen::VectorXd &kernel_vector, int no_elements, int i, double vol_inv,
    double r11, double r22, double r33, double fi, double fj, double fdix1,
    double fdix2, double fdjx1, double fdjx2, double fdiy1, double fdiy2,
    double fdjy1, double fdjy2, double fdiz1, double fdiz2, double fdjz1,
    double fdjz2, double xrel_i1, double xval_i1, double xrel_i2,
    double xval_i2, double yrel_i1, double yval_i1, double yrel_i2,
    double yval_i2, double zrel_i1, double zval_i1, double zrel_i2,
    double zval_i2, double xrel_j1, double xval_j1, double xrel_j2,
    double xval_j2, double yrel_j1, double yval_j1, double yrel_j2,
    double yval_j2, double zrel_j1, double zval_j1, double zrel_j2,
    double zval_j2) {

  double vol_inv_sq = vol_inv * vol_inv;

  // energy/energy
  double diff_sq = r11 * r11 + r22 * r22 + r33 * r33;
  double k_SE = exp(-diff_sq * ls1);
  kernel_vector(0) += sig2 * k_SE * fi * fj / 9;

  double A_1, A_2, A, B1_1, B1_2, B1, B2_1, B2_2, B2, dk_dix, dk_djx,
      d2k_dix_djx, k0, k1, k2, k3;

  double A_xx, B1_xx, B2_xx, d2_xx, dk_dix_xx, dk_djx_xx, k0_xx, k1_xx, k2_xx,
      k3_xx;

  // // fx + exx, exy, exz stress component
  A_1 = xrel_i1 * xrel_j1;
  A_2 = xrel_i2 * xrel_j2;
  A = A_1 + A_2;
  A_xx = A_1 * xval_i1 * xval_j1 + A_2 * xval_i2 * xval_j2;

  B1_1 = r11 * xrel_i1;
  B1_2 = r22 * xrel_i2;
  B1 = B1_1 + B1_2;
  B1_xx = -B1_1 * xval_i1 - B1_2 * xval_i2;

  B2_1 = r11 * xrel_j1;
  B2_2 = r22 * xrel_j2;
  B2 = B2_1 + B2_2;
  B2_xx = -B2_1 * xval_j1 - B2_2 * xval_j2;

  dk_dix = k_SE * B1 * ls2;
  dk_djx = -k_SE * B2 * ls2;
  d2k_dix_djx = k_SE * (A * ls2 - B1 * B2 * ls3);

  dk_dix_xx = k_SE * B1_xx * ls2;
  dk_djx_xx = -k_SE * B2_xx * ls2;
  d2_xx = k_SE * (A_xx * ls2 - B1_xx * B2_xx * ls3);

  k0 = k_SE * (fdix1 + fdix2) * (fdjx1 + fdjx2);
  k1 = dk_dix * fi * (fdjx1 + fdjx2);
  k2 = dk_djx * (fdix1 + fdix2) * fj;
  k3 = d2k_dix_djx * fi * fj;
  double fx = sig2 * (k0 - k1 - k2 + k3);
  kernel_vector(1 + 3 * i) += fx;

  k0_xx = k_SE * (fdix1 * xval_i1 + fdix2 * xval_i2) *
          (fdjx1 * xval_j1 + fdjx2 * xval_j2);
  k1_xx = dk_dix_xx * fi * (-fdjx1 * xval_j1 - fdjx2 * xval_j2);
  k2_xx = dk_djx_xx * (-fdix1 * xval_i1 - fdix2 * xval_i2) * fj;
  k3_xx = d2_xx * fi * fj;
  kernel_vector(no_elements - 6) += sig2 * (k0_xx - k1_xx - k2_xx + k3_xx) / 4;

  // kernel_vector(no_elements - 6) +=
  //     fx * xval1 * xval2 * vol_inv_sq / 4;
  // kernel_vector(no_elements - 5) +=
  //     fx * yval1 * yval2 * vol_inv_sq / 4;
  // kernel_vector(no_elements - 4) +=
  //     fx * zval1 * zval2 * vol_inv_sq / 4;

  // fy
  A_1 = yrel_i1 * yrel_j1;
  A_2 = yrel_i2 * yrel_j2;
  A = A_1 + A_2;

  B1_1 = r11 * yrel_i1;
  B1_2 = r22 * yrel_i2;
  B1 = B1_1 + B1_2;

  B2_1 = r11 * yrel_j1;
  B2_2 = r22 * yrel_j2;
  B2 = B2_1 + B2_2;

  dk_dix = k_SE * B1 * ls2;
  dk_djx = -k_SE * B2 * ls2;
  d2k_dix_djx = k_SE * (A * ls2 - B1 * B2 * ls3);

  k0 = k_SE * (fdiy1 + fdiy2) * (fdjy1 + fdjy2);
  k1 = dk_dix * fi * (fdjy1 + fdjy2);
  k2 = dk_djx * (fdiy1 + fdiy2) * fj;
  k3 = d2k_dix_djx * fi * fj;
  double fy = sig2 * (k0 - k1 - k2 + k3);
  kernel_vector(2 + 3 * i) += fy;

  // fz
  A_1 = zrel_i1 * zrel_j1;
  A_2 = zrel_i2 * zrel_j2;
  A = A_1 + A_2;

  B1_1 = r11 * zrel_i1;
  B1_2 = r22 * zrel_i2;
  B1 = B1_1 + B1_2;

  B2_1 = r11 * zrel_j1;
  B2_2 = r22 * zrel_j2;
  B2 = B2_1 + B2_2;

  dk_dix = k_SE * B1 * ls2;
  dk_djx = -k_SE * B2 * ls2;
  d2k_dix_djx = k_SE * (A * ls2 - B1 * B2 * ls3);

  k0 = k_SE * (fdiz1 + fdiz2) * (fdjz1 + fdjz2);
  k1 = dk_dix * fi * (fdjz1 + fdjz2);
  k2 = dk_djx * (fdiz1 + fdiz2) * fj;
  k3 = d2k_dix_djx * fi * fj;
  double fz = sig2 * (k0 - k1 - k2 + k3);
  kernel_vector(3 + 3 * i) += fz;

  // // from two body version:
  // fx = force_helper(xrel1 * xrel2, rdiff * xrel1, rdiff * xrel2,
  //     c1, fi, fj, -fdi * xrel1, -fdj * xrel2, ls1, ls2, ls3,
  //     sig2);

  // kernel_vector(no_elements - 6) +=
  //     fx * xval1 * xval2 * vol_inv_sq / 4;
  // kernel_vector(no_elements - 5) +=
  //     fx * yval1 * yval2 * vol_inv_sq / 4;
  // kernel_vector(no_elements - 4) +=
  //     fx * zval1 * zval2 * vol_inv_sq / 4;

  // from env_struc partial:
  // // fx, exx, exy, exz
  // double p3 = p2 * ls2 * fi * fj;
  // double p4 = p2 * fi;

  // double fx1 = p3 * r11 * xrel1 + p4 * fdjx1;
  // double fx2 = p3 * r22 * xrel2 + p4 * fdjx2;
  // kernel_vector(1 + 3 * i) += sig2 * (fx1 + fx2) / 3;
  // kernel_vector(no_elements - 6) -=
  //     sig2 * (fx1 * xval1 + fx2 * xval2) * vol_inv / 6;
  // kernel_vector(no_elements - 5) -=
  //     sig2 * (fx1 * yval1 + fx2 * yval2) * vol_inv / 6;
  // kernel_vector(no_elements - 4) -=
  //     sig2 * (fx1 * zval1 + fx2 * zval2) * vol_inv / 6;
}

void ThreeBodyKernel ::env_env_update_2(
    Eigen::VectorXd &kernel_vector, int no_elements, int i, double vol_inv,
    double r12, double r23, double r31, double fi, double fj, double fdix1,
    double fdix2, double fdjx1, double fdjx2, double fdiy1, double fdiy2,
    double fdjy1, double fdjy2, double fdiz1, double fdiz2, double fdjz1,
    double fdjz2, double xrel_i1, double xval_i1, double xrel_i2,
    double xval_i2, double yrel_i1, double yval_i1, double yrel_i2,
    double yval_i2, double zrel_i1, double zval_i1, double zrel_i2,
    double zval_i2, double xrel_j1, double xval_j1, double xrel_j2,
    double xval_j2, double yrel_j1, double yval_j1, double yrel_j2,
    double yval_j2, double zrel_j1, double zval_j1, double zrel_j2,
    double zval_j2) {

  double vol_inv_sq = vol_inv * vol_inv;

  // energy/energy
  double diff_sq = r12 * r12 + r23 * r23 + r31 * r31;
  double k_SE = exp(-diff_sq * ls1);
  kernel_vector(0) += sig2 * k_SE * fi * fj / 9;

  double A_1, A_2, A, B1_1, B1_2, B1, B2_1, B2_2, B2, dk_dix, dk_djx,
      d2k_dix_djx, k0, k1, k2, k3;

  double A_xx, B1_xx, B2_xx, d2_xx, dk_dix_xx, dk_djx_xx, k0_xx, k1_xx, k2_xx,
      k3_xx;

  // // fx + exx, exy, exz stress component
  A = xrel_i1 * xrel_j2;
  A_xx = A * xval_i1 * xval_j2;

  B1_1 = r12 * xrel_i1;
  B1_2 = r23 * xrel_i2;
  B1 = B1_1 + B1_2;
  B1_xx = -B1_1 * xval_i1 - B1_2 * xval_i2;

  B2_1 = r12 * xrel_j2;
  B2_2 = r31 * xrel_j1;
  B2 = B2_1 + B2_2;
  B2_xx = -B2_1 * xval_j2 - B2_2 * xval_j1;

  dk_dix = k_SE * B1 * ls2;
  dk_djx = -k_SE * B2 * ls2;
  d2k_dix_djx = k_SE * (A * ls2 - B1 * B2 * ls3);

  dk_dix_xx = k_SE * B1_xx * ls2;
  dk_djx_xx = -k_SE * B2_xx * ls2;
  d2_xx = k_SE * (A_xx * ls2 - B1_xx * B2_xx * ls3);

  k0 = k_SE * (fdix1 + fdix2) * (fdjx1 + fdjx2);
  k1 = dk_dix * fi * (fdjx1 + fdjx2);
  k2 = dk_djx * (fdix1 + fdix2) * fj;
  k3 = d2k_dix_djx * fi * fj;
  double fx = sig2 * (k0 - k1 - k2 + k3);
  kernel_vector(1 + 3 * i) += fx;

  k0_xx = k_SE * (fdix1 * xval_i1 + fdix2 * xval_i2) *
          (fdjx1 * xval_j1 + fdjx2 * xval_j2);
  k1_xx = dk_dix_xx * fi * (-fdjx1 * xval_j1 - fdjx2 * xval_j2);
  k2_xx = dk_djx_xx * (-fdix1 * xval_i1 - fdix2 * xval_i2) * fj;
  k3_xx = d2_xx * fi * fj;
  kernel_vector(no_elements - 6) += sig2 * (k0_xx - k1_xx - k2_xx + k3_xx) / 4;

  // fy
  A = yrel_i1 * yrel_j2;

  B1_1 = r12 * yrel_i1;
  B1_2 = r23 * yrel_i2;
  B1 = B1_1 + B1_2;

  B2_1 = r12 * yrel_j2;
  B2_2 = r31 * yrel_j1;
  B2 = B2_1 + B2_2;

  dk_dix = k_SE * B1 * ls2;
  dk_djx = -k_SE * B2 * ls2;
  d2k_dix_djx = k_SE * (A * ls2 - B1 * B2 * ls3);

  k0 = k_SE * (fdiy1 + fdiy2) * (fdjy1 + fdjy2);
  k1 = dk_dix * fi * (fdjy1 + fdjy2);
  k2 = dk_djx * (fdiy1 + fdiy2) * fj;
  k3 = d2k_dix_djx * fi * fj;
  double fy = sig2 * (k0 - k1 - k2 + k3);
  kernel_vector(2 + 3 * i) += fy;

  // fz
  A = zrel_i1 * zrel_j2;

  B1_1 = r12 * zrel_i1;
  B1_2 = r23 * zrel_i2;
  B1 = B1_1 + B1_2;

  B2_1 = r12 * zrel_j2;
  B2_2 = r31 * zrel_j1;
  B2 = B2_1 + B2_2;

  dk_dix = k_SE * B1 * ls2;
  dk_djx = -k_SE * B2 * ls2;
  d2k_dix_djx = k_SE * (A * ls2 - B1 * B2 * ls3);

  k0 = k_SE * (fdiz1 + fdiz2) * (fdjz1 + fdjz2);
  k1 = dk_dix * fi * (fdjz1 + fdjz2);
  k2 = dk_djx * (fdiz1 + fdiz2) * fj;
  k3 = d2k_dix_djx * fi * fj;
  double fz = sig2 * (k0 - k1 - k2 + k3);
  kernel_vector(3 + 3 * i) += fz;
}

void ThreeBodyKernel ::struc_struc_update_1(
    Eigen::VectorXd &kernel_vector, int no_elements, int i, int j,
    double mult_factor, double vol_inv, double r11, double r22, double r33,
    double fi, double fj, double fdix1, double fdix2, double fdjx1,
    double fdjx2, double fdiy1, double fdiy2, double fdjy1, double fdjy2,
    double fdiz1, double fdiz2, double fdjz1, double fdjz2, double xrel_i1,
    double xval_i1, double xrel_i2, double xval_i2, double yrel_i1,
    double yval_i1, double yrel_i2, double yval_i2, double zrel_i1,
    double zval_i1, double zrel_i2, double zval_i2, double xrel_j1,
    double xval_j1, double xrel_j2, double xval_j2, double yrel_j1,
    double yval_j1, double yrel_j2, double yval_j2, double zrel_j1,
    double zval_j1, double zrel_j2, double zval_j2) {

  double vol_inv_sq = vol_inv * vol_inv;

  // energy/energy
  double diff_sq = r11 * r11 + r22 * r22 + r33 * r33;
  double k_SE = exp(-diff_sq * ls1);
  kernel_vector(0) += mult_factor * sig2 * k_SE * fi * fj / 9;

  std::vector<double> force_stress_vals;

  // fx + exx, exy, exz stress component
  force_stress_vals = force_stress_helper_1(
      mult_factor, vol_inv_sq, k_SE, r11, r22, r33, fi, fj, fdix1, fdix2, fdjx1,
      fdjx2, xrel_i1, xrel_i2, xrel_j1, xrel_j2, xval_i1, xval_i2, yval_i1,
      yval_i2, zval_i1, zval_i2, xval_j1, xval_j2, yval_j1, yval_j2, zval_j1,
      zval_j2);

  double fx = force_stress_vals[0];
  kernel_vector(no_elements - 6) += force_stress_vals[1];
  kernel_vector(no_elements - 5) += force_stress_vals[2];
  kernel_vector(no_elements - 4) += force_stress_vals[3];

  // fy + eyy, eyz
  force_stress_vals = force_stress_helper_1(
      mult_factor, vol_inv_sq, k_SE, r11, r22, r33, fi, fj, fdiy1, fdiy2, fdjy1,
      fdjy2, yrel_i1, yrel_i2, yrel_j1, yrel_j2, xval_i1, xval_i2, yval_i1,
      yval_i2, zval_i1, zval_i2, xval_j1, xval_j2, yval_j1, yval_j2, zval_j1,
      zval_j2);

  double fy = force_stress_vals[0];
  kernel_vector(no_elements - 3) += force_stress_vals[2];
  kernel_vector(no_elements - 2) += force_stress_vals[3];

  // fz + ezz
  force_stress_vals = force_stress_helper_1(
      mult_factor, vol_inv_sq, k_SE, r11, r22, r33, fi, fj, fdiz1, fdiz2, fdjz1,
      fdjz2, zrel_i1, zrel_i2, zrel_j1, zrel_j2, xval_i1, xval_i2, yval_i1,
      yval_i2, zval_i1, zval_i2, xval_j1, xval_j2, yval_j1, yval_j2, zval_j1,
      zval_j2);

  double fz = force_stress_vals[0];
  kernel_vector(no_elements - 1) += force_stress_vals[3];

  // update force kernels
  if (i == j) {
    kernel_vector(1 + 3 * i) += fx;
    kernel_vector(2 + 3 * i) += fy;
    kernel_vector(3 + 3 * i) += fz;
  }
}

std::vector<double> ThreeBodyKernel ::force_stress_helper_1(
    double mult_factor, double vol_inv_sq, double k_SE, double r11, double r22,
    double r33, double fi, double fj, double fdix1, double fdix2, double fdjx1,
    double fdjx2, double xrel_i1, double xrel_i2, double xrel_j1,
    double xrel_j2, double xval_i1, double xval_i2, double yval_i1,
    double yval_i2, double zval_i1, double zval_i2, double xval_j1,
    double xval_j2, double yval_j1, double yval_j2, double zval_j1,
    double zval_j2) {

  std::vector<double> force_stress_vals(4, 0);

  double A_1, A_2, A, B1_1, B1_2, B1, B2_1, B2_2, B2, dk_dix, dk_djx,
      d2k_dix_djx, k0, k1, k2, k3;

  double A_xx, A_xy, A_xz, B1_xx, B1_xy, B1_xz, B2_xx, B2_xy, B2_xz, d2_xx,
      dk_dix_xx, dk_djx_xx, k0_xx, k1_xx, k2_xx, k3_xx;

  A_1 = xrel_i1 * xrel_j1;
  A_2 = xrel_i2 * xrel_j2;
  A = A_1 + A_2;
  A_xx = A_1 * xval_i1 * xval_j1 + A_2 * xval_i2 * xval_j2;
  A_xy = A_1 * yval_i1 * yval_j1 + A_2 * yval_i2 * yval_j2;
  A_xz = A_1 * zval_i1 * zval_j1 + A_2 * zval_i2 * zval_j2;

  B1_1 = r11 * xrel_i1;
  B1_2 = r22 * xrel_i2;
  B1 = B1_1 + B1_2;
  B1_xx = -B1_1 * xval_i1 - B1_2 * xval_i2;
  B1_xy = -B1_1 * yval_i1 - B1_2 * yval_i2;
  B1_xz = -B1_1 * zval_i1 - B1_2 * zval_i2;

  B2_1 = r11 * xrel_j1;
  B2_2 = r22 * xrel_j2;
  B2 = B2_1 + B2_2;
  B2_xx = -B2_1 * xval_j1 - B2_2 * xval_j2;
  B2_xy = -B2_1 * yval_j1 - B2_2 * yval_j2;
  B2_xz = -B2_1 * zval_j1 - B2_2 * zval_j2;

  // fx
  dk_dix = k_SE * B1 * ls2;
  dk_djx = -k_SE * B2 * ls2;
  d2k_dix_djx = k_SE * (A * ls2 - B1 * B2 * ls3);

  k0 = k_SE * (fdix1 + fdix2) * (fdjx1 + fdjx2);
  k1 = dk_dix * fi * (fdjx1 + fdjx2);
  k2 = dk_djx * (fdix1 + fdix2) * fj;
  k3 = d2k_dix_djx * fi * fj;
  force_stress_vals[0] = sig2 * (k0 - k1 - k2 + k3);

  // exx
  dk_dix_xx = k_SE * B1_xx * ls2;
  dk_djx_xx = -k_SE * B2_xx * ls2;
  d2_xx = k_SE * (A_xx * ls2 - B1_xx * B2_xx * ls3);

  k0_xx = k_SE * (fdix1 * xval_i1 + fdix2 * xval_i2) *
          (fdjx1 * xval_j1 + fdjx2 * xval_j2);
  k1_xx = dk_dix_xx * fi * (-fdjx1 * xval_j1 - fdjx2 * xval_j2);
  k2_xx = dk_djx_xx * (-fdix1 * xval_i1 - fdix2 * xval_i2) * fj;
  k3_xx = d2_xx * fi * fj;

  force_stress_vals[1] =
      sig2 * mult_factor * vol_inv_sq * (k0_xx - k1_xx - k2_xx + k3_xx) / 4;

  // exy
  dk_dix_xx = k_SE * B1_xy * ls2;
  dk_djx_xx = -k_SE * B2_xy * ls2;
  d2_xx = k_SE * (A_xy * ls2 - B1_xy * B2_xy * ls3);

  k0_xx = k_SE * (fdix1 * yval_i1 + fdix2 * yval_i2) *
          (fdjx1 * yval_j1 + fdjx2 * yval_j2);
  k1_xx = dk_dix_xx * fi * (-fdjx1 * yval_j1 - fdjx2 * yval_j2);
  k2_xx = dk_djx_xx * (-fdix1 * yval_i1 - fdix2 * yval_i2) * fj;
  k3_xx = d2_xx * fi * fj;

  force_stress_vals[2] =
      sig2 * mult_factor * vol_inv_sq * (k0_xx - k1_xx - k2_xx + k3_xx) / 4;

  // exz
  dk_dix_xx = k_SE * B1_xz * ls2;
  dk_djx_xx = -k_SE * B2_xz * ls2;
  d2_xx = k_SE * (A_xz * ls2 - B1_xz * B2_xz * ls3);

  k0_xx = k_SE * (fdix1 * zval_i1 + fdix2 * zval_i2) *
          (fdjx1 * zval_j1 + fdjx2 * zval_j2);
  k1_xx = dk_dix_xx * fi * (-fdjx1 * zval_j1 - fdjx2 * zval_j2);
  k2_xx = dk_djx_xx * (-fdix1 * zval_i1 - fdix2 * zval_i2) * fj;
  k3_xx = d2_xx * fi * fj;

  force_stress_vals[3] =
      sig2 * mult_factor * vol_inv_sq * (k0_xx - k1_xx - k2_xx + k3_xx) / 4;

  return force_stress_vals;
}

void ThreeBodyKernel ::struc_struc_update_2(
    Eigen::VectorXd &kernel_vector, int no_elements, int i, int j,
    double mult_factor, double vol_inv, double r12, double r23, double r31,
    double fi, double fj, double fdix1, double fdix2, double fdjx1,
    double fdjx2, double fdiy1, double fdiy2, double fdjy1, double fdjy2,
    double fdiz1, double fdiz2, double fdjz1, double fdjz2, double xrel_i1,
    double xval_i1, double xrel_i2, double xval_i2, double yrel_i1,
    double yval_i1, double yrel_i2, double yval_i2, double zrel_i1,
    double zval_i1, double zrel_i2, double zval_i2, double xrel_j1,
    double xval_j1, double xrel_j2, double xval_j2, double yrel_j1,
    double yval_j1, double yrel_j2, double yval_j2, double zrel_j1,
    double zval_j1, double zrel_j2, double zval_j2) {

  double vol_inv_sq = vol_inv * vol_inv;

  // energy/energy
  double diff_sq = r12 * r12 + r23 * r23 + r31 * r31;
  double k_SE = exp(-diff_sq * ls1);
  kernel_vector(0) += mult_factor * sig2 * k_SE * fi * fj / 9;

  std::vector<double> force_stress_vals;

  // fx + exx, exy, exz stress component
  force_stress_vals = force_stress_helper_2(
      mult_factor, vol_inv_sq, k_SE, r12, r23, r31, fi, fj, fdix1, fdix2, fdjx1,
      fdjx2, xrel_i1, xrel_i2, xrel_j1, xrel_j2, xval_i1, xval_i2, yval_i1,
      yval_i2, zval_i1, zval_i2, xval_j1, xval_j2, yval_j1, yval_j2, zval_j1,
      zval_j2);

  double fx = force_stress_vals[0];
  kernel_vector(no_elements - 6) += force_stress_vals[1];
  kernel_vector(no_elements - 5) += force_stress_vals[2];
  kernel_vector(no_elements - 4) += force_stress_vals[3];

  // fy + eyy, eyz
  force_stress_vals = force_stress_helper_2(
      mult_factor, vol_inv_sq, k_SE, r12, r23, r31, fi, fj, fdiy1, fdiy2, fdjy1,
      fdjy2, yrel_i1, yrel_i2, yrel_j1, yrel_j2, xval_i1, xval_i2, yval_i1,
      yval_i2, zval_i1, zval_i2, xval_j1, xval_j2, yval_j1, yval_j2, zval_j1,
      zval_j2);

  double fy = force_stress_vals[0];
  kernel_vector(no_elements - 3) += force_stress_vals[2];
  kernel_vector(no_elements - 2) += force_stress_vals[3];

  // fz + ezz
  force_stress_vals = force_stress_helper_2(
      mult_factor, vol_inv_sq, k_SE, r12, r23, r31, fi, fj, fdiz1, fdiz2, fdjz1,
      fdjz2, zrel_i1, zrel_i2, zrel_j1, zrel_j2, xval_i1, xval_i2, yval_i1,
      yval_i2, zval_i1, zval_i2, xval_j1, xval_j2, yval_j1, yval_j2, zval_j1,
      zval_j2);

  double fz = force_stress_vals[0];
  kernel_vector(no_elements - 1) += force_stress_vals[3];

  // update force kernels
  if (i == j) {
    kernel_vector(1 + 3 * i) += fx;
    kernel_vector(2 + 3 * i) += fy;
    kernel_vector(3 + 3 * i) += fz;
  }
}

std::vector<double> ThreeBodyKernel ::force_stress_helper_2(
    double mult_factor, double vol_inv_sq, double k_SE, double r12, double r23,
    double r31, double fi, double fj, double fdix1, double fdix2, double fdjx1,
    double fdjx2, double xrel_i1, double xrel_i2, double xrel_j1,
    double xrel_j2, double xval_i1, double xval_i2, double yval_i1,
    double yval_i2, double zval_i1, double zval_i2, double xval_j1,
    double xval_j2, double yval_j1, double yval_j2, double zval_j1,
    double zval_j2) {

  std::vector<double> force_stress_vals(4, 0);

  double A_1, A_2, A, B1_1, B1_2, B1, B2_1, B2_2, B2, dk_dix, dk_djx,
      d2k_dix_djx, k0, k1, k2, k3;

  double A_xx, A_xy, A_xz, B1_xx, B1_xy, B1_xz, B2_xx, B2_xy, B2_xz, d2_xx,
      dk_dix_xx, dk_djx_xx, k0_xx, k1_xx, k2_xx, k3_xx;

  A = xrel_i1 * xrel_j2;
  A_xx = A * xval_i1 * xval_j2;
  A_xy = A * yval_i1 * yval_j2;
  A_xz = A * zval_i1 * zval_j2;

  B1_1 = r12 * xrel_i1;
  B1_2 = r23 * xrel_i2;
  B1 = B1_1 + B1_2;
  B1_xx = -B1_1 * xval_i1 - B1_2 * xval_i2;
  B1_xy = -B1_1 * yval_i1 - B1_2 * yval_i2;
  B1_xz = -B1_1 * zval_i1 - B1_2 * zval_i2;

  B2_1 = r12 * xrel_j2;
  B2_2 = r31 * xrel_j1;
  B2 = B2_1 + B2_2;
  B2_xx = -B2_1 * xval_j2 - B2_2 * xval_j1;
  B2_xy = -B2_1 * yval_j2 - B2_2 * yval_j1;
  B2_xz = -B2_1 * zval_j2 - B2_2 * zval_j1;

  // fx
  dk_dix = k_SE * B1 * ls2;
  dk_djx = -k_SE * B2 * ls2;
  d2k_dix_djx = k_SE * (A * ls2 - B1 * B2 * ls3);

  k0 = k_SE * (fdix1 + fdix2) * (fdjx1 + fdjx2);
  k1 = dk_dix * fi * (fdjx1 + fdjx2);
  k2 = dk_djx * (fdix1 + fdix2) * fj;
  k3 = d2k_dix_djx * fi * fj;
  force_stress_vals[0] = sig2 * (k0 - k1 - k2 + k3);

  // exx
  dk_dix_xx = k_SE * B1_xx * ls2;
  dk_djx_xx = -k_SE * B2_xx * ls2;
  d2_xx = k_SE * (A_xx * ls2 - B1_xx * B2_xx * ls3);

  k0_xx = k_SE * (fdix1 * xval_i1 + fdix2 * xval_i2) *
          (fdjx1 * xval_j1 + fdjx2 * xval_j2);
  k1_xx = dk_dix_xx * fi * (-fdjx1 * xval_j1 - fdjx2 * xval_j2);
  k2_xx = dk_djx_xx * (-fdix1 * xval_i1 - fdix2 * xval_i2) * fj;
  k3_xx = d2_xx * fi * fj;

  force_stress_vals[1] =
      sig2 * mult_factor * vol_inv_sq * (k0_xx - k1_xx - k2_xx + k3_xx) / 4;

  // exy
  dk_dix_xx = k_SE * B1_xy * ls2;
  dk_djx_xx = -k_SE * B2_xy * ls2;
  d2_xx = k_SE * (A_xy * ls2 - B1_xy * B2_xy * ls3);

  k0_xx = k_SE * (fdix1 * yval_i1 + fdix2 * yval_i2) *
          (fdjx1 * yval_j1 + fdjx2 * yval_j2);
  k1_xx = dk_dix_xx * fi * (-fdjx1 * yval_j1 - fdjx2 * yval_j2);
  k2_xx = dk_djx_xx * (-fdix1 * yval_i1 - fdix2 * yval_i2) * fj;
  k3_xx = d2_xx * fi * fj;

  force_stress_vals[2] =
      sig2 * mult_factor * vol_inv_sq * (k0_xx - k1_xx - k2_xx + k3_xx) / 4;

  // exz
  dk_dix_xx = k_SE * B1_xz * ls2;
  dk_djx_xx = -k_SE * B2_xz * ls2;
  d2_xx = k_SE * (A_xz * ls2 - B1_xz * B2_xz * ls3);

  k0_xx = k_SE * (fdix1 * zval_i1 + fdix2 * zval_i2) *
          (fdjx1 * zval_j1 + fdjx2 * zval_j2);
  k1_xx = dk_dix_xx * fi * (-fdjx1 * zval_j1 - fdjx2 * zval_j2);
  k2_xx = dk_djx_xx * (-fdix1 * zval_i1 - fdix2 * zval_i2) * fj;
  k3_xx = d2_xx * fi * fj;

  force_stress_vals[3] =
      sig2 * mult_factor * vol_inv_sq * (k0_xx - k1_xx - k2_xx + k3_xx) / 4;

  return force_stress_vals;
}

Eigen::MatrixXd ThreeBodyKernel ::kernel_transform(Eigen::MatrixXd kernels,
    Eigen::VectorXd new_hyps){

    // Not implemented.
    return Eigen::MatrixXd::Zero(0, 0);
    };
