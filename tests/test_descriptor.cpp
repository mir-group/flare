#include "gtest/gtest.h"
#include "descriptor.h"
#include "single_bond.h"
#include "structure.h"
#include "local_environment.h"
#include "cutoffs.h"
#include "radial.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class DescriptorTest : public ::testing::Test{
    protected:

    double delta = 1e-8;
    int noa = 5;
    int nos = 5;

    Eigen::MatrixXd cell {3, 3}, cell_2 {3, 3}, Rx {3, 3}, Ry {3, 3},
        Rz {3, 3}, R {3, 3};
    std::vector<int> species {0, 2, 1, 3, 4};
    Eigen::MatrixXd positions_1 {noa, 3}, positions_2 {noa, 3},
        positions_3 {noa, 3};

    Structure struc1, struc2, struc3;
    LocalEnvironment env1, env2, env3;
    B1_Calculator desc1, desc2, desc3;
    B2_Calculator desc4, desc5, desc6;

    std::vector<int> descriptor_settings {5, 10, 10};

    double rcut = 3;
    std::vector<double> many_body_cutoffs {rcut};
    
    // Choose arbitrary rotation angles.
    double xrot = 1.28;
    double yrot = -3.21;
    double zrot = 0.42;
 
    // Prepare cutoff.
    std::vector<double> cutoff_hyps;
    void (*cutoff_function)(double *, double, double, std::vector<double>) =
        cos_cutoff;

    // Prepare spherical harmonics.
    int lmax = 10;
    int number_of_harmonics = (lmax + 1) * (lmax + 1);

    // Prepare radial basis set.
    double sigma = 1;
    double first_gauss = 0;
    double final_gauss = 3;
    int N = 10;
    int no_desc = N * nos;
    std::vector<double> radial_hyps = {first_gauss, final_gauss};
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>) = equispaced_gaussians;

    // Initialize matrices.
    int no_descriptors = nos * N * number_of_harmonics;
    Eigen::VectorXd single_bond_vals, single_bond_vals_2;

    Eigen::MatrixXd force_dervs, force_dervs_2, stress_dervs, stress_dervs_2;

    // Set up descriptor calculator.
    std::string radial_string = "chebyshev";
    std::string cutoff_string = "quadratic";
    int descriptor_index = 0;

    DescriptorTest(){
        // Define rotation matrices.
        Rx << 1, 0, 0,
              0, cos(xrot), -sin(xrot),
              0, sin(xrot), cos(xrot);
        Ry << cos(yrot), 0, sin(yrot),
              0, 1, 0,
              -sin(yrot), 0, cos(yrot);
        Rz << cos(zrot), -sin(zrot), 0,
              sin(zrot), cos(zrot), 0,
              0, 0, 1;
        R = Rx * Ry * Rz;

        // Create arbitrary structure and a rotated version of it.
        cell << 10, 0.52, 0.12,
               -0.93, 10, 0.32,
               0.1, -0.2, 10;
        cell_2 = cell * R.transpose();

        positions_1 << 1.2, 0.7, 2.3,
                     3.1, 2.5, 8.9,
                     -1.8, -5.8, 3.0,
                     0.2, 1.1, 2.1,
                     3.2, 1.1, 3.3;
        positions_2 = positions_1 * R.transpose();

        struc1 = Structure(cell, species, positions_1);
        struc2 = Structure(cell_2, species, positions_2);

        env1 = LocalEnvironment(struc1, 0, rcut);
        env2 = LocalEnvironment(struc2, 0, rcut);
        env1.many_body_cutoffs = env2.many_body_cutoffs = many_body_cutoffs;

        single_bond_vals = Eigen::VectorXd::Zero(no_descriptors);
        force_dervs = Eigen::MatrixXd::Zero(noa * 3, no_descriptors);
        stress_dervs = Eigen::MatrixXd::Zero(6, no_descriptors);

        // Create B1 and B2 descriptor calculators.
        desc1 = B1_Calculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings,
            descriptor_index);
        desc2 = desc3 = desc1;

        desc4 = B2_Calculator(radial_string, cutoff_string,
            radial_hyps, cutoff_hyps, descriptor_settings,
            descriptor_index);
        desc5 = desc6 = desc4;
    }

};

TEST_F(DescriptorTest, RotationTest){
    // Check that B1 is rotationally invariant.
    double d1, d2, diff;
    double tol = 1e-10;
            
    desc1.compute(env1);
    desc2.compute(env2);

    for (int n = 0; n < no_desc; n ++){
        d1 = desc1.descriptor_vals(n);
        d2 = desc2.descriptor_vals(n);
        diff = d1 - d2;
        EXPECT_LE(abs(diff), tol);
    }

    int lmax = 8;
    desc4.compute(env1);
    desc5.compute(env2);
    no_desc = desc1.descriptor_vals.rows();

    for (int n = 0; n < no_desc; n ++){
        d1 = desc4.descriptor_vals(n);
        d2 = desc5.descriptor_vals(n);
        diff = d1 - d2;
        EXPECT_LE(abs(diff), tol);
    }
}

TEST_F(DescriptorTest, SingleBond){
    // Check that B1 descriptors match the corresponding elements of the
    // single bond vector.
    double d1, d2, diff;
    double tol = 1e-10;
  
    desc1.compute(env1);
    desc2.compute(env2);

    for (int n = 0; n < no_desc; n ++){
        d1 = desc1.descriptor_vals(n);
        d2 = desc1.single_bond_vals(n);
        diff = d1 - d2;
        EXPECT_LE(abs(diff), tol);
    }
}

TEST_F(DescriptorTest, CentTest){
    double finite_diff, exact, diff;
    double tolerance = 1e-5;
        
    desc1.compute(env1);
    desc2.compute(env2);

    // Perturb the coordinates of the central atom.
    for (int m = 0; m < 3; m ++){
        positions_3 = positions_1;
        positions_3(0, m) += delta;
        struc3 = Structure(cell, species, positions_3);
        env3 = LocalEnvironment(struc3, 0, rcut);
        env3.many_body_cutoffs = many_body_cutoffs;
        desc3.compute(env3);

        // Check derivatives.
        for (int n = 0; n < no_desc; n ++){
            finite_diff = 
                (desc3.descriptor_vals(n) - desc1.descriptor_vals(n)) / delta;
            exact = desc1.descriptor_force_dervs(m, n);
            diff = abs(finite_diff - exact);
            EXPECT_LE(diff, tolerance);
        }
    }

    int lmax = 8;
    desc4.compute(env1);
    desc5.compute(env2);
    no_desc = desc4.descriptor_vals.rows();

    // Perturb the coordinates of the central atom.
    for (int m = 0; m < 3; m ++){
        positions_3 = positions_1;
        positions_3(0, m) += delta;
        struc3 = Structure(cell, species, positions_3);
        env3 = LocalEnvironment(struc3, 0, rcut);
        env3.many_body_cutoffs = many_body_cutoffs;
        desc6.compute(env3);

        // Check derivatives.
        for (int n = 0; n < no_desc; n ++){
            finite_diff = 
                (desc6.descriptor_vals(n) - desc4.descriptor_vals(n)) / delta;
            exact = desc4.descriptor_force_dervs(m, n);
            diff = abs(finite_diff - exact);
            EXPECT_LE(diff, tolerance);
        }
    }
}

TEST_F(DescriptorTest, EnvTest){
    double finite_diff, exact, diff;
    double tolerance = 1e-5;
        
    desc1.compute(env1);
    desc2.compute(env2);

    // Perturb the coordinates of the environment atoms.
    for (int p = 1; p < noa; p ++){
        for (int m = 0; m < 3; m ++){
            positions_3 = positions_1;
            positions_3(p, m) += delta;
            struc3 =  Structure(cell, species, positions_3);
            env3 = LocalEnvironment(struc3, 0, rcut);
            env3.many_body_cutoffs = many_body_cutoffs;          
            desc3.compute(env3);

            // Check derivatives.
            for (int n = 0; n < no_desc; n ++){
                finite_diff = 
                    (desc3.descriptor_vals(n) -
                     desc1.descriptor_vals(n)) / delta;
                exact = desc1.descriptor_force_dervs(p * 3 + m, n);
                diff = abs(finite_diff - exact);
                EXPECT_LE(diff, tolerance);
            }
        }
    }

    int lmax = 8;
    desc4.compute(env1);
    desc5.compute(env2);
    no_desc = desc1.descriptor_vals.rows();

    // Perturb the coordinates of the environment atoms.
    for (int p = 1; p < noa; p ++){
        for (int m = 0; m < 3; m ++){
            positions_3 = positions_1;
            positions_3(p, m) += delta;
            struc3 =  Structure(cell, species, positions_3);
            env3 = LocalEnvironment(struc3, 0, rcut);
            env3.many_body_cutoffs = many_body_cutoffs;
            desc6.compute(env3);

            // Check derivatives.
            for (int n = 0; n < no_desc; n ++){
                finite_diff = 
                    (desc6.descriptor_vals(n) -
                     desc5.descriptor_vals(n)) / delta;
                exact = desc4.descriptor_force_dervs(p * 3 + m, n);
                diff = abs(finite_diff - exact);
                EXPECT_LE(diff, tolerance);
            }
        }
    }
}

TEST_F(DescriptorTest, StressTest){
    int stress_ind = 0;
    double finite_diff, exact, diff;
    double tolerance = 1e-5;
        
    desc1.compute(env1);
    desc2.compute(env2);

    // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
    for (int m = 0; m < 3; m ++){
        for (int n = m; n < 3; n ++){
            cell_2 = cell;
            positions_2 = positions_1;

            // Perform strain.
            cell_2(0, m) += cell(0, n) * delta;
            cell_2(1, m) += cell(1, n) * delta;
            cell_2(2, m) += cell(2, n) * delta;
            for (int k = 0; k < noa; k ++){
                positions_2(k, m) += positions_1(k, n) * delta;
            }

            struc2 = Structure(cell_2, species, positions_2);
            env2 = LocalEnvironment(struc2, 0, rcut);
            env2.many_body_cutoffs = many_body_cutoffs;
            desc2.compute(env2);

            // Check stress derivatives.
            for (int p = 0; p < no_desc; p ++){
                finite_diff = 
                    (desc2.descriptor_vals(p) -
                     desc1.descriptor_vals(p)) / delta;
                exact = desc1.descriptor_stress_dervs(stress_ind, p);
                diff = abs(finite_diff - exact);
                EXPECT_LE(diff, tolerance);
            }

            stress_ind ++;
        }
    }

    int lmax = 8;
    desc4.compute(env1);
    desc5.compute(env2);
    no_desc = desc1.descriptor_vals.rows();
    stress_ind = 0;

    // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
    for (int m = 0; m < 3; m ++){
        for (int n = m; n < 3; n ++){
            cell_2 = cell;
            positions_2 = positions_1;

            // Perform strain.
            cell_2(0, m) += cell(0, n) * delta;
            cell_2(1, m) += cell(1, n) * delta;
            cell_2(2, m) += cell(2, n) * delta;
            for (int k = 0; k < noa; k ++){
                positions_2(k, m) += positions_1(k, n) * delta;
            }

            struc2 = Structure(cell_2, species, positions_2);
            env2 = LocalEnvironment(struc2, 0, rcut);
            env2.many_body_cutoffs = many_body_cutoffs;
            desc5.compute(env2);

            // Check stress derivatives.
            for (int p = 0; p < no_desc; p ++){
                finite_diff = 
                    (desc5.descriptor_vals(p) -
                     desc4.descriptor_vals(p)) / delta;
                exact = desc4.descriptor_stress_dervs(stress_ind, p);
                diff = abs(finite_diff - exact);
                EXPECT_LE(diff, tolerance);
            }

            stress_ind ++;
        }
    }
}
