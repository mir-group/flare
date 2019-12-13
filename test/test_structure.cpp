#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

TEST(StructureTest, TestWrapped){
    // Check that the wrapped coordinates are equivalent to Cartesian coordinates up to lattice translations.

    Eigen::MatrixXd cell(3, 3);
    Eigen::MatrixXd positions(5, 3);

    // Create arbitrary structure.
    cell << 1.3, 0.5, 0.8,
            -1.2, 1, 0.73,
            -0.8, 0.1, 0.9;

    std::vector<int> species {1, 2, 3, 4, 5};

    positions << 1.2, 0.7, 2.3,
                 3.1, 2.5, 8.9,
                 -1.8, -5.8, 3.0,
                 0.2, 1.1, 2.1,
                 3.2, 1.1, 3.3;

    Structure test_struc = 
        Structure(cell, species, positions);
    
    // Take positions minus wrapped positions.
    Eigen::MatrixXd wrap_diff =
        test_struc.positions - test_struc.wrapped_positions;
    
    Eigen::MatrixXd wrap_rel =
        (wrap_diff * test_struc.cell_transpose) *
        test_struc.cell_dot_inverse;

    // Check that it maps to the origin under lattice translations.
    Eigen::MatrixXd check_lat =
        wrap_rel.array().round() - wrap_rel.array();

    for (int i = 0; i < 5; i ++){
        for (int j = 0; j < 3; j ++){
            EXPECT_LE(abs(check_lat(i,j)), 1e-10);
        }
    }
}
