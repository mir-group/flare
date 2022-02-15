#include "test_structure.h"

TEST_F(StructureTest, TestWrapped) {
  // Check that the wrapped coordinates are equivalent to Cartesian coordinates
  // up to lattice translations.

  // Take positions minus wrapped positions.
  Eigen::MatrixXd wrap_diff =
      test_struc.positions - test_struc.wrapped_positions;

  Eigen::MatrixXd wrap_rel =
      (wrap_diff * test_struc.cell_transpose) * test_struc.cell_dot_inverse;

  // Check that it maps to the origin under lattice translations.
  Eigen::MatrixXd check_lat = wrap_rel.array().round() - wrap_rel.array();

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_LE(abs(check_lat(i, j)), 1e-10);
    }
  }
}

TEST_F(StructureTest, TestDescriptor) {
  auto start = std::chrono::steady_clock::now();
  Structure struc1 =
      Structure(cell, species, positions, cutoff, dc);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Structure construction: " << elapsed_seconds.count()
            << "s\n";
}

TEST_F(StructureTest, TestEnvironments) {
  ClusterDescriptor envs;
  envs.add_all_clusters(struc_desc);
}
