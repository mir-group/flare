#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "descriptor.h"
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

class Structure {
public:
  Eigen::VectorXi neighbor_count, cumulative_neighbor_count, structure_indices,
      neighbor_species;
  Eigen::MatrixXd cell, cell_transpose, cell_transpose_inverse, cell_dot,
      cell_dot_inverse, positions, wrapped_positions, relative_positions;
  double cutoff, single_sweep_cutoff, volume;
  int sweep, n_neighbors;
  std::vector<int> species;
  int noa;

  std::vector<Descriptor *> descriptor_calculators;
  std::vector<DescriptorValues> descriptors;

  // Make structure labels empty by default.
  Eigen::VectorXd energy, forces, stresses, mean_efs, variance_efs;
  std::vector<Eigen::VectorXd> mean_contributions, local_uncertainties;

  /**
   Default structure constructor.
   */
  Structure();

  /**
   Basic structure constructor. Holds the cell, species, and positions of a
   periodic structure of atoms.

   @param cell 3x3 array whose rows are the Bravais lattice vectors of the
        periodic cell.
   @param species List of integers denoting the chemical species of each
        atom. Must lie between 0 and s-1 (inclusive), where s is the number of
        species in the system.
   @param positions Nx3 array of atomic coordinates.
   */
  Structure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
            const Eigen::MatrixXd &positions);

  Structure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
            const Eigen::MatrixXd &positions, double cutoff,
            std::vector<Descriptor *> descriptor_calculators);

  Eigen::MatrixXd wrap_positions();
  double get_single_sweep_cutoff();
  void compute_neighbors();
  void compute_descriptors();

  // TODO: Make the Descriptor classes jsonable.
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Structure, neighbor_count,
    cutoff, cumulative_neighbor_count, structure_indices, neighbor_species,
    cell, cell_transpose, cell_transpose_inverse, cell_dot, cell_dot_inverse,
    positions, wrapped_positions, relative_positions, cutoff,
    single_sweep_cutoff, volume, sweep, n_neighbors, species, noa,
    energy, forces, stresses, mean_efs, variance_efs, mean_contributions, 
    local_uncertainties, descriptor_calculators)

  // TODO: Test these methods.
  static void to_json(std::string file_name, const Structure & struc);
  static Structure from_json(std::string file_name);
};

#endif
