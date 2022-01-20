#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "descriptor.h"
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

class Structure {
public:
  /** @name Neighbor attributes
   */
  ///@{
  /** Neighbor count for each atom.
   */
  Eigen::VectorXi neighbor_count;
  
  /** Cumulative neighbor count. The vector has length N+1, where N is the
   *  number of atoms in the structure, with first entry 0.
   */
  Eigen::VectorXi cumulative_neighbor_count;
  
  /** Species of each neighbor. The vector has length equal to the total
   * number of neighbors in the structure.
   */
  Eigen::VectorXi neighbor_species;

  /** ID of each neighbor, where the ID is determined by the initial order
   * of atoms used when the structure was constructed.
   */
  Eigen::VectorXi structure_indices;
  ///@}

  /** @name The periodic box
   */
  ///@{
  Eigen::MatrixXd cell, cell_transpose, cell_transpose_inverse, cell_dot,
      cell_dot_inverse;
  ///@}

  /** @name Atom coordinates */
  ///@{
  Eigen::MatrixXd positions, wrapped_positions, relative_positions;
  ///@}

  double cutoff, single_sweep_cutoff, volume;
  int sweep, n_neighbors;

  /**
   * Species of each atom.
   */
  std::vector<int> species;

  /**
   * Number of atoms in the structure.
   */
  int noa;

  /** @name Descriptors */
  ///@{
  std::vector<Descriptor *> descriptor_calculators;
  std::vector<DescriptorValues> descriptors;
  ///@}

  /** @name Structure labels */
  ///@{
  Eigen::VectorXd energy, forces, stresses;
  ///@}
  
  /** @name Mean and variance predictions */
  ///@{
  Eigen::VectorXd mean_efs, variance_efs;
  std::vector<Eigen::VectorXd> mean_contributions, local_uncertainties;
  ///@}

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

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Structure, neighbor_count,
    cutoff, cumulative_neighbor_count, structure_indices, neighbor_species,
    cell, cell_transpose, cell_transpose_inverse, cell_dot, cell_dot_inverse,
    positions, wrapped_positions, relative_positions, cutoff,
    single_sweep_cutoff, volume, sweep, n_neighbors, species, noa,
    energy, forces, stresses, mean_efs, variance_efs, mean_contributions, 
    local_uncertainties, descriptor_calculators, descriptors)

  static void to_json(std::string file_name, const Structure & struc);
  static Structure from_json(std::string file_name);
};

#endif
