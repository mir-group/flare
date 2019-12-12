#include "ace.h"

Structure :: Structure(const Eigen::MatrixXd & cell,
                       const std::vector<int> & species,
                       const Eigen::MatrixXd & positions){
    // Set cell, species, and positions.
    this->cell = cell;
    this->species = species;
    this->positions = positions;

    cell_transpose = cell.transpose();
    cell_transpose_inverse = cell_transpose.inverse();
    cell_dot = cell * cell_transpose;
    cell_dot_inverse = cell_dot.inverse();

    // Store wrapped positions.
    this->wrapped_positions = this->wrap_positions();

}

Eigen::MatrixXd Structure :: wrap_positions(){
    // Convert Cartesian coordinates to relative coordinates.
    Eigen::MatrixXd relative_positions =
        (positions * this->cell_transpose) * this->cell_dot_inverse;
    
    // Calculate wrapped relative coordinates by subtracting the floor.
    Eigen::MatrixXd relative_floor = relative_positions.array().floor();
    Eigen::MatrixXd relative_wrapped = 
        relative_positions - relative_floor;
    
    Eigen::MatrixXd wrapped_positions =
        (relative_wrapped * this->cell_dot) * this->cell_transpose_inverse;

    return wrapped_positions;
}
