#include "ace.h"

Structure :: Structure(const Eigen::MatrixXd & cell,
                       const std::vector<int> & species,
                       const Eigen::MatrixXd & positions){
    this->cell = cell;
    this->species = species;
    this->positions = positions;

    cell_transpose = cell.transpose();
    cell_transpose_inverse = cell_transpose.inverse();
    cell_dot = cell * cell_transpose;
    cell_dot_inverse = cell_dot.inverse();
}
