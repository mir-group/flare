#include "Structure/structure.h"
#include "Environment/local_environment.h"
#include <cmath>
#include <iostream>

Structure :: Structure(){}

Structure :: Structure(const Eigen::MatrixXd & cell,
                       const std::vector<int> & species,
                       const Eigen::MatrixXd & positions){

    // Set cell, species, and positions.
    set_cell(cell);
    set_positions(positions);
    this->species = species;
    max_cutoff = get_max_cutoff();
    noa = species.size();
}

void Structure :: set_cell(const Eigen::MatrixXd & cell){
    this->cell = cell;
    cell_transpose = cell.transpose();
    cell_transpose_inverse = cell_transpose.inverse();
    cell_dot = cell * cell_transpose;
    cell_dot_inverse = cell_dot.inverse();
    volume = abs(cell.determinant());
}

const Eigen::MatrixXd & Structure :: get_cell(){
    return cell;
}

void Structure :: set_positions(const Eigen::MatrixXd & positions){
    this->positions = positions;
    this->wrapped_positions = wrap_positions();
}

const Eigen::MatrixXd & Structure :: get_positions(){
    return positions;
}

const Eigen::MatrixXd & Structure::get_wrapped_positions(){
    return wrapped_positions;
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

double Structure :: get_max_cutoff(){
    Eigen::MatrixXd vec1 = cell.row(0);
    Eigen::MatrixXd vec2 = cell.row(1);
    Eigen::MatrixXd vec3 = cell.row(2);

    double max_candidates [6];

    double a_dot_b = vec1(0) * vec2(0) + vec1(1) * vec2(1) + vec1(2) * vec2(2);
    double a_dot_c = vec1(0) * vec3(0) + vec1(1) * vec3(1) + vec1(2) * vec3(2);
    double b_dot_c = vec2(0) * vec3(0) + vec2(1) * vec3(1) + vec2(2) * vec3(2);

    double a = sqrt(vec1(0) * vec1(0) + vec1(1) * vec1(1) + vec1(2) * vec1(2));
    double b = sqrt(vec2(0) * vec2(0) + vec2(1) * vec2(1) + vec2(2) * vec2(2));
    double c = sqrt(vec3(0) * vec3(0) + vec3(1) * vec3(1) + vec3(2) * vec3(2));

    max_candidates[0] = a * sqrt(1 - pow(a_dot_b / (a * b), 2));
    max_candidates[1] = b * sqrt(1 - pow(a_dot_b / (a * b), 2));
    max_candidates[2] = a * sqrt(1 - pow(a_dot_c / (a * c), 2));
    max_candidates[3] = c * sqrt(1 - pow(a_dot_c / (a * c), 2));
    max_candidates[4] = b * sqrt(1 - pow(b_dot_c / (b * c), 2));
    max_candidates[5] = c * sqrt(1 - pow(b_dot_c / (b * c), 2));

    double max_cutoff = max_candidates[0];
    for (int i = 0; i < 6; i ++){
        if (max_candidates[i] < max_cutoff){
            max_cutoff = max_candidates[i];
        }
    }

    return max_cutoff;
}

// ----------------------------------------------------------------------------
//                          structure descriptor
// ---------------------------------------------------------------------------- 

StructureDescriptor :: StructureDescriptor(){}

StructureDescriptor :: StructureDescriptor(const Eigen::MatrixXd & cell,
                        const std::vector<int> & species,
                        const Eigen::MatrixXd & positions,
                        double cutoff)
                    : Structure(cell, species, positions){
    this->cutoff = cutoff;
    this->compute_environments();
}

// n-body
StructureDescriptor :: StructureDescriptor(const Eigen::MatrixXd & cell,
    const std::vector<int> & species, const Eigen::MatrixXd & positions,
    double cutoff, std::vector<double> n_body_cutoffs)
                    : Structure(cell, species, positions){

    this->cutoff = cutoff;
    this->n_body_cutoffs = n_body_cutoffs;
    this->compute_nested_environments();
}

// many-body
StructureDescriptor :: StructureDescriptor(const Eigen::MatrixXd & cell,
                        const std::vector<int> & species,
                        const Eigen::MatrixXd & positions,
                        double cutoff,
                        std::vector<double> many_body_cutoffs,
                        std::vector<DescriptorCalculator *>
                            descriptor_calculators)
                    : Structure(cell, species, positions){

    this->descriptor_calculators = descriptor_calculators;
    this->cutoff = cutoff;
    this->many_body_cutoffs = many_body_cutoffs;
    this->compute_descriptors();
}

StructureDescriptor :: StructureDescriptor(const Eigen::MatrixXd & cell,
    const std::vector<int> & species, const Eigen::MatrixXd & positions,
    double cutoff, std::vector<double> n_body_cutoffs,
    std::vector<double> many_body_cutoffs,
    std::vector<DescriptorCalculator *> descriptor_calculators)
    : Structure(cell, species, positions){

    this->descriptor_calculators = descriptor_calculators;
    this->cutoff = cutoff;
    this->n_body_cutoffs = n_body_cutoffs;
    this->many_body_cutoffs = many_body_cutoffs;
    this->nested_descriptors();
}

void StructureDescriptor :: compute_environments(){
    // int noa = species.size();
    LocalEnvironment env;

    for (int i = 0; i < noa; i ++){
        env = LocalEnvironment(*this, i, cutoff);
        local_environments.push_back(env);
    }
}

void StructureDescriptor :: compute_nested_environments(){
    // int noa = species.size();
    LocalEnvironment env;

    for (int i = 0; i < noa; i ++){
        env = LocalEnvironment(*this, i, cutoff, n_body_cutoffs);
        local_environments.push_back(env);
    }
}

void StructureDescriptor :: compute_descriptors(){
    // int noa = species.size();
    LocalEnvironment env;

    for (int i = 0; i < noa; i ++){
        env = LocalEnvironment(*this, i, cutoff, many_body_cutoffs,
            descriptor_calculators);
        env.compute_descriptors_and_gradients();
        local_environments.push_back(env);
    }
}

void StructureDescriptor :: nested_descriptors(){
    // int noa = species.size();
    LocalEnvironment env;

    for (int i = 0; i < noa; i ++){
        env = LocalEnvironment(*this, i, cutoff, n_body_cutoffs,
            many_body_cutoffs, descriptor_calculators);
        env.compute_descriptors_and_gradients();
        local_environments.push_back(env);
    }
}
