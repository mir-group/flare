#include "structure.h"
#include "local_environment.h"
#include <cmath>
#include <iostream>

Structure :: Structure(){}

Structure :: Structure(const Eigen::MatrixXd & cell,
                       const std::vector<int> & species,
                       const Eigen::MatrixXd & positions){
    // Set cell, species, and positions.
    this->cell = cell;
    this->species = species;
    this->positions = positions;
    max_cutoff = get_max_cutoff();
    volume = abs(cell.determinant());
    noa = species.size();

    cell_transpose = cell.transpose();
    cell_transpose_inverse = cell_transpose.inverse();
    cell_dot = cell * cell_transpose;
    cell_dot_inverse = cell_dot.inverse();

    // Store wrapped positions.
    this->wrapped_positions = wrap_positions();

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

StructureDescriptor :: StructureDescriptor(){}

StructureDescriptor :: StructureDescriptor(const Eigen::MatrixXd & cell,
                        const std::vector<int> & species,
                        const Eigen::MatrixXd & positions,
                        double cutoff)
                    : Structure(cell, species, positions){
    this->cutoff = cutoff;
    this->compute_environments();
}

StructureDescriptor :: StructureDescriptor(const Eigen::MatrixXd & cell,
                        const std::vector<int> & species,
                        const Eigen::MatrixXd & positions,
                        DescriptorCalculator & descriptor_calculator,
                        double cutoff)
                    : Structure(cell, species, positions){

    this->descriptor_calculator = &descriptor_calculator;
    this->cutoff = cutoff;
    this->compute_descriptors();
}

void StructureDescriptor :: compute_environments(){
    int noa = species.size();
    LocalEnvironmentDescriptor env;

    for (int i = 0; i < noa; i ++){
        env = LocalEnvironmentDescriptor(*this, i, cutoff);
        environment_descriptors.push_back(env);
    }
}

void StructureDescriptor :: compute_descriptors(){
    int noa = species.size();
    LocalEnvironmentDescriptor env;

    for (int i = 0; i < noa; i ++){
        env = LocalEnvironmentDescriptor(*this, i, cutoff,
                                         descriptor_calculator);
        environment_descriptors.push_back(env);
    }
}

StructureDataset :: StructureDataset(){}

StructureDataset :: StructureDataset(const Eigen::MatrixXd & cell,
                         const std::vector<int> & species,
                         const Eigen::MatrixXd & positions,
                         DescriptorCalculator & descriptor_calculator,
                         double cutoff, std::vector<double> energy,
                         std::vector<double> force_components,
                         std::vector<double> stress_components)
                  : StructureDescriptor(cell, species, positions,
                                        descriptor_calculator, cutoff){

    this->energy = energy;
    this->force_components = force_components;
    this->stress_components = stress_components;
}
