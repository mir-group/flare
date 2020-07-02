#include "Structure/structure.h"
#include "Environment/local_environment.h"
#include "element_coder.h"
#include <cmath>
#include <iostream>

Structure :: Structure(){}

Structure::Structure(const Eigen::MatrixXd & cell,
    const std::vector<int> & species, const Eigen::MatrixXd & positions,
    const std::unordered_map<int, double> & mass_dict,
    const Eigen::MatrixXd & prev_positions,
    const std::vector<std::string> & species_labels){

    set_structure(cell, species, positions, mass_dict, prev_positions,
        species_labels);
}

Structure::Structure(const Eigen::MatrixXd & cell,
    const std::vector<std::string> & species,
    const Eigen::MatrixXd & positions,
    const std::unordered_map<std::string, double> & mass_dict,
    const Eigen::MatrixXd & prev_positions,
    const std::vector<std::string> & species_labels){

    // Convert species strings to integers.
    std::vector<int> coded_species;
    for (int n = 0; n < species.size(); n ++){
        coded_species.push_back(_element_to_Z[species[n]]);
    }

    // Convert mass strings to integers.
    std::unordered_map<int, double> coded_mass_dict;
    for (auto itr = mass_dict.begin(); itr!=mass_dict.end(); itr++){
        coded_mass_dict.insert({{_element_to_Z[itr->first], itr->second}});
    }

    // Set species labels.
    std::vector<std::string> label_input;
    if (species_labels.size() == 0){
        label_input = species;
    }
    else{
        label_input = species_labels;
    }

    set_structure(cell, coded_species, positions, coded_mass_dict,
        prev_positions, label_input);
}

void Structure::set_structure(const Eigen::MatrixXd & cell,
    const std::vector<int> & species,
    const Eigen::MatrixXd & positions,
    const std::unordered_map<int, double> & mass_dict,
    const Eigen::MatrixXd & prev_positions,
    const std::vector<std::string> & species_labels){

    // Set cell, species, and positions.
    set_cell(cell);
    set_positions(positions);
    this->coded_species = species;
    max_cutoff = get_max_cutoff();
    nat = species.size();

    // Set mass dictionary.
    this->mass_dict = mass_dict;

    // Set species labels.
    if (species_labels.size() == 0){
        for(int n = 0; n < nat; n ++){
            this->species_labels.push_back(std::to_string(species[n]));
        }
    }
    else{
        this->species_labels = species_labels;
    }

    // Set previous positions.
    if (prev_positions.rows() == 0){
        this->prev_positions = positions;
    }
    else{
        this->prev_positions = prev_positions;
    }

    // Initialize forces and stds to zero.
    forces = Eigen::MatrixXd::Zero(nat, 3);
    stds = Eigen::MatrixXd::Zero(nat, 3);
}

void Structure :: set_cell(const Eigen::MatrixXd & cell){
    this->cell = cell;
    cell_transpose = cell.transpose();
    cell_transpose_inverse = cell_transpose.inverse();
    cell_dot = cell * cell_transpose;
    cell_dot_inverse = cell_dot.inverse();
    volume = abs(cell.determinant());
}

const Eigen::MatrixXd & Structure :: get_cell() const{
    return cell;
}

void Structure :: set_positions(const Eigen::MatrixXd & positions){
    this->positions = positions;
    this->wrapped_positions = wrap_positions();
}

const Eigen::MatrixXd & Structure :: get_positions() const{
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
    LocalEnvironment env;

    for (int i = 0; i < nat; i ++){
        env = LocalEnvironment(*this, i, cutoff);
        local_environments.push_back(env);
    }
}

void StructureDescriptor :: compute_nested_environments(){
    LocalEnvironment env;

    for (int i = 0; i < nat; i ++){
        env = LocalEnvironment(*this, i, cutoff, n_body_cutoffs);
        local_environments.push_back(env);
    }
}

void StructureDescriptor :: compute_descriptors(){
    LocalEnvironment env;

    for (int i = 0; i < nat; i ++){
        env = LocalEnvironment(*this, i, cutoff, many_body_cutoffs,
            descriptor_calculators);
        env.compute_descriptors_and_gradients();
        local_environments.push_back(env);
    }
}

void StructureDescriptor :: nested_descriptors(){
    LocalEnvironment env;

    for (int i = 0; i < nat; i ++){
        env = LocalEnvironment(*this, i, cutoff, n_body_cutoffs,
            many_body_cutoffs, descriptor_calculators);
        env.compute_descriptors_and_gradients();
        local_environments.push_back(env);
    }
}
