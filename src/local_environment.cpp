#include "ace.h"

LocalEnvironment :: LocalEnvironment(const Structure & structure, int atom,
                         double cutoff){
    this->cutoff = cutoff;
    central_index = atom;
    central_species = structure.species[atom];

    std::vector<int> environment_indices, environment_species;
    std::vector<double> rs, xs, ys, zs;

    compute_environment(environment_indices, environment_species,
                        rs, xs, ys, zs);

    this->environment_indices = environment_indices;
    this->environment_species = environment_species;
    
    this->rs = rs;
    this->xs = xs;
    this->ys = ys;
    this->zs = zs;

}

void LocalEnvironment :: compute_environment(
    std::vector<int> & environment_indices,
    std::vector<int> & environment_species,
    std::vector<double> & rs, std::vector<double> & xs,
    std::vector<double> & ys, std::vector<double> & zs){

    rs.push_back(3.14);
    xs.push_back(6.28);

}