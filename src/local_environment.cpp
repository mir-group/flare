#include "ace.h"
#include <cmath>
#include  <iostream>

LocalEnvironment :: LocalEnvironment(){}

LocalEnvironment :: LocalEnvironment(const Structure & structure, int atom,
                                     double cutoff){
    this->cutoff = cutoff;
    central_index = atom;
    central_species = structure.species[atom];
    noa = structure.wrapped_positions.rows();

    int sweep_val = ceil(cutoff / structure.max_cutoff);
    this->sweep = sweep_val;

    std::vector<int> environment_indices, environment_species,
        neighbor_list;
    std::vector<double> rs, xs, ys, zs;

    compute_environment(structure, noa, atom, cutoff, sweep_val, 
                        environment_indices, environment_species,
                        neighbor_list,
                        rs, xs, ys, zs);

    this->environment_indices = environment_indices;
    this->environment_species = environment_species;
    this->neighbor_list = neighbor_list;
    this->rs = rs;
    this->xs = xs;
    this->ys = ys;
    this->zs = zs;

}

void LocalEnvironment :: compute_environment(
    const Structure & structure,
    int noa, int atom, double cutoff, int sweep_val,
    std::vector<int> & environment_indices,
    std::vector<int> & environment_species,
    std::vector<int> & neighbor_list,
    std::vector<double> & rs, std::vector<double> & xs,
    std::vector<double> & ys, std::vector<double> & zs){

    Eigen::MatrixXd pos_atom = structure.wrapped_positions.row(atom);

    Eigen::MatrixXd vec1 = structure.cell.row(0);
    Eigen::MatrixXd vec2 = structure.cell.row(1);
    Eigen::MatrixXd vec3 = structure.cell.row(2);

    int sweep_no = (2 * sweep_val + 1) * (2 * sweep_val + 1) *
        (2 * sweep_val + 1);

    double * dists = new double[noa * sweep_no]();
    double * xvals = new double[noa * sweep_no]();
    double * yvals = new double[noa * sweep_no]();
    double * zvals = new double[noa * sweep_no]();

    int cutoff_count = 0;
    int counter = 0;

    Eigen::MatrixXd diff_curr, im;
    double dist;

    // Record the distance and position of every image in the cutoff sphere.
    for (int n = 0; n < noa; n++){
        diff_curr = structure.wrapped_positions.row(n) - pos_atom;
        for (int s1 = -sweep_val; s1 < sweep_val + 1; s1++){
            for (int s2 = -sweep_val; s2 < sweep_val + 1; s2++){
                for (int s3 = -sweep_val; s3 < sweep_val + 1; s3++){
                    im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3;
                    dist = sqrt(im(0) * im(0) + im(1) * im(1) + im(2) * im(2));
                    if ((dist < cutoff) && (dist != 0)){
                        dists[counter] = dist;
                        xvals[counter] = im(0);
                        yvals[counter] = im(1);
                        zvals[counter] = im(2);
                        cutoff_count ++;
                    }
                    counter ++;
                }
            }
        }
    }

    // Store information about atoms in the cutoff sphere.
    environment_indices.resize(cutoff_count);
    environment_species.resize(cutoff_count);
    rs.resize(cutoff_count);
    xs.resize(cutoff_count);
    ys.resize(cutoff_count);
    zs.resize(cutoff_count);
    int spec_curr, unique_check;
    double dist_curr;
    int bond_count = 0;
    counter = 0;

    for (int m = 0; m < noa; m++){
        spec_curr = structure.species[m];
        unique_check = 0;
        for (int n = 0; n < sweep_no; n++){
            dist_curr = dists[counter];
            if ((dist_curr < cutoff) && (dist_curr != 0)){
                environment_indices[bond_count] = m;
                environment_species[bond_count] = spec_curr;
                rs[bond_count] = dists[counter];
                xs[bond_count] = xvals[counter];
                ys[bond_count] = yvals[counter];
                zs[bond_count] = zvals[counter];
                bond_count ++;

                // Update neighbor list.
                if (unique_check == 0){
                    neighbor_list.push_back(m);
                    unique_check = 1;
                }
            }
            counter ++;
        }
    }

    delete [] dists; delete [] xvals; delete [] yvals; delete [] zvals;
}
