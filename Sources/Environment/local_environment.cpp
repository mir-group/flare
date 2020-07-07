#include "local_environment.h"
#include <cmath>
#include  <iostream>

LocalEnvironment :: LocalEnvironment(){}

LocalEnvironment :: LocalEnvironment(const Structure & structure, int atom,
                                     double cutoff){

    set_attributes(structure, atom, cutoff);
}

void LocalEnvironment :: set_attributes(const Structure & structure, int atom,
    double cutoff){

    this->cutoff = cutoff;
    this->structure = structure;
    central_index = atom;
    central_species = structure.coded_species[atom];
    noa = structure.wrapped_positions.rows();
    structure_volume = structure.volume;

    int sweep_val = ceil(cutoff / structure.max_cutoff);
    this->sweep = sweep_val;

    std::vector<int> environment_indices, environment_species,
        neighbor_list;
    std::vector<double> rs, xs, ys, zs, xrel, yrel, zrel;
    Eigen::MatrixXd bond_array_2;

    compute_environment(structure, noa, atom, cutoff, sweep_val, 
                        environment_indices, environment_species,
                        neighbor_list,
                        rs, xs, ys, zs, xrel, yrel, zrel, bond_array_2);

    this->environment_indices = environment_indices;
    this->environment_species = environment_species;
    this->neighbor_list = neighbor_list;
    this->rs = rs;
    this->xs = xs;
    this->ys = ys;
    this->zs = zs;
    this->xrel = xrel;
    this->yrel = yrel;
    this->zrel = zrel;
    this->bond_array_2 = bond_array_2;
}

// Constructor that mirrors the Python version.
LocalEnvironment :: LocalEnvironment(const Structure & structure, int atom,
    std::unordered_map<std::string, double> cutoffs, HypsMask cutoffs_mask){
    
    // Find the maximum cutoff in the map.
    bool two_present = false, three_present = false, many_present = false;
    double twobody_cutoff, threebody_cutoff, manybody_cutoff;
    double max_cutoff = 0;
    for (auto itr = cutoffs.begin(); itr!=cutoffs.end(); itr++){
        if (itr->first == "twobody"){
            two_present = true;
            twobody_cutoff = itr->second;
            if (twobody_cutoff > max_cutoff) max_cutoff = twobody_cutoff;
        } else if (itr->first == "threebody"){
            three_present = true;
            threebody_cutoff = itr->second;
            if (threebody_cutoff > max_cutoff) max_cutoff = threebody_cutoff;
        } else if (itr->first == "manybody"){
            many_present = true;
            manybody_cutoff = itr->second;
            if (manybody_cutoff > max_cutoff) max_cutoff = manybody_cutoff;
        }
    }

    // Set the hyperparameter mask.
    this->cutoffs_mask = cutoffs_mask;

    // Set primary structure attributes.
    // TODO: Add attributes that are present in the Python version:
    // bond_array_2, bond_array_3, cross_bond_inds, cross_bond_dists, etc.
    set_attributes(structure, atom, max_cutoff);

    // Create n_body cutoffs list.
    std::vector<double> n_body_cutoffs;
    if (two_present && !three_present){
        n_body_cutoffs = {twobody_cutoff};
    } else if (!two_present && three_present){
        n_body_cutoffs = {threebody_cutoff, threebody_cutoff};
    } else if (two_present && three_present){
        n_body_cutoffs = {twobody_cutoff, threebody_cutoff};
    }

    // Compute n-body indices.
    // TODO: Make compute_indices method sensitive to the hyperparameter mask.
    this->n_body_cutoffs = n_body_cutoffs;
    compute_indices();

    // For the sake of compatibility with the Python version of the code, compute EAM attributes. This should be switched to a descriptor calculator later.

    }

// n-body
LocalEnvironment :: LocalEnvironment(const Structure & structure, int atom,
    double cutoff, std::vector<double> n_body_cutoffs)
    : LocalEnvironment(structure, atom, cutoff){

    this->n_body_cutoffs = n_body_cutoffs;
    compute_indices();
}

// many-body
LocalEnvironment :: LocalEnvironment(const Structure & structure, int atom,
    double cutoff, std::vector<double> many_body_cutoffs,
    std::vector<DescriptorCalculator *> descriptor_calculators)
    : LocalEnvironment(structure, atom, cutoff){

    this->many_body_cutoffs = many_body_cutoffs;
    this->descriptor_calculators = descriptor_calculators;
    compute_indices();
}

// n-body + many-body
LocalEnvironment :: LocalEnvironment(const Structure & structure, int atom,
    double cutoff, std::vector<double> n_body_cutoffs,
    std::vector<double> many_body_cutoffs,
    std::vector<DescriptorCalculator *> descriptor_calculators)
    : LocalEnvironment(structure, atom, cutoff){

    this->n_body_cutoffs = n_body_cutoffs;
    this->many_body_cutoffs = many_body_cutoffs;
    this->descriptor_calculators = descriptor_calculators;
    compute_indices();
}

void LocalEnvironment :: compute_environment(
    const Structure & structure,
    int noa, int atom, double cutoff, int sweep_val,
    std::vector<int> & environment_indices,
    std::vector<int> & environment_species,
    std::vector<int> & neighbor_list,
    std::vector<double> & rs, std::vector<double> & xs,
    std::vector<double> & ys, std::vector<double> & zs,
    std::vector<double> & xrel, std::vector<double> & yrel,
    std::vector<double> & zrel,
    Eigen::MatrixXd & bond_array_2){

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
    xrel.resize(cutoff_count);
    yrel.resize(cutoff_count);
    zrel.resize(cutoff_count);
    bond_array_2.conservativeResize(cutoff_count, 4);
    int spec_curr, unique_check;
    double dist_curr, xcurr, ycurr, zcurr, xr, yr, zr;
    int bond_count = 0;
    counter = 0;

    for (int m = 0; m < noa; m++){
        spec_curr = structure.coded_species[m];

        // Add central atom to the neighbor list.
        unique_check = 0;
        if (m == atom){
            neighbor_list.push_back(m);
            unique_check = 1;
        }

        for (int n = 0; n < sweep_no; n++){
            dist_curr = dists[counter];
            if ((dist_curr < cutoff) && (dist_curr != 0)){
                environment_indices[bond_count] = m;
                environment_species[bond_count] = spec_curr;

                xcurr = xvals[counter];
                ycurr = yvals[counter];
                zcurr = zvals[counter];

                xr = xcurr / dist_curr;
                yr = ycurr / dist_curr;
                zr = zcurr / dist_curr;

                rs[bond_count] = dist_curr;
                xs[bond_count] = xcurr;
                ys[bond_count] = ycurr;
                zs[bond_count] = zcurr;
                xrel[bond_count] = xr;
                yrel[bond_count] = yr;
                zrel[bond_count] = zr;

                bond_array_2(bond_count, 0) = dist_curr;
                bond_array_2(bond_count, 1) = xr;
                bond_array_2(bond_count, 2) = yr;
                bond_array_2(bond_count, 3) = zr;

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

void LocalEnvironment :: compute_indices(){

    int no_atoms = rs.size();

    // Initialize a list of lists storing atom indices.
    int n_cutoffs = n_body_cutoffs.size();
    std::vector<int> empty;
    for (int i = 0; i < n_cutoffs; i ++){
        n_body_indices.push_back(empty);
    }

    int n_mb_cutoffs = many_body_cutoffs.size();
    for (int i = 0; i < n_mb_cutoffs; i ++){
        many_body_indices.push_back(empty);
    }

    // Store indices of atoms inside the 2-, 3-, and many-body cutoff spheres.
    double current_cutoff;
    for (int i = 0; i < no_atoms; i ++){
        double r_curr = rs[i];
        
        // Store n-body indices.
        for (int j = 0; j < n_cutoffs; j ++){
            current_cutoff = n_body_cutoffs[j];
            if (r_curr < current_cutoff){
                n_body_indices[j].push_back(i);
            }
        }

        // Store many-body indices.
        for (int j = 0; j < n_mb_cutoffs; j ++){
            current_cutoff = many_body_cutoffs[j];
            if (r_curr < current_cutoff){
                many_body_indices[j].push_back(i);
            }
        }
    }

    // Store triplets if the 3-body cutoff is nonzero.
    if (n_cutoffs > 1){
        double three_body_cutoff = n_body_cutoffs[1];
        double cross_bond_dist, x1, y1, z1, x2, y2, z2, x_diff, y_diff, z_diff;
        int ind1, ind2;
        std::vector<int> triplet = std::vector<int> {0, 0};
        for (int i = 0; i < n_body_indices[1].size(); i ++){
            ind1 = n_body_indices[1][i];
            x1 = xs[ind1];
            y1 = ys[ind1];
            z1 = zs[ind1];
            for (int j = i + 1; j < n_body_indices[1].size(); j ++){
                ind2 = n_body_indices[1][j];
                x_diff = x1 - xs[ind2];
                y_diff = y1 - ys[ind2];
                z_diff = z1 - zs[ind2];
                cross_bond_dist = 
                    sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
                if (cross_bond_dist <= three_body_cutoff){
                    cross_bond_dists.push_back(cross_bond_dist);
                    triplet[0] = ind1;
                    triplet[1] = ind2;
                    three_body_indices.push_back(triplet);
                }
            }
        }
    }
}

void LocalEnvironment :: compute_descriptors_and_gradients(){
    int n_calculators = descriptor_calculators.size();
    for (int i = 0; i < n_calculators; i ++){
        descriptor_calculators[i]->compute(*this);
        descriptor_vals.push_back(descriptor_calculators[i]->descriptor_vals);
        descriptor_force_dervs.push_back(
            descriptor_calculators[i]->descriptor_force_dervs);
        descriptor_stress_dervs.push_back(
            descriptor_calculators[i]->descriptor_stress_dervs);

        descriptor_norm.push_back(sqrt(
            descriptor_vals[i].dot(descriptor_vals[i])));
        force_dot.push_back(descriptor_force_dervs[i] * descriptor_vals[i]);
        stress_dot.push_back(descriptor_stress_dervs[i] * descriptor_vals[i]);

        // Clear descriptor calculator matrices to save memory.
        descriptor_calculators[i]->destroy_matrices();
    }
}

void LocalEnvironment :: compute_descriptors(){

    int n_calculators = descriptor_calculators.size();
    for (int i = 0; i < n_calculators; i ++){
        descriptor_calculators[i]->compute(*this);
        descriptor_vals.push_back(descriptor_calculators[i]->descriptor_vals);
        descriptor_norm.push_back(sqrt(
            descriptor_vals[i].dot(descriptor_vals[i])));

        // Clear descriptor calculator matrices to save memory.
        descriptor_calculators[i]->destroy_matrices();
    }
}

void LocalEnvironment :: compute_descriptor_squared(){
    // Assumes descriptors have already been computed.
    int n_calculators = descriptor_calculators.size();

    for (int i = 0; i < n_calculators; i ++){
        int desc_size = descriptor_vals[i].size();
        double desc_norm = descriptor_norm[i];
        Eigen::VectorXd desc_sq =
            Eigen::VectorXd::Zero(desc_size * (desc_size + 1) / 2);
        int desc_count = 0;

        for (int j = 0; j < desc_size; j++){
            double desc_norm_j = descriptor_vals[i](j) / desc_norm;

            for (int k = j; k < desc_size; k ++){
                double desc_norm_k = descriptor_vals[i](k) / desc_norm;
                desc_sq(desc_count) = desc_norm_j * desc_norm_k;
                desc_count ++;
            }
        }
        descriptor_squared.push_back(desc_sq);
    }
}

void LocalEnvironment :: compute_neighbor_descriptors(){

    int n_neighbors = neighbor_list.size();
    int n_descriptors = descriptor_calculators.size();
    int neighbor;
    LocalEnvironment env_curr;

    for (int m = 0; m < n_neighbors; m ++){
        neighbor = neighbor_list[m];
        env_curr = LocalEnvironment(structure, neighbor, cutoff,
            many_body_cutoffs, descriptor_calculators);
        env_curr.compute_descriptors_and_gradients();

        // Add neighbor descriptors and norms.
        neighbor_descriptors.push_back(env_curr.descriptor_vals);
        neighbor_descriptor_norms.push_back(env_curr.descriptor_norm);

        // Add neighbor derivatives and force/descriptor dot products.
        std::vector<Eigen::MatrixXd> derivs, dots;
        for (int n = 0; n < n_descriptors; n ++){
            int n_descriptors = env_curr.descriptor_vals[n].size();
            derivs.push_back(
                env_curr.descriptor_force_dervs[n].block(
                    3 * central_index, 0, 3, n_descriptors));
            dots.push_back(env_curr.force_dot[n].block(
                3 * central_index, 0, 3, 1));
        }
        neighbor_force_dervs.push_back(derivs);
        neighbor_force_dots.push_back(dots);
    }
}

void compute_neighbor_descriptors(std::vector<LocalEnvironment> & envs){
    int n_envs = envs.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_envs; i ++){
        envs[i].compute_neighbor_descriptors();
    }
}

void compute_descriptors(std::vector<LocalEnvironment> & envs){
    int n_envs = envs.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_envs; i ++){
        envs[i].compute_descriptors();
    }
}
