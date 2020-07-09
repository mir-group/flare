#include "local_environment.h"
#include <cmath>
#include <iostream>

LocalEnvironment ::LocalEnvironment() {}

LocalEnvironment ::LocalEnvironment(const Structure &structure, int atom,
                                    double cutoff) {

  set_attributes(structure, atom, cutoff);
}

void LocalEnvironment ::set_attributes(const Structure &structure, int atom,
                                       double cutoff) {

  this->cutoff = cutoff;
  this->structure = structure;
  central_index = atom;
  central_species = structure.coded_species[atom];
  noa = structure.wrapped_positions.rows();
  structure_volume = structure.volume;

  int sweep_val = ceil(cutoff / structure.max_cutoff);
  this->sweep = sweep_val;

  std::vector<int> environment_indices, environment_species, neighbor_list;
  std::vector<double> rs, xs, ys, zs, xrel, yrel, zrel;
  Eigen::MatrixXd bond_array_2;

  compute_environment();
}

// Constructor that mirrors the Python version.
LocalEnvironment ::LocalEnvironment(
    const Structure &structure, int atom,
    std::unordered_map<std::string, double> cutoffs, HypsMask cutoffs_mask) {

  // Find the maximum cutoff in the map.
  bool two_present = false, three_present = false, many_present = false;
  double twobody_cutoff, threebody_cutoff, manybody_cutoff;
  double max_cutoff = 0;
  for (auto itr = cutoffs.begin(); itr != cutoffs.end(); itr++) {
    if (itr->first == "twobody") {
      two_present = true;
      twobody_cutoff = itr->second;
      if (twobody_cutoff > max_cutoff)
        max_cutoff = twobody_cutoff;
    } else if (itr->first == "threebody") {
      three_present = true;
      threebody_cutoff = itr->second;
      if (threebody_cutoff > max_cutoff)
        max_cutoff = threebody_cutoff;
    } else if (itr->first == "manybody") {
      many_present = true;
      manybody_cutoff = itr->second;
      if (manybody_cutoff > max_cutoff)
        max_cutoff = manybody_cutoff;
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
  if (two_present && !three_present) {
    n_body_cutoffs = {twobody_cutoff};
  } else if (!two_present && three_present) {
    n_body_cutoffs = {threebody_cutoff, threebody_cutoff};
  } else if (two_present && three_present) {
    n_body_cutoffs = {twobody_cutoff, threebody_cutoff};
  }

  // Compute n-body indices.
  // TODO: Make compute_indices method sensitive to the hyperparameter mask.
  this->n_body_cutoffs = n_body_cutoffs;
  compute_indices();

  // For the sake of compatibility with the Python version of the code, compute
  // EAM attributes. This should be switched to a descriptor calculator later.
}

// n-body
LocalEnvironment ::LocalEnvironment(const Structure &structure, int atom,
                                    double cutoff,
                                    std::vector<double> n_body_cutoffs)
    : LocalEnvironment(structure, atom, cutoff) {

  this->n_body_cutoffs = n_body_cutoffs;
  compute_indices();
}

// many-body
LocalEnvironment ::LocalEnvironment(
    const Structure &structure, int atom, double cutoff,
    std::vector<double> many_body_cutoffs,
    std::vector<DescriptorCalculator *> descriptor_calculators)
    : LocalEnvironment(structure, atom, cutoff) {

  this->many_body_cutoffs = many_body_cutoffs;
  this->descriptor_calculators = descriptor_calculators;
  compute_indices();
}

// n-body + many-body
LocalEnvironment ::LocalEnvironment(
    const Structure &structure, int atom, double cutoff,
    std::vector<double> n_body_cutoffs, std::vector<double> many_body_cutoffs,
    std::vector<DescriptorCalculator *> descriptor_calculators)
    : LocalEnvironment(structure, atom, cutoff) {

  this->n_body_cutoffs = n_body_cutoffs;
  this->many_body_cutoffs = many_body_cutoffs;
  this->descriptor_calculators = descriptor_calculators;
  compute_indices();
}

void LocalEnvironment ::compute_environment() {

  Eigen::MatrixXd pos_atom = structure.wrapped_positions.row(central_index);

  Eigen::MatrixXd vec1 = structure.cell.row(0);
  Eigen::MatrixXd vec2 = structure.cell.row(1);
  Eigen::MatrixXd vec3 = structure.cell.row(2);

  int sweep_no =
      (2 * sweep + 1) * (2 * sweep + 1) * (2 * sweep + 1);

  double *dists = new double[noa * sweep_no]();
  double *xvals = new double[noa * sweep_no]();
  double *yvals = new double[noa * sweep_no]();
  double *zvals = new double[noa * sweep_no]();

  int cutoff_count = 0;
  int counter = 0;

  Eigen::MatrixXd diff_curr, im;
  double dist;

  // Record the distance and position of every image in the cutoff sphere.
  for (int n = 0; n < noa; n++) {
    diff_curr = structure.wrapped_positions.row(n) - pos_atom;
    for (int s1 = -sweep; s1 < sweep + 1; s1++) {
      for (int s2 = -sweep; s2 < sweep + 1; s2++) {
        for (int s3 = -sweep; s3 < sweep + 1; s3++) {
          im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3;
          dist = sqrt(im(0) * im(0) + im(1) * im(1) + im(2) * im(2));
          if ((dist < cutoff) && (dist != 0)) {
            dists[counter] = dist;
            xvals[counter] = im(0);
            yvals[counter] = im(1);
            zvals[counter] = im(2);
            cutoff_count++;
          }
          counter++;
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
  neighbor_list.resize(0);
  etypes.resize(0);
  bond_inds.resize(0);
  bond_array_2.conservativeResize(cutoff_count, 4);
  int spec_curr, unique_check;
  double dist_curr, xcurr, ycurr, zcurr, xr, yr, zr;
  int bond_count = 0;
  counter = 0;

  // Check if separate cutoffs are given for the 2-body sphere.
  bool sepcut = false;
  int bc, bcn, bn;
  double sepcut_value = cutoff;
  if (cutoffs_mask.nspecie > 1
      && cutoffs_mask.twobody_mask.size() > 1
      && cutoffs_mask.twobody_cutoff_list.size() > 1
      && cutoffs_mask.specie_mask.size() > 1){
          sepcut = true;
          bc = cutoffs_mask.specie_mask[central_species];
          bcn = cutoffs_mask.nspecie * bc;
      }

  for (int m = 0; m < noa; m++) {
    spec_curr = structure.coded_species[m];

    // Add central atom to the neighbor list.
    unique_check = 0;
    if (m == central_index) {
      neighbor_list.push_back(m);
      unique_check = 1;
    }

    // If separate cutoffs are given, find the cutoff corresponding to the
    // current atom.
    if (sepcut){
        bn = cutoffs_mask.specie_mask[spec_curr];
        sepcut_value = cutoffs_mask.twobody_cutoff_list[
            cutoffs_mask.twobody_mask[bcn + bn]];
    }

    for (int n = 0; n < sweep_no; n++) {
      dist_curr = dists[counter];
      if ((dist_curr < cutoff) && (dist_curr != 0)) {
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

        // Populate Python attributes.
        if (dist_curr < sepcut_value){
            bond_array_2(bond_count, 0) = dist_curr;
            bond_array_2(bond_count, 1) = xr;
            bond_array_2(bond_count, 2) = yr;
            bond_array_2(bond_count, 3) = zr;
            etypes.push_back(spec_curr);
            bond_inds.push_back(m);
        }

        bond_count++;

        // Update neighbor list.
        if (unique_check == 0) {
          neighbor_list.push_back(m);
          unique_check = 1;
        }
      }
      counter++;
    }
  }

  delete[] dists;
  delete[] xvals;
  delete[] yvals;
  delete[] zvals;
}

void LocalEnvironment ::compute_indices() {

  int no_atoms = rs.size();

  // Initialize a list of lists storing atom indices.
  int n_cutoffs = n_body_cutoffs.size();
  std::vector<int> empty;
  for (int i = 0; i < n_cutoffs; i++) {
    n_body_indices.push_back(empty);
  }

  int n_mb_cutoffs = many_body_cutoffs.size();
  for (int i = 0; i < n_mb_cutoffs; i++) {
    many_body_indices.push_back(empty);
  }

  // Store indices of atoms inside the 2-, 3-, and many-body cutoff spheres.
  double current_cutoff;
  for (int i = 0; i < no_atoms; i++) {
    double r_curr = rs[i];

    // Store n-body indices.
    for (int j = 0; j < n_cutoffs; j++) {
      current_cutoff = n_body_cutoffs[j];
      if (r_curr < current_cutoff) {
        n_body_indices[j].push_back(i);
      }
    }

    // Store many-body indices.
    for (int j = 0; j < n_mb_cutoffs; j++) {
      current_cutoff = many_body_cutoffs[j];
      if (r_curr < current_cutoff) {
        many_body_indices[j].push_back(i);
      }
    }
  }

  // Store triplets if the 3-body cutoff is nonzero.
  if (n_cutoffs > 1) {
    int n_3body = n_body_indices[1].size();
    double three_body_cutoff = n_body_cutoffs[1];
    double cross_bond_dist, x1, y1, z1, x2, y2, z2, r1, r2, x_diff, y_diff,
        z_diff;

    bond_array_3.conservativeResize(n_3body, 4);
    cross_bond_inds = Eigen::MatrixXi::Constant(n_3body, n_3body, -1);
    cross_bond_dists_py.conservativeResize(n_3body, n_3body);
    triplet_counts = Eigen::VectorXi::Zero(n_3body);

    // Check if separate cutoffs are given for the 3-body sphere.
    bool sepcut = false;
    int bc, bcn, bn, bm, bmn, btype_m, btype_n, btype_mn;
    double sepcut_value_1 = three_body_cutoff,
        sepcut_value_2 = three_body_cutoff, sepcut_value_3 = three_body_cutoff;
    if (cutoffs_mask.nspecie > 1
        && cutoffs_mask.threebody_mask.size() > 1
        && cutoffs_mask.threebody_cutoff_list.size() > 1
        && cutoffs_mask.specie_mask.size() > 1){
            sepcut = true;
            bc = cutoffs_mask.specie_mask[central_species];
            bcn = cutoffs_mask.nspecie * bc;
        }

    int ind1, ind2;
    std::vector<int> triplet = std::vector<int>{0, 0};
    int count;
    for (int i = 0; i < n_3body; i++) {
      ind1 = n_body_indices[1][i];
      x1 = xs[ind1];
      y1 = ys[ind1];
      z1 = zs[ind1];
      r1 = rs[ind1];

      // Populate the 3-body bond array.
      bond_array_3(i, 0) = r1;
      bond_array_3(i, 1) = x1 / r1;
      bond_array_3(i, 2) = y1 / r1;
      bond_array_3(i, 3) = z1 / r1;

      // If separate cutoffs are given, find the cutoff corresponding to the
      // current atom.
      if (sepcut){
          bm = cutoffs_mask.specie_mask[environment_species[ind1]];
          btype_m = cutoffs_mask.cut3b_mask[bm + bcn];
          sepcut_value_1 = cutoffs_mask.threebody_cutoff_list[btype_m];
          bmn = cutoffs_mask.nspecie * bm;
      }

      count = i + 1;
      for (int j = i + 1; j < n_body_indices[1].size(); j++) {
        ind2 = n_body_indices[1][j];
        r2 = rs[ind2];
        x_diff = x1 - xs[ind2];
        y_diff = y1 - ys[ind2];
        z_diff = z1 - zs[ind2];
        cross_bond_dist =
            sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
        
        if (sepcut){
            bn = cutoffs_mask.specie_mask[environment_species[ind2]];
            btype_n = cutoffs_mask.cut3b_mask[bn + bcn];
            sepcut_value_2 = cutoffs_mask.threebody_cutoff_list[btype_n];
            btype_mn = cutoffs_mask.cut3b_mask[bn + bmn];
            sepcut_value_3 = cutoffs_mask.threebody_cutoff_list[btype_mn];
        }

        if (r1 <= sepcut_value_1 && r2 <= sepcut_value_2
            && cross_bond_dist <= sepcut_value_3) {
          cross_bond_dists.push_back(cross_bond_dist);
          triplet[0] = ind1;
          triplet[1] = ind2;
          three_body_indices.push_back(triplet);
          
          cross_bond_inds(i, count) = ind2;
          cross_bond_dists_py(i, count) = cross_bond_dist;
          triplet_counts(i) += 1;
          count ++;
        }
      }
    }
  }
}

void LocalEnvironment ::compute_descriptors_and_gradients() {
  int n_calculators = descriptor_calculators.size();
  for (int i = 0; i < n_calculators; i++) {
    descriptor_calculators[i]->compute(*this);
    descriptor_vals.push_back(descriptor_calculators[i]->descriptor_vals);
    descriptor_force_dervs.push_back(
        descriptor_calculators[i]->descriptor_force_dervs);
    descriptor_stress_dervs.push_back(
        descriptor_calculators[i]->descriptor_stress_dervs);

    descriptor_norm.push_back(sqrt(descriptor_vals[i].dot(descriptor_vals[i])));
    force_dot.push_back(descriptor_force_dervs[i] * descriptor_vals[i]);
    stress_dot.push_back(descriptor_stress_dervs[i] * descriptor_vals[i]);

    // Clear descriptor calculator matrices to save memory.
    descriptor_calculators[i]->destroy_matrices();
  }
}

void LocalEnvironment ::compute_descriptors() {

  int n_calculators = descriptor_calculators.size();
  for (int i = 0; i < n_calculators; i++) {
    descriptor_calculators[i]->compute(*this);
    descriptor_vals.push_back(descriptor_calculators[i]->descriptor_vals);
    descriptor_norm.push_back(sqrt(descriptor_vals[i].dot(descriptor_vals[i])));

    // Clear descriptor calculator matrices to save memory.
    descriptor_calculators[i]->destroy_matrices();
  }
}

void LocalEnvironment ::compute_descriptor_squared() {
  // Assumes descriptors have already been computed.
  int n_calculators = descriptor_calculators.size();

  for (int i = 0; i < n_calculators; i++) {
    int desc_size = descriptor_vals[i].size();
    double desc_norm = descriptor_norm[i];
    Eigen::VectorXd desc_sq =
        Eigen::VectorXd::Zero(desc_size * (desc_size + 1) / 2);
    int desc_count = 0;

    for (int j = 0; j < desc_size; j++) {
      double desc_norm_j = descriptor_vals[i](j) / desc_norm;

      for (int k = j; k < desc_size; k++) {
        double desc_norm_k = descriptor_vals[i](k) / desc_norm;
        desc_sq(desc_count) = desc_norm_j * desc_norm_k;
        desc_count++;
      }
    }
    descriptor_squared.push_back(desc_sq);
  }
}

void LocalEnvironment ::compute_neighbor_descriptors() {

  int n_neighbors = neighbor_list.size();
  int n_descriptors = descriptor_calculators.size();
  int neighbor;
  LocalEnvironment env_curr;

  for (int m = 0; m < n_neighbors; m++) {
    neighbor = neighbor_list[m];
    env_curr = LocalEnvironment(structure, neighbor, cutoff, many_body_cutoffs,
                                descriptor_calculators);
    env_curr.compute_descriptors_and_gradients();

    // Add neighbor descriptors and norms.
    neighbor_descriptors.push_back(env_curr.descriptor_vals);
    neighbor_descriptor_norms.push_back(env_curr.descriptor_norm);

    // Add neighbor derivatives and force/descriptor dot products.
    std::vector<Eigen::MatrixXd> derivs, dots;
    for (int n = 0; n < n_descriptors; n++) {
      int n_descriptors = env_curr.descriptor_vals[n].size();
      derivs.push_back(env_curr.descriptor_force_dervs[n].block(
          3 * central_index, 0, 3, n_descriptors));
      dots.push_back(env_curr.force_dot[n].block(3 * central_index, 0, 3, 1));
    }
    neighbor_force_dervs.push_back(derivs);
    neighbor_force_dots.push_back(dots);
  }
}

void compute_neighbor_descriptors(std::vector<LocalEnvironment> &envs) {
  int n_envs = envs.size();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_envs; i++) {
    envs[i].compute_neighbor_descriptors();
  }
}

void compute_descriptors(std::vector<LocalEnvironment> &envs) {
  int n_envs = envs.size();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_envs; i++) {
    envs[i].compute_descriptors();
  }
}
