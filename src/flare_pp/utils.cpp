#include "utils.h"
#include <Eigen/Dense>

#define MAXLINE 1024


template <typename Out>
void utils::split(const std::string &s, char delim, Out result) {
  std::istringstream iss(s);
  std::string item;
  while (std::getline(iss, item, delim)) {
    if (item.length() > 0) *result++ = item;
  }
}

std::vector<std::string> utils::split(const std::string &s, char delim) {
  /* Convert a line of string into a list
   * Similar to the python method str.split()
   */
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

std::tuple<std::vector<Structure>, std::vector<std::vector<std::vector<int>>>>
utils::read_xyz(std::string filename, std::map<std::string, int> species_map) {

  std::ifstream file(filename);
  int n_atoms, atom_ind;
  Eigen::MatrixXd cell, positions;
  Eigen::VectorXd energy, forces, stress;
  std::vector<int> species;
  std::vector<int> sparse_inds;
  int pos_col = 0;
  int forces_col = 0;

  std::vector<Structure> structure_list;
  std::vector<std::vector<int>> sparse_inds_0;
  std::vector<std::string> values;
  int new_frame_line = 0;

  int i;
  
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      values = split(line, ' ');
      if (values.size() == 1) {
        // the 1st line of a block, the number of atoms in a frame
        n_atoms = std::stoi(values[0]);
        cell = Eigen::MatrixXd::Zero(3, 3);
        positions = Eigen::MatrixXd::Zero(n_atoms, 3);
        forces = Eigen::VectorXd::Zero(n_atoms * 3);
        energy = Eigen::VectorXd::Zero(1);;
        stress = Eigen::VectorXd::Zero(6);
        species = std::vector(n_atoms, 0);
        sparse_inds = {};
        atom_ind = 0;
        pos_col = 0;
        forces_col = 0;
        new_frame_line = 0;
      } else if (new_frame_line == 0) {
        // the 2nd line of a block, including cell, energy, stress, sparse indices
        int v = 0;
        while (v < values.size()) {
          if (values[v].find(std::string("Lattice")) != std::string::npos) {
            // Example: Lattice="1 0 0 0 1 0 0 0 1"
            cell(0, 0) = std::stod(values[v].substr(9, values[v].length() - 9));
            cell(0, 1) = std::stod(values[v + 1]);
            cell(0, 2) = std::stod(values[v + 2]);
            cell(1, 0) = std::stod(values[v + 3]);
            cell(1, 1) = std::stod(values[v + 4]);
            cell(1, 2) = std::stod(values[v + 5]);
            cell(2, 0) = std::stod(values[v + 6]);
            cell(2, 1) = std::stod(values[v + 7]);
            cell(2, 2) = std::stod(values[v + 8].substr(0, values[v + 8].length() - 1));
            v += 9;
          } else if (values[v].find(std::string("energy")) != std::string::npos \
                  && values[v].find(std::string("free_energy")) == std::string::npos) {
            // Example: energy=-2.0
            energy(0) = std::stod(values[v].substr(7, values[v].length() - 7));
            v++;
          } else if (values[v].find(std::string("stress")) != std::string::npos) {
            // Example: stress="1 0 0 0 1 0 0 0 1"
            stress(0) = - std::stod(values[v].substr(8, values[v].length() - 8)); // xx
            stress(1) = - std::stod(values[v + 1]); // xy
            stress(2) = - std::stod(values[v + 2]); // xz
            stress(3) = - std::stod(values[v + 4]); // yy
            stress(4) = - std::stod(values[v + 5]); // yz
            stress(5) = - std::stod(values[v + 8].substr(0, values[v + 8].length() - 1)); // zz
            v += 9;
          } else if (values[v].find(std::string("sparse_indices")) != std::string::npos) {
            // Example: sparse_indices="0 2 4 6" or sparse_indices="2"
            size_t n = std::count(values[v].begin(), values[v].end(), '"');
            assert(n <= 1);
            if (n == 0) { // Example: sparse_indices=2 or sparse_indices=
              if (values[v].length() > 15) { 
                sparse_inds.push_back(std::stoi(values[v].substr(15, values[v].length() - 15)));
              }
              v++;
            } else if (n == 1) { // Example: sparse_indices="0 2 4 6"
              sparse_inds.push_back(std::stoi(values[v].substr(16, values[v].length() - 16)));
              v++;
              while (values[v].find(std::string("\"")) == std::string::npos) {
                sparse_inds.push_back(std::stoi(values[v]));
                v++;
              }
              sparse_inds.push_back(std::stoi(values[v].substr(0, values[v].length() - 1)));
              v++;
            }
          } else if (values[v].find(std::string("Properties")) != std::string::npos) {
            // Example: Properties=species:S:1:pos:R:3:forces:R:3:magmoms:R:1 
            std::string str = values[v];
            std::vector<std::string> props = split(str, ':');
            bool find_pos = false;
            bool find_forces = false;
            for (int p = 0; p < props.size(); p += 3) {
              // Find the starting column of positions
              if (props[p].find(std::string("pos")) == std::string::npos) {
                if (!find_pos) pos_col += std::stoi(props[p + 2]);
              } else {
                find_pos = true;
              }
              // Find the starting column of forces
              if (props[p].find(std::string("forces")) == std::string::npos) {
                if (!find_forces) forces_col += std::stoi(props[p + 2]);
              } else {
                find_forces = true;
              }
            }
            v++;
          } else {
            v++;
          }
        }
        new_frame_line = 1;
      } else if (new_frame_line > 0) {
        // the rest n_atoms lines of a block, with format "symbol x y z fx fy fz"
        species[atom_ind] = species_map[values[0]];
        positions(atom_ind, 0) = std::stod(values[pos_col + 0]);
        positions(atom_ind, 1) = std::stod(values[pos_col + 1]);
        positions(atom_ind, 2) = std::stod(values[pos_col + 2]);
        forces(3 * atom_ind + 0) = std::stod(values[forces_col + 0]);
        forces(3 * atom_ind + 1) = std::stod(values[forces_col + 1]);
        forces(3 * atom_ind + 2) = std::stod(values[forces_col + 2]);
        atom_ind++;

        if (new_frame_line == n_atoms) {
          Structure structure(cell, species, positions);
          structure.energy = energy;
          structure.forces = forces;
          structure.stresses = stress;
          structure_list.push_back(structure); 
          sparse_inds_0.push_back(sparse_inds); // TODO: multiple kernels with different sparse inds
        }

        new_frame_line++;
      } else {
        // raise error
        printf("Unknown line!!!");
      }
    }
    file.close();
  }
  std::vector<std::vector<std::vector<int>>> sparse_inds_list;
  sparse_inds_list.push_back(sparse_inds_0);
  return std::make_tuple(structure_list, sparse_inds_list);
}

utils::Timer::Timer() {}

void utils::Timer::tic() {
  t_start = std::chrono::high_resolution_clock::now();
}

void utils::Timer::toc(const char* code_name) {
  t_end = std::chrono::high_resolution_clock::now();
  duration = (double) std::chrono::duration_cast<std::chrono::milliseconds>( t_end - t_start ).count();
  std::cout << "Time: " << code_name << " " << duration << " ms" << std::endl;
}

void utils::Timer::toc(const char* code_name, int rank) {
  t_end = std::chrono::high_resolution_clock::now();
  duration = (double) std::chrono::duration_cast<std::chrono::milliseconds>( t_end - t_start ).count();
  std::cout << "Rank " << rank << " Time: " << code_name << " " << duration << " ms" << std::endl;
}
