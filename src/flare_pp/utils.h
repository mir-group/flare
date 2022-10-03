#include "structure.h"
#include <tuple>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>

#ifndef UTILS_H
#define UTILS_H

namespace utils {

  /**
   Read an .xyz file and return a list of `Structure` objects, with a list of 
   indices of sparse atoms in the structures.

   @param filename A string to specify the name of the .xyz file, which contains
        all the DFT frames with energy, forces and stress.
   @param species_map A `std::map` object that maps chemical symbol (string) to
        species code in the `Structure` object. E.g.
        ```
        std::map<std::string, int> species_map = {{"H", 0,}, {"He", 1,}};
        ```
   @return A tuple of `Structure` list and sparse indices list.
   */
  std::tuple<std::vector<Structure>, std::vector<std::vector<std::vector<int>>>>
  read_xyz(std::string filename, std::map<std::string, int> species_map);

  template <typename Out>
  void split(const std::string &s, char delim, Out result);

  /**
   A useful function that mimic Python's `split` method of a string. 

   @param s A string of type `std::string`.
   @param delim The delimiter to separate the string `s`.
   @return A list of strings separated by the delimiter.
   */
  std::vector<std::string> split(const std::string &s, char delim);

  class Timer;
}

class utils::Timer {
public:
  Timer();

  double duration = 0;
  std::chrono::high_resolution_clock::time_point t_start, t_end;
  void tic();
  void toc(const char*);
  void toc(const char*, int rank);
};

#endif
