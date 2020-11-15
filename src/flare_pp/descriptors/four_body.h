#ifndef FOUR_BODY_H
#define FOUR_BODY_H

#include "compact_descriptor.h"
#include "structure.h"
#include <string>
#include <vector>

class CompactStructure;

class FourBody : public CompactDescriptor {
public:
  double cutoff;
  int n_species;

  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function;
  std::string cutoff_name;
  std::vector<double> cutoff_hyps;

  FourBody();
  FourBody(double cutoff, int n_species, const std::string &cutoff_name,
           const std::vector<double> &cutoff_hyps);

  DescriptorValues compute_struc(Structure &structure);
};

#endif
