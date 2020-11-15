#ifndef THREE_BODY_H
#define THREE_BODY_H

#include "compact_descriptor.h"
#include "compact_structure.h"
#include <string>
#include <vector>

class CompactStructure;

class ThreeBody : public CompactDescriptor {
public:
  double cutoff;
  int n_species;

  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function;
  std::string cutoff_name;
  std::vector<double> cutoff_hyps;

  ThreeBody();
  ThreeBody(double cutoff, int n_species, const std::string &cutoff_name,
            const std::vector<double> &cutoff_hyps);

  DescriptorValues compute_struc(CompactStructure &structure);
};

#endif