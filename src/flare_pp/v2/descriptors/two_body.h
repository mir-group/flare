#ifndef TWO_BODY_H
#define TWO_BODY_H

#include <vector>
#include <string>
#include "compact_descriptor.h"
#include "compact_structure.h"

class CompactStructure;

class TwoBody : public CompactDescriptor {
public:
  double cutoff;
  int n_species;

  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function;
  std::string cutoff_name;
  std::vector<double> cutoff_hyps;

  TwoBody();
  TwoBody(double cutoff, int n_species,
          const std::string &cutoff_name,
          const std::vector<double> &cutoff_hyps);

  DescriptorValues compute_struc(CompactStructure &structure);
};

#endif
