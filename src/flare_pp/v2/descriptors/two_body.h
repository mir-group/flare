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

  TwoBody();
  TwoBody(double cutoff, int n_species);

  DescriptorValues compute_struc(CompactStructure &structure);
};

#endif
