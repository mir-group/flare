#include "two_body.h"

TwoBody ::TwoBody() {}

TwoBody ::TwoBody(double cutoff, int n_species){
    this->cutoff = cutoff;
    this->n_species = n_species;
}

DescriptorValues TwoBody ::compute_struc(CompactStructure &structure){

  // Initialize descriptor values.
  DescriptorValues desc = DescriptorValues();

  desc.n_descriptors = 1;
  desc.n_types = n_species * (n_species + 1) / 2;
  desc.n_atoms = structure.noa;
  desc.volume = structure.volume;

  // Count pairs.

  // Initialize arrays.

  // Store descriptors.

  return desc;
}
