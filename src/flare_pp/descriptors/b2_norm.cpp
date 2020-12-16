#include "b2_norm.h"
#include "cutoffs.h"
#include "descriptor.h"
#include "radial.h"
#include "structure.h"
#include "y_grad.h"
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>

B2_Norm ::B2_Norm() {}

B2_Norm ::B2_Norm(const std::string &radial_basis,
                  const std::string &cutoff_function,
                  const std::vector<double> &radial_hyps,
                  const std::vector<double> &cutoff_hyps,
                  const std::vector<int> &descriptor_settings)
  :B2(radial_basis, cutoff_function, radial_hyps, cutoff_hyps,
      descriptor_settings){}

DescriptorValues B2_Norm::compute_struc(Structure &structure){

}
