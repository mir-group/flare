#include "kernels.h"
#include "cutoffs.h"
#include "local_environment.h"
#include <cmath>
#include <iostream>

Kernel ::Kernel(){};
Kernel ::Kernel(std::vector<double> kernel_hyperparameters) {
  this->kernel_hyperparameters = kernel_hyperparameters;
};

double Kernel ::struc_struc_en(const StructureDescriptor &struc1,
                               const StructureDescriptor &struc2) {

  double kern_val = 0;

  // Double loop over environments.
  LocalEnvironment env1, env2;
  for (int i = 0; i < struc1.noa; i++) {
    env1 = struc1.local_environments[i];
    for (int j = 0; j < struc2.noa; j++) {
      env2 = struc2.local_environments[j];
      kern_val += env_env(env1, env2);
    }
  }

  return kern_val;
}
