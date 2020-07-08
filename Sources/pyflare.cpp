#ifndef PYFLARE_CPP
#define PYFLARE_CPP

#include "Descriptor/py_descriptor.h"
#include "Parameters/py_hyps_mask.h"
#include "Structure/py_structure.h"
#include "Environment/py_local_environment.h"

PYBIND11_MODULE(_C_flare, m) {
  AddDescriptorModule(m);
  AddStructureModule(m);
  AddParametersModule(m);
  AddLocalEnvironmentModule(m);
}

#endif
