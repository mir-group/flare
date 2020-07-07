#ifndef PYFLARE_CPP
#define PYFLARE_CPP

#include "Descriptor/py_descriptor.h"
#include "Structure/py_structure.h"
#include "Parameters/py_hyps_mask.h"

PYBIND11_MODULE(_C_flare, m) {
  AddDescriptorModule(m);
  AddStructureModule(m);
  AddParametersModule(m);
}

#endif