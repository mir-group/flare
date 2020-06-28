#ifndef PYFLARE_CPP
#define PYFLARE_CPP

#include "Descriptor/py_descriptor.h"

PYBIND11_MODULE(_C_flare, m) {
  AddDescriptorModule(m);
}

#endif
