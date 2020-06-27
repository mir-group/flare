#ifndef PYFLARE_CPP
#define PYFLARE_CPP

#include "Harmonics/py_y_grad.h"

PYBIND11_MODULE(_C_flare, m) {
  AddHarmonicsModule(m);
}

#endif
