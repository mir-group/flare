#include "py_descriptor.h"
#include "y_grad.h"

namespace py = pybind11;

void AddDescriptorModule(py::module m) {
  auto subm = m.def_submodule("harmonics");
  subm.def("get_Y", &get_Y);
}
