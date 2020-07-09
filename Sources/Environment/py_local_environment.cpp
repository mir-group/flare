#include "py_local_environment.h"
#include "local_environment.h"
#include "Structure/structure.h"
#include "Parameters/hyps_mask.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

void AddLocalEnvironmentModule(py::module m) {
  py::class_<LocalEnvironment>(m, "AtomicEnvironment")
      // Constructor with integer species.
      .def(
          py::init<const Structure &, int,
                   std::unordered_map<std::string, double>, HypsMask>(),
          py::arg("structure"), py::arg("atom"), py::arg("cutoffs"),
          py::arg("cutoffs_mask") = HypsMask{})

      // Environment methods.
      .def("compute_environment", &LocalEnvironment::compute_environment)
      .def("compute_indices", &LocalEnvironment::compute_indices)

      // 2-body attributes.
      .def_readonly("bond_array_2", &LocalEnvironment::bond_array_2)
      .def_readonly("etypes", &LocalEnvironment::etypes)
      .def_readonly("bond_inds", &LocalEnvironment::bond_inds)

      // 3-body attributes.
      .def_readonly("bond_array_3", &LocalEnvironment::bond_array_3)
      .def_readonly("cross_bond_inds", &LocalEnvironment::cross_bond_inds)
      .def_readonly("cross_bond_dists", &LocalEnvironment::cross_bond_dists_py)
      .def_readonly("triplet_counts", &LocalEnvironment::triplet_counts);
}
