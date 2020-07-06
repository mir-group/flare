#include "py_hyps_mask.h"
#include "hyps_mask.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

namespace py = pybind11;

void AddParametersModule(py::module m) {
    py::class_<HypsMask>(m, "HypsMask")
        .def(py::init<>())
        .def_readwrite("nspecie", &HypsMask::nspecie)
        .def_readwrite("ntwobody", &HypsMask::ntwobody)
        .def_readwrite("nthreebody", &HypsMask::nthreebody)
        .def_readwrite("nmanybody", &HypsMask::nmanybody)
        .def_readwrite("twobody_start", &HypsMask::twobody_start)
        .def_readwrite("threebody_start", &HypsMask::threebody_start)
        .def_readwrite("manybody_start", &HypsMask::manybody_start)
        .def_readwrite("ncut3b", &HypsMask::ncut3b)
        .def_readwrite("specie_mask", &HypsMask::specie_mask)
        .def_readwrite("twobody_mask", &HypsMask::twobody_mask)
        .def_readwrite("threebody_mask", &HypsMask::threebody_mask)
        .def_readwrite("manybody_mask", &HypsMask::manybody_mask)
        .def_readwrite("cut3b_mask", &HypsMask::cut3b_mask)
        .def_readwrite("map", &HypsMask::map)
        .def_readwrite("twobody_cutoff_list", &HypsMask::twobody_cutoff_list)
        .def_readwrite("threebody_cutoff_list",
            &HypsMask::threebody_cutoff_list)
        .def_readwrite("manybody_cutoff_list",
            &HypsMask::manybody_cutoff_list)
        .def_readwrite("original_hyps", &HypsMask::original_hyps)
        .def_readwrite("hyps", &HypsMask::hyps)
        .def_readwrite("original_labels", &HypsMask::original_labels)
        .def_readwrite("kernels", &HypsMask::kernels)
        .def_readwrite("hyp_labels", &HypsMask::hyp_labels)
        .def_readwrite("train_noise", &HypsMask::train_noise)
        .def_readwrite("energy_noise", &HypsMask::energy_noise)
        .def_readwrite("kernel_name", &HypsMask::kernel_name)
        .def_readwrite("cutoffs", &HypsMask::cutoffs)

        // TODO: complete pickle definition
        .def(py::pickle(
            [](const HypsMask &s) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(s.cutoffs);
            },
            [](py::tuple t) { // __setstate__

                /* Create a new C++ instance */
                HypsMask p;

                /* Assign any additional state */
                p.cutoffs = 
                    t[0].cast<std::unordered_map<std::string, double>>();

                return p;
            }));
}
