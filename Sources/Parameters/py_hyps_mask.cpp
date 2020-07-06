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
                return py::make_tuple(
                    s.nspecie, s.ntwobody, s.nthreebody, s.nmanybody,
                    s.twobody_start, s.threebody_start, s.manybody_start,
                    s.ncut3b, s.specie_mask, s.twobody_mask, s.threebody_mask,
                    s.manybody_mask, s.cut3b_mask, s.map,
                    s.twobody_cutoff_list, s.threebody_cutoff_list,
                    s.manybody_cutoff_list, s.original_hyps, s.hyps,
                    s.original_labels, s.kernels, s.hyp_labels, 
                    s.train_noise, s.energy_noise, s.kernel_name,
                    s.cutoffs);
            },
            [](py::tuple t) { // __setstate__

                /* Create a new C++ instance */
                HypsMask p;

                /* Assign any additional state */
                p.nspecie = t[0].cast<int>();
                p.ntwobody = t[1].cast<int>();
                p.nthreebody = t[2].cast<int>();
                p.nmanybody = t[3].cast<int>();
                p.twobody_start = t[4].cast<int>();
                p.threebody_start = t[5].cast<int>();
                p.manybody_start = t[6].cast<int>();
                p.ncut3b = t[7].cast<int>();
                p.specie_mask = t[8].cast<std::vector<int>>();
                p.twobody_mask = t[9].cast<std::vector<int>>();
                p.threebody_mask = t[10].cast<std::vector<int>>();
                p.manybody_mask = t[11].cast<std::vector<int>>();
                p.cut3b_mask = t[12].cast<std::vector<int>>();
                p.map = t[13].cast<std::vector<int>>();
                p.twobody_cutoff_list = t[14].cast<std::vector<double>>();
                p.threebody_cutoff_list = t[15].cast<std::vector<double>>();
                p.manybody_cutoff_list = t[16].cast<std::vector<double>>();
                p.original_hyps = t[17].cast<std::vector<double>>();
                p.hyps = t[18].cast<std::vector<double>>();
                p.original_labels = t[19].cast<std::vector<std::string>>();
                p.kernels = t[20].cast<std::vector<std::string>>();
                p.hyp_labels = t[21].cast<std::vector<std::string>>();
                p.train_noise = t[22].cast<bool>();
                p.energy_noise = t[23].cast<double>();
                p.kernel_name = t[24].cast<std::string>();
                p.cutoffs =
                    t[25].cast<std::unordered_map<std::string, double>>();

                return p;
            }));
}
