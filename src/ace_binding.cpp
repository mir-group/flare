#include "y_grad.h"
#include "structure.h"
#include "local_environment.h"
#include "descriptor.h"
#include "sparse_gp.h"
#include "kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <vector>
#include <Eigen/Dense>

namespace py = pybind11;

PYBIND11_MODULE(ace, m){
    // Structure
    py::class_<Structure>(m, "Structure")
        .def(py::init<const Eigen::MatrixXd &,
                      const std::vector<int> &,
                      const Eigen::MatrixXd &>())
        .def_readwrite("cell", &Structure::cell)
        .def_readwrite("species", &Structure::species)          
        .def_readwrite("positions", &Structure::positions)
        .def_readwrite("cell_transpose", &Structure::cell_transpose)
        .def_readwrite("wrapped_positions", &Structure::wrapped_positions)
        .def_readwrite("volume", &Structure::volume)
        .def("wrap_positions", &Structure::wrap_positions);

    py::class_<StructureDescriptor, Structure>(m, "StructureDescriptor")
        .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                      const Eigen::MatrixXd &, double>())
        .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                      const Eigen::MatrixXd &, double,
                      DescriptorCalculator *>())
        .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                      const Eigen::MatrixXd &, double, std::vector<double>>())
        .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                      const Eigen::MatrixXd &, double, std::vector<double>,
                      DescriptorCalculator *>())        
        .def_readwrite("local_environments",
            &StructureDescriptor::local_environments)
        .def_readwrite("energy", &StructureDescriptor::energy)
        .def_readwrite("forces", &StructureDescriptor::forces)
        .def_readwrite("stresses", &StructureDescriptor::stresses)
        .def_readwrite("cutoff", &StructureDescriptor::cutoff)
        .def_readwrite("nested_cutoffs", &StructureDescriptor::nested_cutoffs);

    // Local environment
    py::class_<LocalEnvironment>(m, "LocalEnvironment")
        .def(py::init<const Structure &, int, double>())
        .def_readwrite("sweep", &LocalEnvironment::sweep)
        .def_readwrite("structure_volume", &LocalEnvironment::structure_volume)
        .def_readwrite("central_index", &LocalEnvironment::central_index)
        .def_readwrite("central_species", &LocalEnvironment::central_species)
        .def_readwrite("noa", &LocalEnvironment::noa)
        .def_readwrite("environment_indices",
            &LocalEnvironment::environment_indices)
        .def_readwrite("environment_species",
            &LocalEnvironment::environment_species)
        .def_readwrite("rs", &LocalEnvironment::rs)
        .def_readwrite("xs", &LocalEnvironment::xs)
        .def_readwrite("ys", &LocalEnvironment::ys)
        .def_readwrite("zs", &LocalEnvironment::zs)
        .def_readwrite("cutoff", &LocalEnvironment::cutoff)
        .def_readwrite("descriptor_vals", &LocalEnvironment::descriptor_vals)
        .def_readwrite("descriptor_force_dervs",
            &LocalEnvironment::descriptor_force_dervs)
        .def_readwrite("descriptor_stress_dervs",
            &LocalEnvironment::descriptor_stress_dervs);

        Eigen::VectorXd descriptor_vals;
        Eigen::MatrixXd descriptor_force_dervs, descriptor_stress_dervs,
            force_dot, stress_dot;
    // Descriptor calculators
    py::class_<DescriptorCalculator>(m, "DescriptorCalculator")
        .def("compute", &DescriptorCalculator::compute)
        .def_readwrite("radial_basis", &DescriptorCalculator::radial_basis)
        .def_readwrite("cutoff_function", 
            &DescriptorCalculator::cutoff_function)
        .def_readwrite("descriptor_vals",
            &DescriptorCalculator::descriptor_vals)
        .def_readwrite("descriptor_force_dervs",
            &DescriptorCalculator::descriptor_force_dervs)
        .def_readwrite("descriptor_stress_dervs",
            &DescriptorCalculator::descriptor_stress_dervs)
        .def_readwrite("radial_hyps",
            &DescriptorCalculator::radial_hyps)
        .def_readwrite("cutoff_hyps",
            &DescriptorCalculator::cutoff_hyps)
        .def_readwrite("descriptor_settings",
            &DescriptorCalculator::descriptor_settings);

    py::class_<B2_Calculator, DescriptorCalculator>(m, "B2_Calculator")
        .def(py::init<const std::string &, const std::string &,
             const std::vector<double> &, const std::vector<double> &,
             const std::vector<int> &>());

    // Kernel functions
    py::class_<Kernel>(m, "Kernel")
        .def("env_env", &Kernel::env_env)
        .def("env_struc", &Kernel::env_struc);

    py::class_<TwoBodyKernel, Kernel>(m, "TwoBodyKernel")
        .def(py::init<double, double, std::string, std::vector<double>>());
    
    py::class_<ThreeBodyKernel, Kernel>(m, "ThreeBodyKernel")
        .def(py::init<double, double, std::string, std::vector<double>>());
    
    py::class_<DotProductKernel, Kernel>(m, "DotProductKernel")
        .def(py::init<double, double>());

    // Sparse GP
    py::class_<SparseGP>(m, "SparseGP")
        .def(py::init<std::vector<Kernel *>, double, double, double>())
        .def_readwrite("Kuu", &SparseGP::Kuu)
        .def_readwrite("Kuf", &SparseGP::Kuf)
        .def_readwrite("Sigma", &SparseGP::Sigma)
        .def_readwrite("noise", &SparseGP::noise)
        .def_readwrite("noise_matrix", &SparseGP::noise_matrix)
        .def_readwrite("kernels", &SparseGP::kernels)
        .def_readwrite("alpha", &SparseGP::alpha)
        .def_readwrite("y", &SparseGP::y)
        .def_readwrite("hyperparameters", &SparseGP::hyperparameters)
        .def("add_sparse_environment",
             &SparseGP::add_sparse_environment)
        .def("add_sparse_environment_serial",
             &SparseGP::add_sparse_environment_serial)
        .def("add_training_structure", &SparseGP::add_training_structure)
        .def("add_training_structure_serial",
            &SparseGP::add_training_structure_serial)
        .def("update_alpha", &SparseGP::update_alpha)
        .def("predict", &SparseGP::predict)
        .def("predict_serial", &SparseGP::predict_serial);
}
