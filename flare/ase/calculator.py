""":class:`FLARE_Calculator` is a calculator compatible with `ASE`.
You can build up `ASE Atoms` for your atomic structure, and use `get_forces`,
`get_potential_energy` as general `ASE Calculators`, and use it in
`ASE Molecular Dynamics` and our `ASE OTF` training module. For the usage
users can refer to `ASE Calculator module <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_
and `ASE Calculator tutorial <https://wiki.fysik.dtu.dk/ase/ase/atoms.html#adding-a-calculator>`_."""

import warnings
import numpy as np
import multiprocessing as mp
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.mgp import MappedGaussianProcess
from flare.predict import (
    predict_on_structure_par_en,
    predict_on_structure_en,
    predict_on_structure_efs,
    predict_on_structure_efs_par,
)
from ase.calculators.calculator import Calculator


def get_rebuild_from_err(err_msg, rebuild_dict, newbound_dict):
    warnings.warn("Re-build map with a new lower bound")
    re_dict = err_msg.args[0]
    nb_dict = err_msg.args[1]
    for xb in re_dict:  # collect two & three body maps
        if xb in rebuild_dict:
            for s_ind, spc in enumerate(re_dict[xb]):  # collect all species
                if spc in rebuild_dict[xb]:
                    spc_ind = rebuild_dict[xb].index(spc)
                    if nb_dict[xb][s_ind] < newbound_dict[xb][spc_ind]:
                        newbound_dict[xb][spc_ind] = nb_dict[xb][s_ind]
                else:
                    rebuild_dict[xb].append(spc)
                    newbound_dict[xb].append(nb_dict[xb][s_ind])
        else:
            rebuild_dict[xb] = re_dict[xb]
            newbound_dict[xb] = nb_dict[xb]


class FLARE_Calculator(Calculator):
    """
    Build FLARE as an ASE Calculator, which is compatible with ASE Atoms and
    Molecular Dynamics.
    Args:
        gp_model (GaussianProcess): FLARE's Gaussian process object
        mgp_model (MappedGaussianProcess): FLARE's Mapped Gaussian Process
            object. `None` by default. MGP will only be used if `use_mapping`
            is set to True.
        par (Bool): set to `True` if parallelize the prediction. `False` by
            default.
        use_mapping (Bool): set to `True` if use MGP for prediction. `False`
            by default.
    """

    def __init__(self, gp_model, mgp_model=None, par=False, use_mapping=False):
        super().__init__()  # all set to default values, TODO: change
        self.mgp_model = mgp_model
        self.gp_model = gp_model
        self.use_mapping = use_mapping
        self.par = par
        self.results = {}

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.results.keys():
            if not allow_calculation:
                return None
            self.calculate(atoms)
        return self.results[name]

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.get_property("energy", atoms)

    def get_forces(self, atoms):
        return self.get_property("forces", atoms)

    def get_stress(self, atoms):
        return self.get_property("stress", atoms)

    def get_uncertainties(self, atoms):
        return self.get_property("stds", atoms)

    def calculate(self, atoms):
        """
        Calculate properties including: energy, local energies, forces,
            stress, uncertainties.

        Args:
            atoms (FLARE_Atoms): FLARE_Atoms object
        """

        if self.use_mapping:
            self.calculate_mgp(atoms)
        else:
            self.calculate_gp(atoms)

        # get global properties
        volume = atoms.get_volume()
        total_stress = np.sum(self.results["partial_stresses"], axis=0)
        self.results["stress"] = total_stress / volume
        self.results["energy"] = np.sum(self.results["local_energies"])

    def calculate_gp(self, atoms):
        # Compute energy, forces, and stresses and their uncertainties
        if self.par:
            res = predict_on_structure_efs_par(
                atoms, self.gp_model, write_to_structure=False
            )
        else:
            res = predict_on_structure_efs(
                atoms, self.gp_model, write_to_structure=False
            )

        # Set the energy, force, and stress attributes of the calculator.
        res_name = [
            "local_energies",
            "forces",
            "partial_stresses",
            "local_energy_stds",
            "stds",
            "partial_stress_stds",
        ]
        res_dims = [1, 3, 6, 1, 3, 6]
        for i in range(len(res_name)):
            if len(res[i].shape) == 2:
                assert (
                    res[i].shape[1] == res_dims[i],
                    f"{res_name[i]} shape doesn't match, "
                    f"{res[i].shape[1]} and {res_dims[i]}",
                )
            elif len(res[i].shape) == 1:
                assert res_dims[i] == 1

            self.results[res_name[i]] = res[i]

    def calculate_mgp(self, atoms):
        nat = len(atoms)

        self.results["forces"] = np.zeros((nat, 3))
        self.results["partial_stresses"] = np.zeros((nat, 6))
        self.results["stds"] = np.zeros((nat, 3))
        self.results["local_energies"] = np.zeros(nat)

        rebuild_dict = {}
        newbound_dict = {}
        repredict_atoms = []
        for n in range(nat):
            chemenv = AtomicEnvironment(
                atoms, n, self.gp_model.cutoffs, cutoffs_mask=self.mgp_model.hyps_mask
            )

            # TODO: Check that stress is being calculated correctly.
            try:
                f, v, vir, e = self.mgp_model.predict(chemenv)
                self.results["forces"][n] = f
                self.results["partial_stresses"][n] = vir
                self.results["stds"][n] = np.sqrt(np.absolute(v))
                self.results["local_energies"][n] = e

            except ValueError as err_msg:  # if lower_bound error is raised
                get_rebuild_from_err(err_msg, rebuild_dict, newbound_dict)
                repredict_atoms.append((n, chemenv))

        if len(rebuild_dict) > 0:
            # rebuild map for those problematic species
            for xb in rebuild_dict:
                for s_ind, spc in enumerate(rebuild_dict[xb]):
                    map_ind = self.mgp_model.maps[xb].find_map_index(spc)
                    rebuild_map = self.mgp_model.maps[xb].maps[map_ind]
                    rebuild_map.set_bounds(
                        newbound_dict[xb][s_ind], rebuild_map.bounds[1]
                    )
                    rebuild_map.build_map_container()
                    rebuild_map.build_map(self.gp_model)

            # re-predict forces, energies, etc. for those problematic atoms
            for ra in repredict_atoms:
                n, chemenv = ra
                f, v, vir, e = self.mgp_model.predict(chemenv)
                self.results["forces"][n] = f
                self.results["partial_stresses"][n] = vir
                self.results["stds"][n] = np.sqrt(np.absolute(v))
                self.results["local_energies"][n] = e

    def calculation_required(self, atoms, quantities):
        return True
