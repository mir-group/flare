""":class:`FLARE_Calculator` is a calculator compatible with `ASE`.
You can build up `ASE Atoms` for your atomic structure, and use `get_forces`,
`get_potential_energy` as general `ASE Calculators`, and use it in
`ASE Molecular Dynamics` and our `ASE OTF` training module. For the usage
users can refer to `ASE Calculator module <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_
and `ASE Calculator tutorial <https://wiki.fysik.dtu.dk/ase/ase/atoms.html#adding-a-calculator>`_."""

import warnings
import numpy as np
import multiprocessing as mp
import json

from flare.descriptors.env import AtomicEnvironment
from . import GaussianProcess
from ..mgp import MappedGaussianProcess
from .predict import (
    predict_on_structure_par_en,
    predict_on_structure_en,
    predict_on_structure_efs,
    predict_on_structure_efs_par,
)
from flare.utils import NumpyEncoder

from ase.calculators.calculator import Calculator, all_changes


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

    implemented_properties = ["energy", "forces", "stress", "stds"]

    def __init__(
        self, gp_model, mgp_model=None, par=False, use_mapping=False, **kwargs
    ):
        super().__init__()  # all set to default values, TODO: change
        self.mgp_model = mgp_model
        self.gp_model = gp_model
        self.use_mapping = use_mapping
        self.par = par
        self.results = {}

    def get_uncertainties(self, atoms):
        return self.get_property("stds", atoms)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, local energies, forces,
            stress, uncertainties.

        Args:
            atoms (FLARE_Atoms): FLARE_Atoms object
        """

        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if properties is None:
            properties = self.implemented_properties

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
        self.results["local_energies"] = res[0]
        self.results["forces"] = res[1]
        self.results["partial_stresses"] = -res[2][:, [0, 3, 5, 4, 2, 1]]
        self.results["local_energy_stds"] = res[3]
        self.results["stds"] = res[4]
        self.results["partial_stress_stds"] = res[5][:, [0, 3, 5, 4, 2, 1]]

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
                self.results["stds"][n][0] = np.sqrt(np.absolute(v))
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
                self.results["partial_stresses"][n] = -vir[[0, 3, 5, 4, 2, 1]]
                self.results["stds"][n][0] = np.sqrt(np.absolute(v))
                self.results["local_energies"][n] = e

    def calculation_required(self, atoms, quantities):
        return True

    def as_dict(self):
        outdict = {}

        outdict["class"] = self.__class__.__name__

        gp_dict = self.gp_model.as_dict()
        outdict["gp_model"] = gp_dict

        outdict["use_mapping"] = self.use_mapping
        if self.use_mapping:
            mgp_dict = self.mgp_model.as_dict()
            outdict["mgp_model"] = mgp_dict
        else:
            outdict["mgp_model"] = None

        outdict["par"] = self.par
        outdict["results"] = self.results
        return outdict

    @staticmethod
    def from_dict(dct):
        dct["gp_model"] = GaussianProcess.from_dict(dct["gp_model"])
        if dct["use_mapping"]:
            dct["mgp_model"] = MappedGaussianProcess.from_dict(dct["mgp_model"])

        calc = FLARE_Calculator(**dct)
        res = dct["results"]
        for key in res:
            if isinstance(res[key], float):
                calc.results[key] = res[key]
            if isinstance(res[key], list):
                calc.results[key] = np.array(res[key])

        if dct["use_mapping"]:
            for xb in calc.mgp_model.maps:
                xb_map = calc.mgp_model.maps[xb]
                xb_map.hyps_mask = calc.gp_model.hyps_mask

        return calc

    def write_model(self, name):
        if ".json" != name[-5:]:
            name += ".json"
        with open(name, "w") as f:
            json.dump(self.as_dict(), f, cls=NumpyEncoder)

    @staticmethod
    def from_file(name):
        with open(name, "r") as f:
            calc = FLARE_Calculator.from_dict(json.loads(f.readline()))

        return calc

    def build_map(self):
        self.mgp_model.build_map(self.gp_model)
