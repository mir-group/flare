''':class:`FLARE_Calculator` is a calculator compatible with `ASE`.
You can build up `ASE Atoms` for your atomic structure, and use `get_forces`,
`get_potential_energy` as general `ASE Calculators`, and use it in
`ASE Molecular Dynamics` and our `ASE OTF` training module. For the usage
users can refer to `ASE Calculator module <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_
and `ASE Calculator tutorial <https://wiki.fysik.dtu.dk/ase/ase/atoms.html#adding-a-calculator>`_.'''

import warnings
import numpy as np
import multiprocessing as mp
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.mgp import MappedGaussianProcess
from flare.predict import predict_on_structure_par_en, \
    predict_on_structure_en, predict_on_structure_efs, \
    predict_on_structure_efs_par
from ase.calculators.calculator import Calculator


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

    def get_property(self, name, atoms=None, allow_calculation=True,
                     structure=None):
        if name not in self.results.keys():
            if not allow_calculation:
                return None
            self.calculate(atoms, structure)
        return self.results[name]

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.get_property('energy', atoms)

    def get_forces(self, atoms):
        return self.get_property('forces', atoms)

    def get_stress(self, atoms):
        return self.get_property('stress', atoms)

    def calculate(self, atoms, structure):
        '''
        Calculate properties including: energy, local energies, forces,
            stress, uncertainties.

        :param atoms: ASE Atoms object
        :type atoms: Atoms
        '''

        # If a structure isn't given, create one based on the atoms object.
        if structure is None:
            structure = Structure(
                np.array(atoms.cell), atoms.get_atomic_numbers(),
                atoms.positions)

        if self.use_mapping:
            if self.par:
                self.calculate_mgp_par(atoms, structure)
            else:
                self.calculate_mgp_serial(atoms, structure)
        else:
            self.calculate_gp(atoms, structure)

    def calculate_gp(self, atoms, structure):
        # Compute energy, forces, and stresses and their uncertainties, and
        # write them to the structure object.
        if self.par:
            local_energies, forces, partial_stresses, _, _, _ = \
                predict_on_structure_efs_par(structure, self.gp_model)
        else:
            local_energies, forces, partial_stresses, _, _, _ = \
                predict_on_structure_efs(structure, self.gp_model)

        # Set the energy, force, and stress attributes of the calculator.
        self.results['energy'] = np.sum(local_energies)
        self.results['forces'] = forces
        volume = atoms.get_volume()
        total_stress = np.sum(partial_stresses, axis=0)
        self.results['stress'] = total_stress / volume

    def calculate_mgp_serial(self, atoms, structure):
        nat = len(atoms)

        self.results['forces'] = np.zeros((nat, 3))
        partial_stresses = np.zeros((nat, 6))
        stds = np.zeros((nat, 3))
        local_energies = np.zeros(nat)

        for n in range(nat):
            chemenv = AtomicEnvironment(
                structure, n, self.gp_model.cutoffs,
                cutoffs_mask=self.mgp_model.hyps_mask)

            # TODO: Check that stress is being calculated correctly.
            try:
                f, v, vir, e = self.mgp_model.predict(chemenv, mean_only=False)
            except ValueError:  # if lower_bound error is raised
                warnings.warn('Re-build map with a new lower bound')
                self.mgp_model.build_map(self.gp_model)

                f, v, vir, e = self.mgp_model.predict(chemenv, mean_only=False)

            self.results['forces'][n] = f
            partial_stresses[n] = vir
            stds[n] = np.sqrt(np.absolute(v))
            local_energies[n] = e

        volume = atoms.get_volume()
        total_stress = np.sum(partial_stresses, axis=0)
        self.results['stress'] = total_stress / volume
        self.results['energy'] = np.sum(local_energies)

        # Record structure attributes.
        structure.local_energies = local_energies
        structure.forces = np.copy(self.results['forces'])
        structure.partial_stresses = partial_stresses
        structure.stds = stds

    def calculate_mgp_par(self, atoms, structure):
        # TODO: to be done
        self.calculate_mgp_serial(atoms, structure)

    def calculation_required(self, atoms, quantities):
        return True
