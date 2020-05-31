""":class:`FLARE_Calculator` is a calculator compatible with `ASE`. You can build up `ASE Atoms` for your atomic structure, and use `get_forces`, `get_potential_energy` as general `ASE Calculators`, and use it in `ASE Molecular Dynamics` and our `ASE OTF` training module."""
import warnings
import numpy as np
import multiprocessing as mp
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.mgp.mgp import MappedGaussianProcess
from flare.predict import predict_on_structure_par_en, predict_on_structure_en
from ase.calculators.calculator import Calculator

class FLARE_Calculator(Calculator):
    """Build FLARE as an ASE Calculator, which is compatible with ASE Atoms and Molecular Dynamics.

    :param gp_model: FLARE's Gaussian process object
    :type gp_model: GaussianProcess
    :param mgp_model: FLARE's Mapped Gaussian Process object. `None` by default. MGP will only be used if `use_mapping` is set to True
    :type mgp_model: MappedGaussianProcess
    :param par: set to `True` if parallelize the prediction. `False` by default.
    :type par: Bool
    :param use_mapping: set to `True` if use MGP for prediction. `False` by default.
    :type use_mapping: Bool
    """

    def __init__(self, gp_model, mgp_model=None, par=False, use_mapping=False):
        super().__init__() # all set to default values, TODO: change
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
        return self.get_property('energy', atoms)


    def get_forces(self, atoms):
        return self.get_property('forces', atoms)


    def get_stress(self, atoms):
        if not self.use_mapping:
            warnings.warn('Stress is only implemented in MGP, not in GP. Will return zeros.')
        return self.get_property('stress', atoms)


    def get_uncertainties(self, atoms):
        return self.get_property('stds', atoms)


    def calculate(self, atoms):
        '''
        calculate properties including: energy, local energies, forces, stress, uncertainties

        :param atoms: ASE Atoms object
        :type atoms: Atoms
        '''
        if self.use_mapping:
            if self.par:
                self.calculate_mgp_par(atoms)
            else:
                self.calculate_mgp_serial(atoms)
        else:
            self.calculate_gp(atoms)


    def calculate_gp(self, atoms):
        nat = len(atoms)
        struc_curr = Structure(np.array(atoms.cell),
                               atoms.get_atomic_numbers(),
                               atoms.positions)

        if self.par:
            forces, stds, local_energies = \
                    predict_on_structure_par_en(struc_curr, self.gp_model)
        else:
            forces, stds, local_energies = \
                    predict_on_structure_en(struc_curr, self.gp_model)

        self.results['forces'] = forces
        self.results['stds'] = stds
        self.results['local_energies'] = local_energies
        self.results['energy'] = np.sum(local_energies)
        atoms.get_uncertainties = self.get_uncertainties

        # GP stress not implemented yet
        self.results['stresses'] = np.zeros((nat, 6))
        volume = atoms.get_volume()
        total_stress = np.sum(self.results['stresses'], axis=0)
        self.results['stress'] = total_stress / volume

        return forces


    def calculate_mgp_serial(self, atoms):
        nat = len(atoms)
        struc_curr = Structure(np.array(atoms.cell),
                               atoms.get_atomic_numbers(),
                               atoms.positions)

        self.results['forces'] = np.zeros((nat, 3))
        self.results['stresses'] = np.zeros((nat, 6))
        self.results['stds'] = np.zeros((nat, 3))
        self.results['local_energies'] = np.zeros(nat)

        for n in range(nat):
            chemenv = AtomicEnvironment(struc_curr, n,
                                        self.gp_model.cutoffs,
                                        cutoffs_mask = self.mgp_model.hyps_mask)
            f, v, vir, e = self.mgp_model.predict(chemenv, mean_only=False)
            self.results['forces'][n] = f
            self.results['stresses'][n] = vir
            self.results['stds'][n] = np.sqrt(np.absolute(v))
            self.results['local_energies'][n] = e

        volume = atoms.get_volume()
        total_stress = np.sum(self.results['stresses'], axis=0)
        self.results['stress'] = total_stress / volume
        self.results['energy'] = np.sum(self.results['local_energies'])

        atoms.get_uncertainties = self.get_uncertainties


    def calculate_mgp_par(self, atoms):
        # TODO: to be done
        self.calculate_mgp_serial(atoms)


    def calculation_required(self, atoms, quantities):
        return True

