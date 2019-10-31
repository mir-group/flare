""":class:`FLARE_Calculator` is a calculator compatible with `ASE`. You can build up `ASE Atoms` for your atomic structure, and use `get_forces`, `get_potential_energy` as general `ASE Calculators`, and use it in `ASE Molecular Dynamics` and our `ASE OTF` training module."""
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

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if self.use_mapping:
            print('MGP energy mapping not implemented, give GP prediction')
        forces = self.get_forces_gp(atoms)
        return self.results['energy']

    def get_forces(self, atoms):
        if self.use_mapping:
            if self.par:
                return self.get_forces_mgp_par(atoms)
            else:
                return self.get_forces_mgp_serial(atoms)
        else:
            return self.get_forces_gp(atoms)

    def get_forces_gp(self, atoms):
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

        self.results['stds'] = stds
        self.results['local_energies'] = local_energies
        self.results['energy'] = np.sum(local_energies)
        atoms.get_uncertainties = self.get_uncertainties

        return forces

    def get_forces_mgp_serial(self, atoms):
        nat = len(atoms)
        struc_curr = Structure(np.array(atoms.cell), 
                               atoms.get_atomic_numbers(),
                               atoms.positions)

        forces = np.zeros((nat, 3))
        stds = np.zeros((nat, 3))
        for n in range(nat):
            chemenv = AtomicEnvironment(struc_curr, n,
                                        self.mgp_model.cutoffs)
            f, v = self.mgp_model.predict(chemenv, mean_only=False)
            forces[n] = f
            stds[n] = np.sqrt(np.absolute(v))

        self.results['stds'] = stds
        atoms.get_uncertainties = self.get_uncertainties
        return forces

    def get_forces_mgp_par(self, atoms):
        return self.get_forces_mgp_serial(atoms)
#        comm = MPI.COMM_WORLD
#        size = comm.Get_size()
#        rank = comm.Get_rank()
#
#        nat = len(atoms)
#        struc_curr = Structure(np.array(atoms.cell), 
#                               atoms.get_atomic_numbers(),
#                               atoms.positions)
#        
#        NumPerRank = nat // size
#        NumRemainder = nat % size
#        forces = None
#        stds = None
#        if rank < nat:
#            if rank <= NumRemainder:
#                N = NumPerRank
#                intercept = 0
#            else:
#                N = NumPerRank - 1
#                intercept = NumRemainder
#
#            forces_sub = np.zeros((N, 3))
#            stds_sub = np.zeros((N, 3))
#            for i in range(N):
#                n = intercept + rank * N + i
#                chemenv = AtomicEnvironment(struc_curr, n,
#                                self.mgp_model.cutoffs)
#                f, v = self.mgp_model.predict(chemenv, mean_only=False)
#                forces_sub[i, :] = f
#                stds_sub[i, :] = np.sqrt(np.absolute(v))
#            print('rank:', rank, ', forces_sub:', N)
#
#        if rank == 0:
#            forces = np.empty((nat, 3))
#            stds = np.empty((nat, 3))
#
#        comm.Gather(forces_sub, forces, root=0)
#        comm.Gather(stds_sub, stds, root=0)
#
#        self.results['stds'] = stds
#        atoms.get_uncertainties = self.get_uncertainties
#        return forces

    def get_stress(self, atoms):
        return np.eye(3)

    def calculation_required(self, atoms, quantities):
        return True

    def get_uncertainties(self):
        return self.results['stds']

    def train_gp(self, monitor=True):
        self.gp_model.train(monitor)

    def build_mgp(self, skip=True):
        # l_bound not implemented

        if skip:
            return 1

        # set svd rank based on the training set, grid number and threshold 1000
        grid_params = self.mgp_model.grid_params
        struc_params = self.mgp_model.struc_params
        lmp_file_name = self.mgp_model.lmp_file_name
        mean_only = self.mgp_model.mean_only
        container_only = False

        train_size = len(self.gp_model.training_data)
        rank_2 = np.min([1000, grid_params['grid_num_2'], train_size*3])
        rank_3 = np.min([1000, grid_params['grid_num_3'][0]**3, train_size*3])
        grid_params['svd_rank_2'] = rank_2
        grid_params['svd_rank_3'] = rank_3
       
        hyps = self.gp_model.hyps
        cutoffs = self.gp_model.cutoffs
        self.mgp_model = MappedGaussianProcess(hyps, cutoffs,
                        grid_params, struc_params, mean_only,
                        container_only, self.gp_model, lmp_file_name)



#from ase import io
#from ase.calculators.espresso import Espresso
#def read_results(self):
#    output = io.read(self.label + '.pwo', parallel=False)
#    self.calc = output.calc
#    self.results = output.calc.results
#
#Espresso.read_results = read_results
