import json
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from typing import List, Union, Tuple
import warnings
from ase.calculators.singlepoint import SinglePointCalculator
from flare.atoms import FLARE_Atoms
from flare.utils import NumpyEncoder

from mpi4py import MPI
from memory_profiler import profile

from flare.bffs.sgp._C_flare import ParallelSGP
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.calculator import SGP_Calculator, sort_variances
from flare.bffs.sgp.utils import convert_to_flarepp_structure


class ParSGP_Wrapper(SGP_Wrapper):
    """Wrapper class used to make the C++ sparse GP object compatible with
    OTF. Methods and properties are designed to mirror the GP class."""

    def __init__(
        self,
        kernels: List,
        descriptor_calculators: List,
        cutoff: float,
        sigma_e: float,
        sigma_f: float,
        sigma_s: float,
        species_map: dict,
        variance_type: str = "SOR",
        single_atom_energies: dict = None,
        energy_training=True,
        force_training=True,
        stress_training=True,
        max_iterations=10,
        opt_method="BFGS",
        bounds=None,
    ):

        super().__init__(
            kernels=kernels,
            descriptor_calculators=descriptor_calculators,
            cutoff=cutoff,
            sigma_e=sigma_e,
            sigma_f=sigma_f,
            sigma_s=sigma_s,
            species_map=species_map,
            variance_type=variance_type,
            single_atom_energies=single_atom_energies,
            energy_training=energy_training,
            force_training=force_training,
            stress_training=stress_training,
            max_iterations=max_iterations,
            opt_method=opt_method,
            bounds=bounds,
        )
        self.sparse_gp = ParallelSGP(kernels, sigma_e, sigma_f, sigma_s)
        self.training_structures = []
        self.training_sparse_indices = [[] for i in range(len(descriptor_calculators))]

    @property
    def training_data(self):
        # global training dataset
        return self.training_structures

    @property
    def local_training_data(self):
        # local training dataset on the current process
        return self.sparse_gp.training_structures

    #@profile
    def build(
        self,
        training_strucs: List[FLARE_Atoms],
        training_sparse_indices: List[List[List[int]]],
        update,
    ):
        # Check the shape of sparse_indices
        assert (
            len(training_sparse_indices[0][0]) >= 0
        ), """Sparse indices should be a list
                [[[atom1_of_kernel1_of_struc1, ...], 
                  [atom1_of_kernel2_of_struc1, ...]],
                 [[atom1_of_kernel1_of_struc2, ...], 
                  [atom1_of_kernel2_of_struc2, ...]]]"""

        # Convert flare Structure or FLARE_Atoms to flare_pp Structure
        struc_list = []
        for structure in training_strucs:
            try:
                energy = structure.get_potential_energy()
            except AttributeError:
                energy = structure.potential_energy
            forces = structure.get_forces()

            # Convert stress order to xx, xy, xz, yy, yz, zz
            s = structure.get_stress()
            stress = None
            if s is not None:
                if len(s) == 6:
                    stress = -s[[0, 5, 4, 1, 3, 2]]
                elif s.shape == (3, 3):
                    stress = -np.array(
                        [s[0, 0], s[0, 1], s[0, 2], s[1, 1], s[1, 2], s[2, 2]]
                    )

            structure_descriptor = convert_to_flarepp_structure(
                structure,
                self.species_map,
                energy,
                forces,
                stress,
                self.energy_training,
                self.force_training,
                self.stress_training,
                self.single_atom_energies,
                cutoff=None,
                descriptor_calculators=None,
            )

            struc_list.append(structure_descriptor)

        n_types = len(self.species_map)

        self.sparse_gp.build(
            struc_list,
            self.cutoff,
            self.descriptor_calculators,
            training_sparse_indices,
            n_types,
            update,
        )

        self.training_structures = training_strucs
        self.training_sparse_indices = training_sparse_indices 


    #@profile
    def update_db(
        self,
        structure,
        custom_range=(),
        mode: str = "all",
    ):
        if mode == "all":
            sparse_inds = [np.arange(len(structure)).tolist() for d in self.descriptor_calculators]
        elif mode == "uncertain":
            gp_calc = SGP_Calculator(self)
            uncertainties = gp_calc.get_uncertainties(structure)
            if len(custom_range) == len(self.descriptor_calculators):
                sparse_inds = []
                for i in range(len(self.descriptor_calculators)):
                    sorted_ind = np.argsort(-uncertainties[:, i]).tolist()
                    sparse_inds.append(sorted_ind[: custom_range[i]])
            else:
                raise Exception(
                    "The custom_range should have length equal to the number of descriptors/kernels if mode='uncertain'"
                )
        elif mode == "specific":
            if len(custom_range) == len(self.descriptor_calculators):
                sparse_inds = custom_range
            else:
                raise Exception(
                    "The custom_range should have length equal to the number of descriptors/kernels if mode='specific'"
                )

        elif mode == "random":
            if len(custom_range) == len(self.descriptor_calculators):
                sparse_inds = [
                    np.random.choice(
                        len(structure), size=custom_range[i], replace=False
                    ).tolist()
                    for i in range(len(self.descriptor_calculators))
                ]
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                sparse_inds = comm.bcast(sparse_inds, root=0)
            else:
                raise Exception(
                    "The custom_range should have length equal to the number of descriptors/kernels if mode='random'"
                )
        else:
            raise NotImplementedError

        self.training_structures.append(structure)
        for k in range(len(self.descriptor_calculators)):
            self.training_sparse_indices[k].append(sparse_inds[k])

        # build a new SGP
        if len(self.training_structures) == 1:
            update = False
        else:
            update = True
        self.build(self.training_structures, self.training_sparse_indices, update=update)

    def predict_on_structures(self, struc_list):
        # convert ASE Atoms to c++ Structure with descriptors not computed
        struc_desc_list = []
        for structure in struc_list:
            structure_descriptor = convert_to_flarepp_structure(structure, self.species_map)
            struc_desc_list.append(structure_descriptor)

        struc_desc_list = self.sparse_gp.predict_on_structures(
            struc_desc_list, self.cutoff, self.descriptor_calculators
        )

        for s in range(len(struc_list)):
            results = {}
            mean_efs = deepcopy(struc_desc_list[s].mean_efs)
            results["energy"] = mean_efs[0]
            results["forces"] = mean_efs[1:-6].reshape(-1, 3)

            # Convert stress to ASE format.
            flare_stress = mean_efs[-6:]
            ase_stress = - flare_stress[[0, 3, 5, 4, 2, 1]]
            results["stress"] = ase_stress

            ## save uncertainties
            ## TODO: add "atom_indices" attribute to struc_desc_list for sort_variances()

            #n_kern = len(self.descriptor_calculators)
            #stds_full = np.zeros((len(struc_list), 3))
            #assert n_kern <= 3, NotImplementedError  # now only print out 3 components

            #for k in range(n_kern):
            #    variances = all_results[s][1][k]
            #    sorted_variances = sort_variances(struc_desc_list[s], variances)
            #    stds = np.zeros(len(sorted_variances))
            #    for n in range(len(sorted_variances)):
            #        var = sorted_variances[n]
            #        if var > 0:
            #            stds[n] = np.sqrt(var)
            #        else:
            #            stds[n] = -np.sqrt(np.abs(var))

            #    # Divide by the signal std to get a unitless value.
            #    stds_full[:, k] = stds / self.gp_model.hyps[k]

            #results["stds"] = stds_full

            if struc_list[s].calc is not None:
                struc_list[s].calc.results = results
            else:
                calc = SinglePointCalculator(struc_list[s])
                calc.results = results
                struc_list[s].calc = calc
