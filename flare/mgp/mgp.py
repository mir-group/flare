'''
:class:`MappedGaussianProcess` uses splines to build up interpolation\
function of the low-dimensional decomposition of Gaussian Process, \
with little loss of accuracy. Refer to \
`Vandermause et al. <https://www.nature.com/articles/s41524-020-0283-z>`_, \
`Glielmo et al. <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.184307>`_
'''
import time, os, math, inspect, subprocess, json, warnings, pickle
import numpy as np
import multiprocessing as mp

from copy import deepcopy
from typing import List

from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.kernels.utils import str_to_kernel_set
from flare.utils.element_coder import NumpyEncoder

from flare.mgp.map2b import Map2body
from flare.mgp.map3b import Map3body
from flare.mgp.utils import str_to_mapped_kernel

class MappedGaussianProcess:
    '''
    Build Mapped Gaussian Process (MGP)
    and automatically save coefficients for LAMMPS pair style.

    Args:
        grid_params (dict): Parameters for the mapping itself, such as
            grid size of spline fit, etc. As described below.
        species_list (dict): List of all the (unique) species included during
            the training that need to be mapped
        map_force (bool): if True, do force mapping; otherwise do energy mapping,
            default is False
        mean_only (bool): if True: only build mapping for mean (force)
        container_only (bool): if True: only build splines container
            (with no coefficients); if False: Attempt to build map immediately
        GP (GaussianProcess): None or a GaussianProcess object. If a GP is input,
            and container_only is False, automatically build a mapping corresponding
            to the GaussianProcess.
        lmp_file_name (str): LAMMPS coefficient file name

    For `grid_params`, please set up the following keys and values
    Args:
        'two_body': if 2-body is present, set as a dictionary of parameters 
                    for 2-body mapping

    Examples:

    >>> grid_params = {'twobody': {'grid_num': [64]},
    ...                'threebody': {'grid_num': [64, 64, 64]}}
    '''

    def __init__(self,
                 grid_params: dict,
                 species_list: list=[],
                 map_force: bool=False,
                 GP: GaussianProcess=None,
                 mean_only: bool=False,
                 container_only: bool=True,
                 lmp_file_name: str='lmp.mgp',
                 n_cpus: int=None,
                 n_sample: int=100):

        # load all arguments as attributes
        self.map_force = map_force
        self.mean_only = mean_only
        self.lmp_file_name = lmp_file_name
        self.n_cpus = n_cpus
        self.n_sample = n_sample
        self.grid_params = grid_params
        self.species_list = species_list
        self.hyps_mask = None
        self.cutoffs = None

        if (GP is not None):
            self.hyps_mask = GP.hyps_mask
            self.cutoffs = GP.cutoffs

        if 'load_grid' not in grid_params.keys():
            grid_params['load_grid'] = None
        if 'update' not in grid_params.keys():
            grid_params['update'] = False
        if 'lower_bound_relax' not in grid_params.keys():
            grid_params['lower_bound_relax'] = 0.1

        self.maps = {}
        args = [species_list, map_force, GP, mean_only,\
                container_only, lmp_file_name, \
                grid_params['load_grid'], grid_params['update'],\
                grid_params['lower_bound_relax'],
                n_cpus, n_sample]

        optional_xb_params = ['lower_bound', 'upper_bound', 'svd_rank']
        for key in grid_params.keys():
            if 'body' in key:
                if 'twobody' == key:
                    mapxbody = Map2body
                elif 'threebody' == key:
                    mapxbody = Map3body
                else:
                    raise KeyError("Only 'twobody' & 'threebody' are allowed")

                xb_dict = grid_params[key]

                # set to 'auto' if the param is not given
                for oxp in optional_xb_params:
                    if oxp not in xb_dict.keys():
                        xb_dict[oxp] = 'auto'

                xb_args = [xb_dict['grid_num'], xb_dict['lower_bound'],
                           xb_dict['upper_bound'], xb_dict['svd_rank']]
                xb_maps = mapxbody(xb_args + args)
                self.maps[key] = xb_maps

        self.mean_only = mean_only

    def build_map(self, GP):

        self.hyps_mask = GP.hyps_mask
        self.cutoffs = GP.cutoffs

        for xb in self.maps:
            self.maps[xb].build_map(GP)

        # write to lammps pair style coefficient file
        self.write_lmp_file(self.lmp_file_name)


    def predict(self, atom_env: AtomicEnvironment, mean_only: bool = False,
            ) -> (float, 'ndarray', 'ndarray', float):
        '''
        predict force, variance, stress and local energy for given
        atomic environment

        Args:
            atom_env: atomic environment (with a center atom and its neighbors)
            mean_only: if True: only predict force (variance is always 0)

        Return:
            force: 3d array of atomic force
            variance: 3d array of the predictive variance
            stress: 6d array of the virial stress
            energy: the local energy (atomic energy)
        '''

        if self.mean_only:  # if not build mapping for var
            mean_only = True

        force = virial = kern = v = energy = 0
        for xb in self.maps:
            pred = self.maps[xb].predict(atom_env, mean_only)
            force += pred[0]
            virial += pred[1]
            kern += pred[2]
            v += pred[3]
            energy += pred[4]

        variance = kern - np.sum(v**2, axis=0)

        return force, variance, virial, energy


    def write_lmp_file(self, lammps_name):
        '''
        write the coefficients to a file that can be used by lammps pair style
        '''

        f = open(lammps_name, 'w')

        # write header
        header_comment = '''# #2bodyarray #3bodyarray\n# elem1 elem2 a b order\n\n'''
        f.write(header_comment)
        header = ''
        xbodies = ['twobody', 'threebody']
        for xb in xbodies:
            if xb in self.maps.keys():
                num = len(self.maps[xb].maps)
            else:
                num = 0
            header += f'{num} '
        f.write(header + '\n')

        # write coefficients
        for xb in self.maps.keys():
            self.maps[xb].write(f)

        f.close()

    def as_dict(self) -> dict:
        """
        Dictionary representation of the MGP model.
        """

        out_dict = deepcopy(dict(vars(self)))

        # Uncertainty mappings currently not serializable;
        if not self.mean_only:
            warnings.warn("Uncertainty mappings cannot be serialized, "
                          "and so the MGP dict outputted will not have "
                          "them.", Warning)
            out_dict['mean_only'] = True

        # Iterate through the mappings for various bodies
        for i in self.bodies:
            kern_info = f'kernel{i}b_info'
            kernel, ek, efk, cutoffs, hyps, hyps_mask = out_dict[kern_info]
            out_dict[kern_info] = (kernel.__name__, efk.__name__,
                                   cutoffs, hyps, hyps_mask)

        # only save the coefficients
        out_dict['maps_2'] = [map_2.mean.__coeffs__ for map_2 in self.maps_2]
        out_dict['maps_3'] = [map_3.mean.__coeffs__ for map_3 in self.maps_3]

        # don't need these since they are built in the __init__ function
        key_list = ['spcs_set', ]
        for key in key_list:
            if out_dict.get(key) is not None:
                del out_dict[key]

        return out_dict

    @staticmethod
    def from_dict(dictionary: dict):
        """
        Create MGP object from dictionary representation.
        """
        new_mgp = MappedGaussianProcess(grid_params=dictionary['grid_params'],
                                        species_list=dictionary['species_list'],
                                        map_force=dictionary['map_force'],
                                        GP=None,
                                        mean_only=dictionary['mean_only'],
                                        container_only=True,
                                        lmp_file_name=dictionary['lmp_file_name'],
                                        n_cpus=dictionary['n_cpus'],
                                        n_sample=dictionary['n_sample'])

        # Restore kernel_info
        for i in dictionary['bodies']:
            kern_info = f'kernel{i}b_info'
            hyps_mask = dictionary[kern_info][-1]

            kernel_info = dictionary[kern_info]
            kernel_name = kernel_info[0]
            kernel, _, ek, efk = str_to_kernel_set([kernel_name], 'mc', hyps_mask)
            kernel_info[0] = kernel
            kernel_info[1] = ek
            kernel_info[2] = efk
            setattr(new_mgp, kern_info, kernel_info)

        # Fill up the model with the saved coeffs
        for m, map_2 in enumerate(new_mgp.maps_2):
            map_2.mean.__coeffs__ = np.array(dictionary['maps_2'][m])
        for m, map_3 in enumerate(new_mgp.maps_3):
            map_3.mean.__coeffs__ = np.array(dictionary['maps_3'][m])

        # Set GP
        if dictionary.get('GP'):
            new_mgp.GP = GaussianProcess.from_dict(dictionary.get("GP"))

        return new_mgp

    def write_model(self, name: str, format='json'):
        """
        Write everything necessary to re-load and re-use the model
        :param model_name:
        :return:
        """
        if 'json' in format.lower():
            with open(f'{name}.json', 'w') as f:
                json.dump(self.as_dict(), f, cls=NumpyEncoder)

        elif 'pickle' in format.lower() or 'binary' in format.lower():
            with open(f'{name}.pickle', 'wb') as f:
                pickle.dump(self, f)

        else:
            raise ValueError("Requested format not found.")



    @staticmethod
    def from_file(filename: str):
        if '.json' in filename:
            with open(filename, 'r') as f:
                model = \
                    MappedGaussianProcess.from_dict(json.loads(f.readline()))
            return model

        elif 'pickle' in filename:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise NotImplementedError


