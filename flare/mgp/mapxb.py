import json
from flare.utils.element_coder import NumpyEncoder, element_to_Z, Z_to_element

import os, logging, warnings
import numpy as np
import multiprocessing as mp

from copy import deepcopy
from math import ceil, floor
from scipy.linalg import solve_triangular
from typing import List

from flare.env import AtomicEnvironment
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set
import flare.kernels.cutoffs as cf
from flare.gp import GaussianProcess
from flare.gp_algebra import (
    partition_vector,
    energy_force_vector_unit,
    force_energy_vector_unit,
    energy_energy_vector_unit,
    force_force_vector_unit,
    _global_training_data,
    _global_training_structures,
)
from flare.parameters import Parameters
from flare.struc import Structure

from flare.mgp.splines_methods import PCASplines, CubicSpline


class MapXbody:
    def __init__(
        self,
        grid_num: List,
        lower_bound: List or str = "auto",
        upper_bound: List or str = "auto",
        svd_rank="auto",
        coded_species: list = [],
        var_map: str = None,
        container_only: bool = True,
        lmp_file_name: str = "lmp.mgp",
        load_grid: str = None,
        lower_bound_relax: float = 0.1,
        GP: GaussianProcess = None,
        n_cpus: int = None,
        n_sample: int = 10,
        hyps_mask: dict = None,
        hyps: list = None,
        **kwargs,
    ):

        # load all arguments as attributes
        self.grid_num = np.array(grid_num)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.svd_rank = svd_rank
        self.coded_species = coded_species
        self.var_map = var_map
        self.lmp_file_name = lmp_file_name
        self.load_grid = load_grid
        self.lower_bound_relax = lower_bound_relax
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        self.spc = []
        self.maps = []
        self.kernel_info = None
        self.hyps_mask = hyps_mask
        self.hyps = hyps

        self.build_bond_struc(coded_species)

        bounds = [self.lower_bound, self.upper_bound]
        self.build_map_container(bounds)

        if (not container_only) and (GP is not None) and (len(GP.training_data) > 0):
            self.build_map(GP)

    def build_bond_struc(self, coded_species):
        raise NotImplementedError("need to be implemented in child class")

    def get_arrays(self, atom_env):
        raise NotImplementedError("need to be implemented in child class")

    def build_map_container(self, bounds):
        """
        construct an empty spline container without coefficients.
        """

        self.maps = []
        for spc in self.spc:
            m = self.singlexbody(bounds=bounds, species=spc, **self.__dict__)
            self.maps.append(m)

    def build_map(self, GP):
        """
        generate/load grids and get spline coefficients
        """

        self.kernel_info = get_kernel_term(self.kernel_name, GP.hyps_mask, GP.hyps)
        self.hyps_mask = GP.hyps_mask
        self.hyps = GP.hyps

        for m in self.maps:
            m.build_map(GP)

    def predict(self, atom_env):
        f_spcs = np.zeros(3)
        vir_spcs = np.zeros(6)
        v_spcs = 0
        e_spcs = 0
        kern = 0

        if len(atom_env.bond_array_2) == 0:
            return f_spcs, vir_spcs, kern, v_spcs, e_spcs

        en_kernel, cutoffs, hyps, hyps_mask = self.kernel_info

        args = from_mask_to_args(hyps, cutoffs, hyps_mask)

        if self.var_map == "pca":
            kern = en_kernel(atom_env, atom_env, *args)

        spcs, comp_r, comp_xyz = self.get_arrays(atom_env)

        # predict for each species
        rebuild_spc = []
        new_bounds = []
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = self.find_map_index(spc)
            try:
                f, vir, v, e = self.maps[map_ind].predict(lengths, xyzs)
                f_spcs += f
                vir_spcs += vir
                v_spcs += v
                e_spcs += e
            except ValueError as err_msg:
                rebuild_spc.append(err_msg.args[0])
                new_bounds.append(err_msg.args[1])

        if len(rebuild_spc) > 0:
            raise ValueError(
                rebuild_spc,
                new_bounds,
                f"The {self.kernel_name} map needs re-constructing.",
            )

        return f_spcs, vir_spcs, kern, v_spcs, e_spcs

    def as_dict(self) -> dict:
        """
        Dictionary representation of the MGP model.
        """

        out_dict = deepcopy(dict(vars(self)))
        out_dict.pop("kernel_info")

        # only save the mean coefficients and var if var_map == 'simple'
        if self.var_map == 'simple':
            out_dict["maps"] = [[m.mean.__coeffs__ for m in self.maps],
                                [m.var.__coeffs__ for m in self.maps]]
        else:
            out_dict["maps"] = [[m.mean.__coeffs__ for m in self.maps]]
            if self.var_map == 'pca':
                warnings.warn("var_map='pca' is too heavy to dump, change to var_map=None")
                out_dict["var_map"] = None

        out_dict["bounds"] = [m.bounds for m in self.maps]

        # rm keys since they are built in the __init__ function
        key_list = ["singlexbody", "spc"]
        for key in key_list:
            if out_dict.get(key) is not None:
                del out_dict[key]

        return out_dict

    @staticmethod
    def from_dict(dictionary: dict, mapxbody):
        """
        Create MGP object from dictionary representation.
        """

        if "container_only" not in dictionary:
            dictionary["container_only"] = True

        new_mgp = mapxbody(**dictionary)

        # Restore kernel_info
        new_mgp.kernel_info = get_kernel_term(
            dictionary["kernel_name"], dictionary["hyps_mask"], dictionary["hyps"]
        )

        # Fill up the model with the saved coeffs
        for m in range(len(new_mgp.maps)):
            singlexb = new_mgp.maps[m]
            bounds = dictionary["bounds"][m]
            singlexb.set_bounds(bounds[0], bounds[1])
            singlexb.build_map_container()
            singlexb.mean.__coeffs__ = np.array(dictionary["maps"][0][m])
            if new_mgp.var_map == 'simple':
                singlexb.var.__coeffs__ = np.array(dictionary["maps"][1][m])

        return new_mgp


    def write(self, f, write_var):
        for m in self.maps:
            m.write(f, write_var)


class SingleMapXbody:
    def __init__(
        self,
        grid_num: int = 1,
        bounds="auto",
        species: list = [],
        svd_rank=0,
        var_map: str = None,
        load_grid=None,
        lower_bound_relax=0.1,
        n_cpus: int = None,
        n_sample: int = 100,
        **kwargs,
    ):

        self.grid_num = grid_num
        self.bounds = deepcopy(bounds)
        self.species = species
        self.svd_rank = svd_rank
        self.var_map = var_map
        self.load_grid = load_grid
        self.lower_bound_relax = lower_bound_relax
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        self.auto_lower = bounds[0] == "auto"
        if self.auto_lower:
            lower_bound = None
        else:
            lower_bound = bounds[0]

        self.auto_upper = bounds[1] == "auto"
        if self.auto_upper:
            upper_bound = None
        else:
            upper_bound = bounds[1]

        self.set_bounds(lower_bound, upper_bound)

        self.hyps_mask = None

        if not self.auto_lower and not self.auto_upper:
            self.build_map_container()

    def set_bounds(self, lower_bound, upper_bound):
        raise NotImplementedError("need to be implemented in child class")

    def construct_grids(self):
        raise NotImplementedError("need to be implemented in child class")

    def LoadGrid(self):
        if "mgp_grids" not in os.listdir(self.load_grid):
            raise FileNotFoundError(
                "Please set 'load_grid' as the location of mgp_grids folder"
            )

        grid_path = f"{self.load_grid}/mgp_grids/{self.bodies}_{self.species_code}"
        grid_mean = np.load(f"{grid_path}_mean.npy")
        grid_vars = np.load(f"{grid_path}_var.npy", allow_pickle=True)
        return grid_mean, grid_vars

    def GenGrid(self, GP):
        """
        To use GP to predict value on each grid point, we need to generate the
        kernel vector kv whose length is the same as the training set size.

        1. We divide the training set into several batches, corresponding to
           different segments of kv
        2. Distribute each batch to a processor, i.e. each processor calculate
           the kv segment of one batch for all grids
        3. Collect kv segments and form a complete kv vector for each grid,
           and calculate the grid value by multiplying the complete kv vector
           with GP.alpha
        """

        if self.load_grid is not None:
            return self.LoadGrid()

        if self.n_cpus is None:
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus

        # -------- get training data info ----------
        n_envs = len(GP.training_data)
        n_strucs = len(GP.training_structures)

        if (n_envs == 0) and (n_strucs == 0):
            warnings.warn("No training data, will return 0")
            return np.zeros([n_grid]), None

        # ------ construct grids ------
        n_grid = np.prod(self.grid_num)
        grid_mean = np.zeros([n_grid])
        if self.var_map is not None:
            grid_vars = np.zeros([n_grid, len(GP.alpha)])
        else:
            grid_vars = None

        # ------- call gengrid functions ---------------
        kernel_info = get_kernel_term(self.kernel_name, GP.hyps_mask, GP.hyps)
        args = [GP.name, kernel_info]

        k12_v_force = self._gengrid_par(args, True, n_envs, processes)
        k12_v_energy = self._gengrid_par(args, False, n_strucs, processes)

        k12_v_all = np.hstack([k12_v_force, k12_v_energy])
        del k12_v_force
        del k12_v_energy

        # ------- compute bond means and variances ---------------
        grid_mean = k12_v_all @ GP.alpha
        grid_mean = np.reshape(grid_mean, self.grid_num)

        if self.var_map is not None:
            grid_vars = solve_triangular(GP.l_mat, k12_v_all.T, lower=True).T

            if self.var_map == "simple":
                self_kern = self._gengrid_var_simple(kernel_info)
                grid_vars = np.sqrt(self_kern - np.sum(grid_vars ** 2, axis=1))
                grid_vars = np.expand_dims(grid_vars, axis=1)

            tensor_shape = np.array([*self.grid_num, grid_vars.shape[1]])
            grid_vars = np.reshape(grid_vars, tensor_shape)

        # ------ save mean and var to file -------
        if "mgp_grids" not in os.listdir("./"):
            os.mkdir("mgp_grids")

        grid_path = f"mgp_grids/{self.bodies}_{self.species_code}"
        np.save(f"{grid_path}_mean", grid_mean)
        np.save(f"{grid_path}_var", grid_vars)

        return grid_mean, grid_vars

    def _gengrid_par(self, args, force_block, n_envs, processes):

        if n_envs == 0:
            n_grid = np.prod(self.grid_num)
            return np.empty((n_grid, 0))

        gengrid_func = self._gengrid_inner

        if processes == 1:
            return gengrid_func(*args, force_block, 0, n_envs)

        with mp.Pool(processes=processes) as pool:

            block_id, nbatch = partition_vector(self.n_sample, n_envs, processes)

            k12_slice = []
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_slice.append(
                    pool.apply_async(gengrid_func, args=args + [force_block, s, e])
                )
            k12_matrix = []
            for ibatch in range(nbatch):
                k12_matrix += [k12_slice[ibatch].get()]
            pool.close()
            pool.join()
        del k12_slice
        k12_v_force = np.hstack(k12_matrix)
        del k12_matrix

        return k12_v_force

    def _gengrid_inner(self, name, kernel_info, force_block, s, e):
        """
        Loop over different parts of the training set. from element s to element e

        Args:
            name: name of the gp instance
            s: start index of the training data parition
            e: end index of the training data parition
            kernel_info: return value of the get_3b_kernel
        """

        _, cutoffs, hyps, hyps_mask = kernel_info

        r_cut = cutoffs[self.kernel_name]

        n_grids = np.prod(self.grid_num)

        if np.any(np.array(self.bounds[1]) <= 0.0):
            if force_block:
                return np.zeros((n_grids, (e-s)*3))
            else:
                return np.zeros((n_grids, e-s))
 
        grids = self.construct_grids()
        coords = np.zeros(
            (grids.shape[0], self.grid_dim * 3), dtype=np.float64
        )  # padding 0
        coords[:, 0] = np.ones_like(coords[:, 0])

        fj, fdj = self.grid_cutoff(
            grids, r_cut, coords, derivative=True, cutoff_func=cf.quadratic_cutoff
        )
        fdj = fdj[:, [0]]

        if force_block:
            training_data = _global_training_data[name]
            kern_type = f"energy_force"
        else:
            training_data = _global_training_structures[name]
            kern_type = f"energy_energy"

        k_v = []
        chunk_size = 32 ** 3
        if n_grids > chunk_size:
            n_chunk = ceil(n_grids / chunk_size)
        else:
            n_chunk = 1

        for m_index in range(s, e):
            data = training_data[m_index]
            kern_vec = []
            for g in range(n_chunk):
                gs = chunk_size * g
                ge = np.min((chunk_size * (g + 1), n_grids))
                grid_chunk = grids[gs:ge, :]
                fj_chunk = fj[gs:ge, :]
                fdj_chunk = fdj[gs:ge, :]
                kv_chunk = self.get_grid_kernel(
                    kern_type, data, kernel_info, grid_chunk, fj_chunk, fdj_chunk,
                )
                kern_vec.append(kv_chunk)
            kern_vec = np.hstack(kern_vec)
            k_v.append(kern_vec)

        if len(k_v) > 0:
            k_v = np.vstack(k_v).T
        else:
            k_v = np.zeros((n_grids, 0))

        return k_v

    def _gengrid_var_simple(self, kernel_info):
        """
        Generate grids for variance upper bound, based on the inequality:
        V(c, p)^2 <= V(c, c) V(p, p)
        where c, p are two bonds/triplets or environments
        """

        _, cutoffs, hyps, hyps_mask = kernel_info

        r_cut = cutoffs[self.kernel_name]

        grids = self.construct_grids()
        coords = np.zeros(
            (grids.shape[0], self.grid_dim * 3), dtype=np.float64
        )  # padding 0
        coords[:, 0] = np.ones_like(coords[:, 0])

        fj, fdj = self.grid_cutoff(
            grids, r_cut, coords, derivative=True, cutoff_func=cf.quadratic_cutoff
        )
        fdj = fdj[:, [0]]

        return self.get_self_kernel(kernel_info, grids, fj, fdj)

    def build_map_container(self):
        """
        build 1-d spline function for mean, 2-d for var
        """
        if np.any(np.array(self.bounds[1]) <= 0.0):
            bounds = [np.zeros_like(self.bounds[0]), np.ones_like(self.bounds[1])]
        else:
            bounds = self.bounds

        self.mean = CubicSpline(bounds[0], bounds[1], orders=self.grid_num)

        if self.var_map == "pca":
            if self.svd_rank == "auto":
                warnings.warn(
                    "The containers for variance are not built because svd_rank='auto'"
                )

            elif isinstance(self.svd_rank, int):
                self.var = PCASplines(
                    bounds[0],
                    bounds[1],
                    orders=self.grid_num,
                    svd_rank=self.svd_rank,
                )

        if self.var_map == "simple":
            self.var = CubicSpline(bounds[0], bounds[1], orders=self.grid_num)

    def update_bounds(self, GP):
        rebuild_container = False

        # double check the container and the GP is consistent
        if not Parameters.compare_dict(GP.hyps_mask, self.hyps_mask):
            rebuild_container = True

        lower_bound = self.bounds[0]
        min_dist = self.search_lower_bound(GP)
        # change lower bound only when there appears a smaller distance
        if lower_bound is None or min_dist < np.max(lower_bound):
            lower_bound = np.max((min_dist - self.lower_bound_relax, 0.0))
            rebuild_container = True

            warnings.warn(
                f"The minimal distance in training data is lower than "
                f"the current lower bound, will reset lower bound to {lower_bound}"
            )

        upper_bound = self.bounds[1]
        if self.auto_upper or upper_bound is None:
            gp_cutoffs = Parameters.get_cutoff(
                self.kernel_name, self.species, GP.hyps_mask
            )
            if upper_bound is None or np.any(gp_cutoffs > upper_bound):
                upper_bound = gp_cutoffs
                rebuild_container = True

        if rebuild_container:
            self.set_bounds(lower_bound, upper_bound)
            self.build_map_container()

    def build_map(self, GP):

        self.update_bounds(GP)

        y_mean, y_var = self.GenGrid(GP)
        self.mean.set_values(y_mean)

        if self.var_map == "pca" and self.svd_rank == "auto":
            self.var = PCASplines(
                self.bounds[0],
                self.bounds[1],
                orders=self.grid_num,
                svd_rank=np.min(y_var.shape),
            )

        if self.var_map is not None:
            self.var.set_values(y_var)

        self.hyps_mask = deepcopy(GP.hyps_mask)

    def __str__(self):
        info = f"""{self.__class__.__name__}
        species: {self.species}
        lower bound: {self.bounds[0]}, auto_lower = {self.auto_lower}
        upper bound: {self.bounds[1]}, auto_upper = {self.auto_upper}
        grid num: {self.grid_num}
        lower bound relaxation: {self.lower_bound_relax}
        load grid from: {self.load_grid}\n"""

        if self.var_map is None:
            info += f"        without variance\n"
        elif self.var_map == "pca":
            info += f"        with PCA variance, svd_rank = {self.svd_rank}\n"
        elif self.var_map == "simple":
            info += f"        with simple variance"

        return info

    def search_lower_bound(self, GP):
        """
        If the lower bound is set to be 'auto', search the minimal interatomic
        distances in the training set of GP.
        """
        upper_bound = Parameters.get_cutoff(
            self.kernel_name, self.species, GP.hyps_mask
        )

        lower_bound = np.min(upper_bound)
        for env in _global_training_data[GP.name]:
            if len(env.bond_array_2) == 0:
                continue

            min_dist = env.bond_array_2[0][0]
            if min_dist < lower_bound:
                lower_bound = min_dist

        for struc in _global_training_structures[GP.name]:
            for env in struc:
                if len(env.bond_array_2) == 0:
                    continue

                min_dist = env.bond_array_2[0][0]
                if min_dist < lower_bound:
                    lower_bound = min_dist

        return lower_bound

    def predict(self, lengths, xyzs):
        """
        predict force and variance contribution of one component
        """

        min_dist = np.min(lengths)
        if min_dist < np.max(self.bounds[0]):
            raise ValueError(
                self.species,
                min_dist,
                f"The minimal distance {min_dist:.3f}"
                f" is below the mgp lower bound {self.bounds[0]}",
            )
        
        max_dist = np.max(lengths)
        if max_dist > np.min(self.bounds[1]):
            raise Exception(
                self.species,
                max_dist,
                f"The atomic environment should have cutoff smaller"
                f" than the GP cutoff"
            )

        lengths = np.array(lengths)
        xyzs = np.array(xyzs)

        n_neigh = self.bodies - 1
        # predict forces and energy
        e_0, f_0 = self.mean(lengths, with_derivatives=True)
        e = np.sum(e_0)  # energy
        f_d = np.zeros((lengths.shape[0], n_neigh, 3))
        for b in range(n_neigh):
            f_d[:, b, :] = np.diag(f_0[:, b, 0]) @ xyzs[:, b]
        f = self.bodies * np.sum(f_d, axis=(0, 1))

        # predict var
        v = 0
        if self.var_map == "simple":
            v_0 = self.var(lengths)
            v = np.sum(v_0)
        elif self.var_map == "pca":
            v_0 = self.var(lengths)
            v_0 = np.sum(v_0, axis=1)
            v_0 = np.expand_dims(v_0, axis=1)
            v = self.var.V @ v_0

        # predict virial stress
        vir = np.zeros(6)
        vir_order = (
            (0, 0),
            (1, 1),
            (2, 2),
            (1, 2),
            (0, 2),
            (0, 1),
        )  # match the ASE order
        for i in range(6):
            for b in range(n_neigh):
                vir_i = (
                    f_d[:, b, vir_order[i][0]]
                    * xyzs[:, b, vir_order[i][1]]
                    * lengths[:, b]
                )
                vir[i] += np.sum(vir_i)

        vir *= self.bodies / 2
        return f, vir, v, e


    def write(self, f, write_var, permute=False):
        """ 
        Write LAMMPS coefficient file

        This implementation only works for 2b and 3b. User should
        implement overload in the actual class if the new kernel
        has different coefficient format

        In the future, it should be changed to writing in bin/hex
        instead of decimal
        """

        # write header
        elems = self.species_code.split("_")

        a = self.bounds[0]
        b = self.bounds[1]
        order = self.grid_num

        header = " ".join(elems)
        header += " " + " ".join(map(repr, a))
        header += " " + " ".join(map(repr, b))
        header += " " + " ".join(map(str, order))
        f.write(header + "\n")

        # write coeffs
        if write_var:
            coefs = self.var.__coeffs__
        else:
            coefs = self.mean.__coeffs__

        self.write_flatten_coeff(f, coefs)


    def write_flatten_coeff(self, f, coefs):
        """
        flatten the coefficient and write it as
        a block. each line has no more than 5 element.
        the accuracy is restricted to .10
        """
        coefs = coefs.reshape([-1])
        for c, coef in enumerate(coefs):
            f.write(" " + repr(coef))
            if c % 5 == 4 and c != len(coefs) - 1:
                f.write("\n")
        f.write("\n")


# -----------------------------------------------------------------------------
#                               Functions
# -----------------------------------------------------------------------------


def get_kernel_term(kernel_name, hyps_mask, hyps):
    hyps, cutoffs, hyps_mask = Parameters.get_component_mask(
        hyps_mask, kernel_name, hyps=hyps
    )
    kernel, _, ek, efk, _, _, _ = str_to_kernel_set([kernel_name], "mc", hyps_mask)
    return (ek, cutoffs, hyps, hyps_mask)
