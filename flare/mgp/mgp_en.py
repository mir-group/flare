import time, os, subprocess, math, inspect
import numpy as np
import multiprocessing as mp

from scipy.linalg import solve_triangular

import flare.kernels.mc_simple as mc
import flare.mgp.utils as utils

from flare import gp, struc, kernels
from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.cutoffs import quadratic_cutoff
from flare.mgp.utils import get_bonds, get_triplets, get_triplets_en
#     self_two_body_mc_jit, self_three_body_mc_jit
from flare.mgp.splines_methods import PCASplines, CubicSpline


class MappedGaussianProcess:
    '''
    Build Mapped Gaussian Process (MGP) and automatically save coefficients for LAMMPS pair style.
    :param: hyps: GP hyps
    :param: cutoffs: GP cutoffs
    :param: struc_params : information of training data
    :param: grid_params : setting of grids for mapping
    :param: mean_only : if True: only build mapping for mean (force)
    :param: container_only : if True: only build splines container (with no coefficients)
    :param: GP: None or a GaussianProcess object. If input a GP, then build mapping when creating MappedGaussianProcess object
    :param: lmp_file_name : lammps coefficient file name
    Examples:

    >>> struc_params = {'species': [0, 1],
                        'cube_lat': cell, # should input the cell matrix
                        'mass_dict': {'0': 27 * unit, '1': 16 * unit}}
    >>> grid_params =  {'bounds_2': [[1.2], [3.5]],
                                    # [[lower_bound], [upper_bound]]
                        'bounds_3': [[1.2, 1.2, 0], [3.5, 3.5, np.pi]],
                                    # [[lower,lower,0],[upper,upper,np.pi]]
                        'grid_num_2': 64,
                        'grid_num_3': [16, 16, 16],
                        'svd_rank_2': 64,
                        'svd_rank_3': 16**3,
                        'update': True, # if True: accelerating grids
                                        # generating by saving intermediate
                                        # coeff when generating grids
                        'load_grid': None}
    '''

    def __init__(self,
                 grid_params: dict,
                 struc_params: dict,
                 GP=None,
                 mean_only=False,
                 container_only=True,
                 lmp_file_name='lmp.mgp',
                 n_cpus=None,
                 nsample=100):

        # get all arguments as attributes
        arg_dict = inspect.getargvalues(inspect.currentframe())[3]
        del arg_dict['self'], arg_dict['GP']
        self.__dict__.update(arg_dict)
        self.__dict__.update(grid_params)

        self.hyps = GP.hyps
        self.cutoffs = GP.cutoffs
        self.kernel_name = GP.kernel_name
        self.bodies = []
        if "two" in GP.kernel_name:
            self.bodies.append(2)
        if "three" in GP.kernel_name:
            self.bodies.append(3)

        self.build_bond_struc(struc_params)
        self.maps_2 = []
        self.maps_3 = []
        self.build_map_container()

        if not container_only and (GP is not None) and (len(GP.training_data) > 0):
            self.build_map(GP)

    def build_map_container(self):
        '''
        construct an empty spline container without coefficients
        '''
        if 2 in self.bodies:
            for b_struc in self.bond_struc[0]:
                map_2 = Map2body(self.grid_num_2, self.bounds_2,
                                 b_struc, self.svd_rank_2,
                                 self.mean_only, self.n_cpus, self.nsample)
                self.maps_2.append(map_2)
        if 3 in self.bodies:
            for b_struc in self.bond_struc[1]:
                map_3 = Map3body(self.grid_num_3, self.bounds_3,
                                 b_struc, self.svd_rank_3,
                                 self.mean_only,
                                 self.grid_params['load_grid'],
                                 self.update)
                self.maps_3.append(map_3)

    def build_map(self, GP):
        '''
        generate/load grids and get spline coefficients
        '''
        for map_2 in self.maps_2:
            map_2.build_map(GP)
        for map_3 in self.maps_3:
            map_3.build_map(GP)

        # write to lammps pair style coefficient file
        self.write_lmp_file(self.lmp_file_name)

    def build_bond_struc(self, struc_params):

        '''
        build a bond structure, used in grid generating
        '''

        cutoff = np.min(self.cutoffs)
        cell = struc_params['cube_lat']
        mass_dict = struc_params['mass_dict']
        species_list = struc_params['species']
        N_spc = len(species_list)

        # ------------------- 2 body (2 atoms (1 bond) config) ---------------
        bodies = 2
        bond_struc_2 = []
        spc_2 = []
        for spc1_ind, spc1 in enumerate(species_list):
            for spc2 in species_list[spc1_ind:]:
                species = [spc1, spc2]
                spc_2.append(species)
                positions = [[(i+1)/(bodies+1)*cutoff, 0, 0]
                             for i in range(bodies)]
                spc_struc = \
                    struc.Structure(cell, species, positions, mass_dict)
                spc_struc.coded_species = np.array(species)
                bond_struc_2.append(spc_struc)

        # ------------------- 3 body (3 atoms (1 triplet) config) -------------
        bodies = 3
        bond_struc_3 = []
        spc_3 = []
        for spc1_ind in range(N_spc):
            spc1 = species_list[spc1_ind]
            for spc2_ind in range(N_spc):  # (spc1_ind, N_spc):
                spc2 = species_list[spc2_ind]
                for spc3_ind in range(N_spc):  # (spc2_ind, N_spc):
                    spc3 = species_list[spc3_ind]
                    species = [spc1, spc2, spc3]
                    spc_3.append(species)
                    positions = [[(i+1)/(bodies+1)*cutoff, 0, 0]
                                 for i in range(bodies)]
                    spc_struc = struc.Structure(cell, species, positions,
                                                mass_dict)
                    spc_struc.coded_species = np.array(species)
                    bond_struc_3.append(spc_struc)
#                    if spc1 != spc2:
#                        species = [spc2, spc3, spc1]
#                        spc_3.append(species)
#                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                    for i in range(bodies)]
#                        spc_struc = struc.Structure(cell, species, positions,
#                                                    mass_dict)
#                        spc_struc.coded_species = np.array(species)
#                        bond_struc_3.append(spc_struc)
#                    if spc2 != spc3:
#                        species = [spc3, spc1, spc2]
#                        spc_3.append(species)
#                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                    for i in range(bodies)]
#                        spc_struc = struc.Structure(cell, species, positions,
#                                                    mass_dict)
#                        spc_struc.coded_species = np.array(species)
#                        bond_struc_3.append(spc_struc)

        self.bond_struc = [bond_struc_2, bond_struc_3]
        self.spcs = [spc_2, spc_3]

    def predict(self, atom_env: AtomicEnvironment, mean_only: bool=False):
        '''
        predict force and variance for given atomic environment
        :param atom_env: atomic environment (with a center atom and its neighbors)
        :param mean_only: if True: only predict force (variance is always 0)
        '''
        if self.mean_only:  # if not build mapping for var
            mean_only = True

        # ---------------- predict for two body -------------------
        f2 = vir2 = kern2 = v2 = e2 = 0
        if 2 in self.bodies:
            sig2, ls2 = self.hyps[:2]
            r_cut2 = self.cutoffs[0]

            f2, vir2, kern2, v2, e2 = \
                self.predict_multicomponent(atom_env, sig2, ls2, r_cut2,
                                            self.get_2body_comp, self.spcs[0],
                                            self.maps_2, mean_only)

        # ---------------- predict for three body -------------------
        f3 = vir3 = kern3 = v3 = e3 = 0
        if 3 in self.bodies:
            sig3, ls3, _ = self.hyps[-3:]
            r_cut3 = self.cutoffs[1]

            f3, vir3, kern3, v3, e3 = \
                self.predict_multicomponent(atom_env, sig3, ls3, r_cut3,
                                            self.get_3body_comp, self.spcs[1],
                                            self.maps_3, mean_only)

        f = f2 + f3
        vir = vir2 + vir3
        v = kern2 + kern3 - np.sum((v2 + v3)**2, axis=0)
        e = e2 + e3

        return f, v, vir, e

    def get_2body_comp(self, atom_env, sig, ls, r_cut):
        '''
        get bonds grouped by species
        '''
        bond_array_2 = atom_env.bond_array_2
        ctype = atom_env.ctype
        etypes = atom_env.etypes

        kern2 = np.zeros(3)
        for d in range(3):
            kern2[d] = \
                self_two_body_mc_jit(bond_array_2, ctype, etypes, d+1, sig, ls,
                                     r_cut, quadratic_cutoff)

        spcs, comp_r, comp_xyz = get_bonds(ctype, etypes, bond_array_2)
        return kern2, spcs, comp_r, comp_xyz

    def get_3body_comp(self, atom_env, sig, ls, r_cut):
        '''
        get triplets and grouped by species
        '''
        bond_array_3 = atom_env.bond_array_3
        cross_bond_inds = atom_env.cross_bond_inds
        cross_bond_dists = atom_env.cross_bond_dists
        triplets = atom_env.triplet_counts
        ctype = atom_env.ctype
        etypes = atom_env.etypes

#        kern3 = np.zeros(3)
#        for d in range(3):
#            kern3[d] = self_three_body_mc_jit(bond_array_3, cross_bond_inds,
#                    cross_bond_dists, triplets, ctype, etypes, d+1, sig, ls,
#                    r_cut, quadratic_cutoff)

        kern3_gp = np.zeros(3)
        for d in range(3):
            kern3_gp[d] = mc.three_body_mc(atom_env, atom_env, d+1, d+1,
                                        self.hyps[-3:], self.cutoffs)

        spcs, comp_r, comp_xyz = \
            get_triplets_en(ctype, etypes, bond_array_3,
                            cross_bond_inds, cross_bond_dists, triplets)

        return kern3_gp, spcs, comp_r, comp_xyz

    def predict_multicomponent(self, atom_env, sig, ls, r_cut, get_comp,
                               spcs_list, mappings, mean_only):
        '''
        Add up results from `predict_component` to get the total contribution
        of all species
        '''
        f_spcs = 0
        vir_spcs = 0
        v_spcs = 0
        e_spcs = 0

        kern, spcs, comp_r, comp_xyz = get_comp(atom_env, sig, ls, r_cut)

        # predict for each species
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = spcs_list.index(spc)
            f, vir, v, e = self.predict_component(lengths, xyzs,
                    mappings[map_ind],  mean_only)
            f_spcs += f
            vir_spcs += vir
            v_spcs += v
            e_spcs += e

        return f_spcs, vir_spcs, kern, v_spcs, e_spcs

    def predict_component(self, lengths, xyzs, mapping, mean_only):
        '''
        predict force and variance contribution of one component
        '''
        lengths = np.array(lengths)
        xyzs = np.array(xyzs)

        # predict mean
        e_0, f_0 = mapping.mean(lengths, with_derivatives=True)
        e = np.sum(e_0) # energy

        # predict forces and stress
        vir = np.zeros(6)
        vir_order = ((0,0), (1,1), (2,2), (0,1), (0,2), (1,2))

        # two-body
        if lengths.shape[-1] == 1:
            f_d = np.diag(f_0[:,0,0]) @ xyzs
            f = 2 * np.sum(f_d, axis=0) # force: need to check prefactor 2

            for i in range(6):
                vir_i = f_d[:,vir_order[i][0]]\
                        * xyzs[:,vir_order[i][1]] * lengths[:,0]
                vir[i] = np.sum(vir_i)

        # three-body
        if lengths.shape[-1] == 3:
#            factor1 = 1/lengths[:,1] - 1/lengths[:,0] * lengths[:,2]
#            factor2 = 1/lengths[:,0] - 1/lengths[:,1] * lengths[:,2]
#            f_d1 = np.diag(f_0[:,0,0]+f_0[:,2,0]*factor1) @ xyzs[:,0,:]
#            f_d2 = np.diag(f_0[:,1,0]+f_0[:,2,0]*factor2) @ xyzs[:,1,:]
            f_d1 = np.diag(f_0[:,0,0]) @ xyzs[:,0,:]
            f_d2 = np.diag(f_0[:,1,0]) @ xyzs[:,1,:]
            f_d = f_d1 + f_d2
            f = 3 * np.sum(f_d, axis=0) # force: need to check prefactor 3

            for i in range(6):
                vir_i1 = f_d1[:,vir_order[i][0]]\
                       * xyzs[:,0,vir_order[i][1]] * lengths[:,0]
                vir_i2 = f_d2[:,vir_order[i][0]]\
                       * xyzs[:,1,vir_order[i][1]] * lengths[:,1]
                vir[i] = np.sum(vir_i1 + vir_i2)
            vir *= 1.5


        # predict var
        v = np.zeros(3)
        if not mean_only:
            v_0 = mapping.var(lengths)
            v_d = v_0 @ xyzs
            v = mapping.var.V @ v_d
        return f, vir, v, e

    def write_two_body(self, f):
        a = self.bounds_2[0][0]
        b = self.bounds_2[1][0]
        order = self.grid_num_2

        for ind, spc in enumerate(self.spcs[0]):
            coefs_2 = self.maps_2[ind].mean.__coeffs__

            elem1 = spc[0]
            elem2 = spc[1]
            header_2 = '{elem1} {elem2} {a} {b} {order}\n'\
                .format(elem1=elem1, elem2=elem2, a=a, b=b, order=order)
            f.write(header_2)

            for c, coef in enumerate(coefs_2):
                f.write('{:.10e} '.format(coef))
                if c % 5 == 4 and c != len(coefs_2)-1:
                    f.write('\n')

            f.write('\n')

    def write_three_body(self, f):
        a = self.bounds_3[0]
        b = self.bounds_3[1]
        order = self.grid_num_3

        for ind, spc in enumerate(self.spcs[1]):
            coefs_3 = self.maps_3[ind].mean.__coeffs__

            elem1 = spc[0]
            elem2 = spc[1]
            elem3 = spc[2]

            header_3 = '{elem1} {elem2} {elem3} {a1} {a2} {a3} {b1}'\
                       ' {b2} {b3:.10e} {order1} {order2} {order3}\n'\
                .format(elem1=elem1, elem2=elem2, elem3=elem3,
                        a1=a[0], a2=a[1], a3=a[2],
                        b1=b[0], b2=b[1], b3=b[2],
                        order1=order[0], order2=order[1], order3=order[2])
            f.write(header_3)

            n = 0
            for i in range(coefs_3.shape[0]):
                for j in range(coefs_3.shape[1]):
                    for k in range(coefs_3.shape[2]):
                        coef = coefs_3[i, j, k]
                        f.write('{:.10e} '.format(coef))
                        if n % 5 == 4:
                            f.write('\n')
                        n += 1

            f.write('\n')


    def write_lmp_file(self, lammps_name):
        '''
        write the coefficients to a file that can be used by lammps pair style
        '''

        # write header
        f = open(lammps_name, 'w')

        header_comment = '''# #2bodyarray #3bodyarray\n# elem1 elem2 a b order
        '''
        f.write(header_comment)

        twobodyarray = len(self.spcs[0])
        threebodyarray = len(self.spcs[1])
        header = '\n{} {}\n'.format(twobodyarray, threebodyarray)
        f.write(header)

        # write two body
        if twobodyarray > 0:
            self.write_two_body(f)

        # write three body
        if threebodyarray > 0:
            self.write_three_body(f)

        f.close()


class Map2body:
    def __init__(self, grid_num, bounds, bond_struc,
                 svd_rank=0, mean_only=False, n_cpus=None, nsample=100):
        '''
        Build 2-body MGP
        '''

        arg_dict = inspect.getargvalues(inspect.currentframe())[3]
        del arg_dict['self']
        self.__dict__.update(arg_dict)

        self.build_map_container()


    def GenGrid(self, GP):

        '''
        To use GP to predict value on each grid point, we need to generate the
        kernel vector kv whose length is the same as the training set size.

        1. We divide the training set into several batches, corresponding to
           different segments of kv
        2. Distribute each batch to a processor, i.e. each processor calculate
           the kv segment of one batch for all grids
        3. Collect kv segments and form a complete kv vector for each grid,
           and calculate the grid value by multiplying the complete kv vector
           with GP.alpha
        '''

        kernel_info = utils.get_2bkernel(GP)

        if (self.n_cpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus

        # ------ construct grids ------
        nop = self.grid_num
        bond_lengths = np.linspace(self.bounds[0][0], self.bounds[1][0], nop)
        bond_means = np.zeros([nop])
        if not self.mean_only:
            bond_vars = np.zeros([nop, len(GP.alpha)])
        else:
            bond_vars = None
        env12 = AtomicEnvironment(self.bond_struc, 0, GP.cutoffs)

        with mp.Pool(processes=processes) as pool:
            # A_list = pool.map(self._GenGrid_inner_most, pool_list)
            # break it into pieces
            size = len(GP.training_data)
            nsample = self.nsample
            ns = int(math.ceil(size/nsample/processes))*processes
            nsample = int(math.ceil(size/ns))

            #print("prepare the package for parallelization")
            block_id = []
            nbatch = 0
            for ibatch in range(ns):
                s1 = int(nsample*ibatch)
                e1 = int(np.min([s1 + nsample, size]))
                block_id += [(s1, e1)]
                #print("block", ibatch, s1, e1)
                nbatch += 1

            k12_slice = []
            k12_v_all = np.zeros([len(bond_lengths), size*3])
            count = 0
            base = 0
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_slice.append(pool.apply_async(self._GenGrid_inner,
                                                  args=(GP.training_data[s:e],
                                                        bond_lengths,
                                                        env12, kernel_info)))
                count += 1
                if (count > processes*2):
                    for ibase in range(count):
                        s, e = block_id[ibase+base]
                        k12_v_all[:, s*3:e*3] = k12_slice[ibase].get()
                    del k12_slice
                    k12_slice = []
                    count = 0
                    base = ibatch+1
            if (count > 0):
               for ibase in range(count):
                   s, e = block_id[ibase+base]
                   k12_v_all[:, s*3:e*3] = k12_slice[ibase].get()
               del k12_slice
            pool.close()
            pool.join()

        for b, r in enumerate(bond_lengths):
            k12_v = k12_v_all[b, :]
            bond_means[b] = np.matmul(k12_v, GP.alpha)
            if not self.mean_only:
                bond_vars[b, :] = solve_triangular(GP.l_mat, k12_v, lower=True)

        return bond_means, bond_vars


    def _GenGrid_inner(self, training_data, bond_lengths,
            env12, kernel_info):

        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, en_force_kernel, cutoffs, hyps, hyps_mask = kernel_info
        size = len(training_data)
        k12_v = np.zeros([len(bond_lengths), size*3])
        for b, r in enumerate(bond_lengths):
            env12.bond_array_2 = np.array([[r, 1, 0, 0]])
            k12_v[b, :] = utils.en_kern_vec(training_data,
                                            env12, en_force_kernel,
                                            hyps, cutoffs)
        return k12_v


    def build_map_container(self):

        '''
        build 1-d spline function for mean, 2-d for var
        '''
        self.mean = CubicSpline(self.bounds[0], self.bounds[1],
                                orders=[self.grid_num])

        if not self.mean_only:
            self.var = PCASplines(self.bounds[0], self.bounds[1],
                                  orders=[self.grid_num],
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):
        y_mean, y_var = self.GenGrid(GP)
        self.mean.set_values(y_mean)
        if not self.mean_only:
            self.var.set_values(y_var)



class Map3body:

    def __init__(self, grid_num, bounds, bond_struc,
            svd_rank=0, mean_only=False, load_grid=None, update=True,
            n_cpus=None, nsample=100):
        '''
        Build 3-body MGP
        '''

        # get all arguments as attributes
        arg_dict = inspect.getargvalues(inspect.currentframe())[3]
        del arg_dict['self']
        self.__dict__.update(arg_dict)

        spc = bond_struc.coded_species
        self.species_code = str(spc[0]) + str(spc[1]) + str(spc[2])

        self.build_map_container()


    def GenGrid(self, GP, processes=mp.cpu_count()):

        '''
        To use GP to predict value on each grid point, we need to generate the
        kernel vector kv whose length is the same as the training set size.

        1. We divide the training set into several batches, corresponding to
           different segments of kv
        2. Distribute each batch to a processor, i.e. each processor calculate
           the kv segment of one batch for all grids
        3. Collect kv segments and form a complete kv vector for each grid,
           and calculate the grid value by multiplying the complete kv vector
           with GP.alpha
        '''

        if (self.n_cpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus
        if processes == 1:
            self.GenGrid_serial(GP)

        # ------ get 3body kernel info ------
        kernel_info = utils.get_3bkernel(GP)

        # ------ construct grids ------
        n1, n2, n12 = self.grid_num
        bonds1  = np.linspace(self.bounds[0][0], self.bounds[1][0], n1)
        bonds2  = np.linspace(self.bounds[0][0], self.bounds[1][0], n2)
        bonds12 = np.linspace(self.bounds[0][2], self.bounds[1][2], n12)
        grid_means = np.zeros([n1, n2, n12])

        if not self.mean_only:
            grid_vars = np.zeros([n1, n2, n12, len(GP.alpha)])
        else:
            grid_vars = None

        env12 = AtomicEnvironment(self.bond_struc, 0, GP.cutoffs)

        with mp.Pool(processes=processes) as pool:
            if self.update:
                if self.kv3name in os.listdir():
                    subprocess.run(['rm', '-rf', self.kv3name])
                subprocess.run(['mkdir', self.kv3name])

            #print("prepare the package for parallelization")
            size = len(GP.training_data)
            nsample = self.nsample
            ns = int(math.ceil(size/nsample/processes))*processes
            nsample = int(math.ceil(size/ns))

            block_id = []
            nbatch = 0
            for ibatch in range(ns):
                s1 = int(nsample*ibatch)
                e1 = int(np.min([s1 + nsample, size]))
                block_id += [(s1, e1)]
                #print("block", ibatch, s1, e1)
                nbatch += 1

            k12_slice = []
            #print('before for', ns, nsample, time.time())
            count = 0
            base = 0
            k12_v_all = np.zeros([len(bonds1), len(bonds2), len(bonds12),
                                  size * 3])
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_slice.append(pool.apply_async(self._GenGrid_inner,
                                                  args=(GP.training_data[s:e],
                                                        bonds1, bonds2, bonds12,
                                                        env12, kernel_info)))
                #print('send', ibatch, ns, s, e, time.time())
                count += 1
                if (count > processes*2):
                    for ibase in range(count):
                        s, e = block_id[ibase+base]
                        k12_v_all[:, :, :, s*3:e*3] = k12_slice[ibase].get()
                    del k12_slice
                    k12_slice = []
                    count = 0
                    base = ibatch+1
            if (count > 0):
               for ibase in range(count):
                   s, e = block_id[ibase+base]
                   k12_v_all[:, :, :, s*3:e*3] = k12_slice[ibase].get()
               del k12_slice

            pool.close()
            pool.join()

        for b12 in range(len(bonds12)):
            for b1 in range(len(bonds1)):
                for b2 in range(len(bonds2)):
                    k12_v = k12_v_all[b1, b2, b12, :]
                    grid_means[b1, b2, b12] = np.matmul(k12_v, GP.alpha)
                    if not self.mean_only:
                        grid_vars[b1, b2, b12, :] = solve_triangular(GP.l_mat,
                            k12_v, lower=True)

        # ------ save mean and var to file -------
        np.save('grid3_mean_'+self.species_code, grid_means)
        np.save('grid3_var_'+self.species_code, grid_vars)

        return grid_means, grid_vars

    def _GenGrid_inner(self, training_data, bonds1, bonds2, bonds12, env12, kernel_info):

        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, en_force_kernel, cutoffs, hyps, hyps_mask = kernel_info
        # open saved k vector file, and write to new file
        size = len(training_data)*3
        k12_v = np.zeros([len(bonds1), len(bonds2), len(bonds12), size])
        for b12, r12 in enumerate(bonds12):
            for b1, r1 in enumerate(bonds1):
                for b2, r2 in enumerate(bonds2):

                    #r12 = np.sqrt(r1**2 + r2**2 - 2*r1*r2*c12)
                    #if (r1>r12+r2) or (r2>r12+r1) or (r12>r1+r2): # not a triangle
                    #    k12_v[b1, b2, b12, :] = np.zeros(size)
                    #    continue

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])
                    k12_v[b1, b2, b12, :] = utils.en_kern_vec(training_data,
                                                              env12, en_force_kernel,
                                                              hyps, cutoffs)

        return k12_v


    def build_map_container(self):

        '''
        build 3-d spline function for mean,
        3-d for the low rank approximation of L^{-1}k*
        '''

       # create spline interpolation class object
        self.mean = CubicSpline(self.bounds[0], self.bounds[1],
                                orders=self.grid_num)

        if not self.mean_only:
            self.var = PCASplines(self.bounds[0], self.bounds[1],
                                  orders=self.grid_num,
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):
        # Load grid or generate grid values
        if not self.load_grid:
            y_mean, y_var = self.GenGrid(GP)
        else:
            y_mean = np.load(self.load_grid+'grid3_mean_'+\
                    self.species_code+'.npy')
            y_var = np.load(self.load_grid+'grid3_var_'+\
                    self.species_code+'.npy')

        self.mean.set_values(y_mean)
        if not self.mean_only:
            self.var.set_values(y_var)

