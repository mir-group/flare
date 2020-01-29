import numpy as np
import pymatgen as pmg
import flare as flare

from flare.predict import predict_on_atom
from flare.gp import GaussianProcess

from typing import List, Tuple, Union, Callable
from flare.struc import Structure

from pymatgen import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

import time

from pymatgen.analysis.defects.utils import StructureMotifInterstitial

# TODO: ALLOW FOR A FEW RELAXATION FRAMES TO PROBE ANY INTERESTING ADJACENT
# CONFIGURATIONS


def get_rand_vec(distance,min_distance=0):
    # Implementation borrowed from Pymatgen structure object's
    # perturb method.
    vector = np.random.randn(3)
    vnorm = np.linalg.norm(vector)
    dist = distance
    if isinstance(min_distance, (float, int)):
        dist = np.random.uniform(min_distance, dist)
    return vector / vnorm * dist if vnorm != 0 else get_rand_vec(distance,
                                                                 min_distance)



def check_local_threshold(structure: pmg.core.Structure,
                          site,
                          threshold:float,
                          gp: GaussianProcess,
                          verbose: int = 0)->bool:
    """

    :param structure:
    :param i:
    :param cutoff:
    :param threshold:
    :return:
    """
    
    if verbose:
        print("Now checking around neighbors of "
              "site {} in a ball of radius {}".format(site, np.max(gp.cutoffs)))
        now = time.time()


    to_check_sites = structure.get_sites_in_sphere(site.coords,
                                r=np.max(gp.cutoffs),include_index=True,
                                                   include_image=True)
    if verbose:
        print(f'Now checking error on {len(to_check_sites)} sites:')

    cutoff = np.max(gp.cutoffs)

    to_check_idxs = [site[2] for site in to_check_sites]
    flare_struc = Structure.from_pmg_structure(structure)
    
    # Loop over atoms and break if one is high uncertainty
    for i in to_check_idxs:
        if np.max(predict_on_atom((flare_struc,i,gp))[1])>threshold:
            if verbose:
                print("Condition triggered: Took {} seconds".format(time.time()-now))
            return False
    if verbose:
        print("No high uncertainty configurations found: Took {} seconds".format(time.time()-now))
    return True


def check_new_site_threshold(base_structure,
                         position: 'np.ndarray',
                         new_atom_type: Union[str, Molecule],
                         gp: GaussianProcess,
                         threshold: float,
                         verbose: int = 0 ):
    """
    Given a base structure, a new position, and an atom type,
    add the atom and test if it triggers the uncertainty

    :param base_structure:
    :param site:
    :param new_atom:
    :param gp:
    :param threshold:
    :param verbose:
    :return:
    """


    # Convert to pymatgen structure to inspect atoms around radius
    new_structure = base_structure.to_pmg_structure()

    if isinstance(new_atom_type, str):
        new_structure.append(species=new_atom_type,
                             coords=position, coords_are_cartesian=True)
    else:
        raise NotImplementedError
    target_site = new_structure.sites[-1]

    # Check if too-high uncertainty
    pre = time.time()
    triggers_threshold = check_local_threshold(structure = new_structure,
                                             site=target_site,
                                             threshold=threshold, gp=gp,
                                             verbose=verbose)
    post = time.time()
    if verbose > 2:
        print("Time to check configuration: {}".format(post - pre))

    return triggers_threshold, new_structure



class Sampler(object):

    def __init__(self, base_structure: flare.struc.Structure,
                 gp:GaussianProcess,
                 dft_module: str,
                 sampling_protocol: str = None,
                 rel_threshold: float = 1,
                 abs_threshold: float = .5):
        """

        :param base_structure: Base structure to permute
        :param gp:  Gaussian Process to develop
        :param dft_module: String
        :param sampling_protocol: TBD but will determine which way to go
        :param rel_threshold: Relative uncertainty threshold
        :param abs_threshold: Absolute uncertainty thre
        """
        self.base_structure = base_structure
        self.gp = gp
        self.dft_module = dft_module

        self.sampling_protocol = None

        self.cutoff = np.max(gp.cutoffs)

        if rel_threshold == 0:
            self.threshold = abs_threshold
        elif abs_threshold == 0:
            self.threshold = rel_threshold * gp.hyps[-1]
        else:
            self.threshold = min(rel_threshold * gp.hyps[-1], abs_threshold)

    def permutation_site_sampler(self, site_positions: np.array, new_atom:
        'str',depth: int = 1,cur_depth = 0, halt_number: int = np.inf,
                                 veto_function: Callable = None,
                                 verbose:int = 0,
                                 relax = False,
                                 kick: float =.05)->List[
        pmg.core.Structure]:
        """
        :param site_positions: List of positions within the unit cell with 
        which to gauge new sites. 
        :param new_atom: Element of new atom to include. Will later be 
        extended to molecules.
        :param depth: Number of simultaneous sites to consider (depth) Not yet 
        implemented.
        :param halt_number: When this many structures have been sampled, 
        stop (can help to produce a managable set of configurations to run)
        :param verbose: Will increase output based on setting
        :return: 
        """

        high_uncertainty_configs = []

        for new_atom_coord in site_positions:

            within_thresh, new_struc = check_new_site_threshold(
                self.base_structure, new_atom_coord,
                new_atom_type=new_atom, gp = self.gp,
                threshold=self.threshold)

            if not within_thresh:
                high_uncertainty_configs.append(new_struc)

            if len(high_uncertainty_configs) >= halt_number:
                break

        return high_uncertainty_configs


    def neb_idpp_site_sampler(self,point_1: np.array, point_2:
    np.array,
                              new_atom:'str',depth: int = 1,
                              cur_depth = 0, halt_number: int = np.inf,
                                 veto_function: Callable = None,
                                 verbose:int = 0,
                                 relax = False,
                                 kick: float =.05)->List[
        pmg.core.Structure]:
        """
        :param site_positions: List of positions within the unit cell with 
        which to gauge new sites. 
        :param new_atom: Element of new atom to include. Will later be 
        extended to molecules.
        :param depth: Number of simultaneous sites to consider (depth) Not yet 
        implemented.
        :param halt_number: When this many structures have been sampled, 
        stop (can help to produce a managable set of configurations to run)
        :param verbose: Will increase output based on setting
        :return: 
        """


        high_uncertainty_configs = []

        aaa = AseAtomsAdaptor()

        new_struc_1 = self.base_structure.to_pmg_structure()
        new_struc_1.append(new_atom,point_1, coords_are_cartesian=True)
        new_struc_2 = self.base_structure.to_pmg_structure()
        new_struc_2.append(new_atom,point_2, coords_are_cartesian=True)

        ase_struc_1 = aaa.get_atoms(new_struc_1)
        ase_struc_2 = aaa.get_atoms(new_struc_2)

        return high_uncertainty_configs

    def cheap_interp_site_sampler(self, new_points: np.ndarray,
                               new_atom: 'str',
                                  halt_number: int = np.inf,
                                  search_hump = False)->List[
        pmg.core.Structure]:
        """
        :param site_positions: List of positions within the unit cell with
        which to gauge new sites.
        :param new_atom: Element of new atom to include. Will later be
        extended to molecules.
        :param depth: Number of simultaneous sites to consider (depth) Not yet
        implemented.
        :param halt_number: When this many structures have been sampled,
        stop (can help to produce a managable set of configurations to run)
        :param verbose: Will increase output based on setting
        :param search_hump: Will try above and below the midpoint
        :return:
        """

        high_uncertainty_configs = []


        new_points = np.array(new_points)

        for pt1 in new_points[:]:
            for pt2 in new_points[:]:

                if np.equal(pt1, pt2):
                    continue
                joint_struc = self.base_structure.to_pmg_structure()

                joint_struc.append(new_atom, pt1, coords_are_cartesian=True)
                joint_struc.append(new_atom, pt2, coords_are_cartesian=True)

                if joint_struc.get_distance(-2, -1) > 3:
                    continue

                new_struc_1 = self.base_structure.to_pmg_structure()
                new_struc_2 = self.base_structure.to_pmg_structure()

                new_struc_1.append(new_atom, pt1, coords_are_cartesian=True)
                new_struc_2.append(new_atom, pt2, coords_are_cartesian=True)

                interp_strucs =  new_struc_1.interpolate(new_struc_2,
                                                         nimages=3)[1]
                assert len(interp_strucs) == 3
                mid_struc = interp_strucs[1]

                within_thresh = check_local_threshold(
                    mid_struc, mid_struc.sites[-1], gp=self.gp,
                    threshold=self.threshold)

                if not within_thresh and mid_struc.is_valid(.5):
                    high_uncertainty_configs.append(mid_struc)

                if search_hump:
                    mid_struc.translate_sites([-1],np.array([0,0,-.3]))
                    if mid_struc.is_valid(.5) and not check_local_threshold(
                        mid_struc, mid_struc.sites[-1], gp=self.gp,
                        threshold=self.threshold):
                        high_uncertainty_configs.append(mid_struc)

                    mid_struc.translate_sites([-1], np.array([0, 0, .45]))
                    within_thresh = check_local_threshold(
                        mid_struc, mid_struc.sites[-1], gp=self.gp,
                        threshold=self.threshold)
                    if not within_thresh:
                        high_uncertainty_configs.append(mid_struc)

                if len(high_uncertainty_configs) >= halt_number:
                    break
            if len(high_uncertainty_configs) >= halt_number:
                break

        return high_uncertainty_configs

    def substitution_site_sampler(self, target_indexes: List[int],
                                  new_species: str,
                                  target_species: Union[str,List[str]] =[],
                                  halt: int = np.inf,
                                  depth: int = 1):
        """
        Samples substitution

        :param target_indexes:
        :param target_species: Not supported yet
        :param new_species:
        :param depth:
        :return:
        """

        if len(target_indexes) == 0 and len(target_species) == 0:
            raise ValueError("Need to have target indices or target species!")

        if len(target_species)>0:
            raise NotImplementedError

        # Loop through indexes and swap out, test the uncertainty

        high_uncertainty_configs = []
        for site in target_indexes:

            new_structure = self.base_structure.to_pmg_structure()


            new_structure.sites[site].species = new_species

            if not check_local_threshold(new_structure, site,
                                         self.cutoff, self.threshold, self.gp):
                high_uncertainty_configs.append(new_structure)

            if len(high_uncertainty_configs) >= halt:
                break

        return high_uncertainty_configs



    def random_substitution_site_sampler(self, transformation_dict,
                                  target_species: Union[str,List[str]],
                                  new_species: str, depth: int = 1):
        """
        Generates several `alloy' versions of the structure passed in
        and tests the uncertainty associated with the configurations.

        :param transformation_dict:
        :param target_species:
        :param new_species:
        :param depth:
        :return:
        """

        raise NotImplementedError()
