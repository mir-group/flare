import numpy as np
import pymatgen as pmg
import flare as flare

from flare.predict import predict_on_atom
from flare.gp import GaussianProcess

from typing import List, Tuple, Union, Callable
from flare.struc import Structure

from pymatgen.analysis.defects.utils import StructureMotifInterstitial

# TODO: ALLOW FOR A FEW RELAXATION FRAMES TO PROBE ANY INTERESTING ADJACENT
# CONFIGURATIONS


def check_local_threshold(structure: pmg.core.Structure, site,
                          cutoff:float,
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

    to_check_sites = structure.get_neighbors(site, cutoff)
    if verbose:
        print(f'Now checking error on {len(to_check_sites)} sites:')
    to_check_idxs = [site.index for site in to_check_sites]

    stds = [predict_on_atom((Structure.from_pmg_structure(structure), i,
                             gp))[1] for i in
            to_check_idxs]

    if np.max(stds) > threshold:
        return False
    else:
        return True


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
        'str',depth = 1, halt_number: int = np.inf,
                                 veto_function: Callable = None,
                                 verbose:int = 0)->List[
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

        for new_site in site_positions:
            new_structure = self.base_structure.to_pmg_structure()

            new_structure.append(species=new_atom,
                                 coords=new_site)

            target_site = new_structure.sites[-1]

            # Check if too-high uncertainty
            if not check_local_threshold(new_structure, target_site,
                                         self.cutoff, self.threshold,self.gp):
                high_uncertainty_configs.append(target_site)

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
                                         self.cutoff, self.threshold,self.gp):
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
