import warnings
import numpy as np

from numpy import array
from numba import njit
from math import exp, floor
from typing import Callable

from flare.env import AtomicEnvironment
from flare.kernels.cutoffs import quadratic_cutoff
from flare.kernels.utils import str_to_kernel_set
from flare.parameters import Parameters


def str_to_mapped_kernel(name: str, component: str = "mc", hyps_mask: dict = None):
    """
    Return kernels and kernel gradient function based on a string.
    If it contains 'sc', it will use the kernel in sc module;
    otherwise, it uses the kernel in mc_simple;
    if sc is not included and multihyps is True,
    it will use the kernel in mc_sephyps module.
    Otherwise, it will use the kernel in the sc module.

    Args:

    name (str): name for kernels. example: "2+3mc"
    multihyps (bool, optional): True for using multiple hyperparameter groups

    :return: mapped kernel function, kernel gradient, energy kernel,
             energy_and_force kernel

    """

    multihyps = True
    if hyps_mask is None:
        multihyps = False
    elif hyps_mask["nspecie"] == 1:
        multihyps = False

    # b2 = Two body in use, b3 = Three body in use
    b2 = False
    many = False
    b3 = False
    for s in ["2", "two"]:
        if s in name.lower() or s == name.lower():
            b2 = True

    for s in ["3", "three"]:
        if s in name.lower() or s == name.lower():
            b3 = True

    if b2:
        if multihyps:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
    elif b3:
        if multihyps:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
    else:
        warnings.warn("The mapped kernel for many-body is not implemented.")
        return None


def get_kernel_term(kernel_name, component, hyps_mask, hyps, grid_kernel=False):
    """
    Args
        term (str): 'twobody' or 'threebody'
    """
    if grid_kernel:
        stks = str_to_mapped_kernel
        kernel_name_list = kernel_name
    else:
        stks = str_to_kernel_set
        kernel_name_list = [kernel_name]

    kernel, _, ek, efk, _, _, _ = stks(kernel_name_list, component, hyps_mask)

    # hyps_mask is modified here
    hyps, cutoffs, hyps_mask = Parameters.get_component_mask(
        hyps_mask, kernel_name, hyps=hyps
    )

    return (kernel, ek, efk, cutoffs, hyps, hyps_mask)
