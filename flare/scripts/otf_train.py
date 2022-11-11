import time, os, shutil, glob, subprocess, sys, json
from copy import deepcopy
import pytest
import pkgutil, pyclbr
import importlib
import inspect
import numpy as np

from flare.learners.otf import OTF
from flare.md.fake import FakeDFT

from ase import units
import ase.calculators as ase_calculators
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase import io
from ase.symbols import symbols2numbers

import yaml


def get_super_cell(atoms_config):
    """
    Set up Supercell from ASE Atoms
    """
    # parse parameters
    atoms_file = atoms_config.get("file")
    atoms_format = atoms_config.get("format", None)
    atoms_index = atoms_config.get("index", -1)
    replicate = atoms_config.get("replicate", [1, 1, 1])
    jitter = atoms_config.get("jitter", 0)

    if atoms_format is not None:
        super_cell = io.read(atoms_file, format=atoms_format, index=atoms_index)
    else:
        super_cell = io.read(atoms_file)
    super_cell *= replicate

    # jitter positions to give nonzero force on first frame
    super_cell.positions += (2 * np.random.rand(len(super_cell), 3) - 1) * jitter
    return super_cell


def get_dft_calc(dft_config):
    """
    Set up ASE DFT calculator
    """
    dft_calc_name = dft_config.get("name", "LennardJones")
    dft_calc_kwargs = dft_config.get("kwargs", {})
    dft_calc_params = dft_config.get("params", {})

    if dft_calc_name == "FakeDFT":
        dft_calc = FakeDFT(**dft_calc_kwargs)
        dft_calc.set(**dft_calc_params)
        return dft_calc

    # find the module including the ASE DFT calculator class by name
    dft_module_name = ""
    for importer, modname, ispkg in pkgutil.iter_modules(ase_calculators.__path__):
        module_info = pyclbr.readmodule("ase.calculators." + modname)
        if dft_calc_name in module_info:
            dft_module_name = modname
            break

    # if the module is not found in the current folder, then search the sub-directory
    if not dft_module_name:
        for calc_dir in os.listdir(ase_calculators.__path__[0]):
            dir_path = ase_calculators.__path__[0] + "/" + calc_dir
            if os.path.isdir(dir_path):
                for importer, modname, ispkg in pkgutil.iter_modules([dir_path]):
                    module_info = pyclbr.readmodule(
                        "ase.calculators." + calc_dir + "." + modname
                    )
                    if dft_calc_name in module_info:
                        dft_module_name = calc_dir + "." + modname
                        break

    # import ASE DFT calculator module, and build a DFT calculator class object
    dft_calc = None
    dft_module = importlib.import_module("ase.calculators." + dft_module_name)
    for name, obj in inspect.getmembers(dft_module, inspect.isclass):
        if name == dft_calc_name:
            dft_calc = obj(**dft_calc_kwargs)
    dft_calc.set(**dft_calc_params)
    return dft_calc


def get_flare_calc(flare_config):
    """
    Set up ASE flare calculator
    """
    gp_name = flare_config.get("gp")
    if gp_name == "GaussianProcess":
        return get_gp_calc(flare_config)
    elif gp_name == "SGP_Wrapper":
        return get_sgp_calc(flare_config)
    else:
        raise NotImplementedError(f"{gp_name} is not implemented")


def get_gp_calc(flare_config):
    """
    Return a FLARE_Calculator with gp from GaussianProcess
    """
    from flare.bffs.gp import GaussianProcess
    from flare.bffs.mgp import MappedGaussianProcess
    from flare.bffs.gp.calculator import FLARE_Calculator
    from flare.utils.parameter_helper import ParameterHelper

    gp_file = flare_config.get("file", None)

    # Load GP from file
    if gp_file is not None:
        with open(gp_file, "r") as f:
            gp_dct = json.loads(f.readline())
            if gp_dct.get("class", None) == "FLARE_Calculator":
                flare_calc = FLARE_Calculator.from_file(gp_file)
            else:
                gp, _ = GaussianProcess.from_file(gp_file)
                flare_calc = FLARE_Calculator(gp)
        return flare_calc

    # Create gaussian process model
    kernels = flare_config.get("kernels")
    hyps = flare_config.get("hyps", "random")
    opt_algorithm = flare_config.get("opt_algorithm", "BFGS")
    max_iterations = flare_config.get("max_iterations", 20)
    bounds = flare_config.get("bounds", None)

    gp_parameters = flare_config.get("gp_parameters")
    n_cpus = flare_config.get("n_cpus", 1)
    use_mapping = flare_config.get("use_mapping", False)

    # set up GP hyperparameters
    pm = ParameterHelper(
        kernels=kernels,
        random=True,
        parameters=gp_parameters,
    )
    hm = pm.as_dict()
    if hyps == "random":
        hyps = hm["hyps"]

    gp_model = GaussianProcess(
        kernels=kernels,
        component="mc",
        hyps=hyps,
        cutoffs=hm["cutoffs"],
        hyps_mask=None,
        hyp_labels=hm["hyp_labels"],
        opt_algorithm=opt_algorithm,
        maxiter=max_iterations,
        parallel=n_cpus > 1,
        per_atom_par=flare_config.get("per_atom_par", True),
        n_cpus=n_cpus,
        n_sample=flare_config.get("n_sample", 100),
        output=None,
        name=flare_config.get("name", "default_gp"),
        energy_noise=flare_config.get("energy_noise", 0.01),
    )

    # create mapped gaussian process
    if use_mapping:
        grid_params = flare_config.get("grid_params")
        var_map = flare_config.get("var_map", "pca")
        unique_species = flare_config.get("unique_species")
        coded_unique_species = symbols2numbers(unique_species)
        mgp_model = MappedGaussianProcess(
            grid_params=grid_params,
            unique_species=coded_unique_species,
            n_cpus=n_cpus,
            var_map=var_map,
        )
    else:
        mgp_model = None

    flare_calc = FLARE_Calculator(
        gp_model=gp_model,
        mgp_model=mgp_model,
        par=n_cpus > 1,
        use_mapping=use_mapping,
    )
    return flare_calc, kernels


def get_sgp_calc(flare_config):
    """
    Return a SGP_Calculator with sgp from SparseGP
    """
    from flare.bffs.sgp._C_flare import NormalizedDotProduct, SquaredExponential
    from flare.bffs.sgp._C_flare import B2, B3, TwoBody, ThreeBody, FourBody
    from flare.bffs.sgp import SGP_Wrapper
    from flare.bffs.sgp.calculator import SGP_Calculator

    sgp_file = flare_config.get("file", None)

    # Load sparse GP from file
    if sgp_file is not None:
        with open(sgp_file, "r") as f:
            gp_dct = json.loads(f.readline())
            if gp_dct.get("class", None) == "SGP_Calculator":
                flare_calc, kernels = SGP_Calculator.from_file(sgp_file)
            else:
                sgp, kernels = SGP_Wrapper.from_file(sgp_file)
                flare_calc = SGP_Calculator(sgp)
        return flare_calc, kernels

    kernels = flare_config.get("kernels")
    opt_algorithm = flare_config.get("opt_algorithm", "BFGS")
    max_iterations = flare_config.get("max_iterations", 20)
    bounds = flare_config.get("bounds", None)
    use_mapping = flare_config.get("use_mapping", False)

    # Define kernels.
    kernels = []
    for k in flare_config["kernels"]:
        if k["name"] == "NormalizedDotProduct":
            kernels.append(NormalizedDotProduct(k["sigma"], k["power"]))
        elif k["name"] == "SquaredExponential":
            kernels.append(SquaredExponential(k["sigma"], k["ls"]))
        else:
            raise NotImplementedError(f"{k['name']} kernel is not implemented")

    # Define descriptor calculators.
    n_species = len(flare_config["species"])
    cutoff = flare_config["cutoff"]
    descriptors = []
    for d in flare_config["descriptors"]:
        if "cutoff_matrix" in d:  # multiple cutoffs
            assert np.allclose(np.array(d["cutoff_matrix"]).shape, (n_species, n_species)),\
                "cutoff_matrix needs to be of shape (n_species, n_species)"

        if d["name"] == "B2":
            radial_hyps = [0.0, cutoff]
            cutoff_hyps = []
            descriptor_settings = [n_species, d["nmax"], d["lmax"]]
            if "cutoff_matrix" in d:  # multiple cutoffs
                desc_calc = B2(
                    d["radial_basis"],
                    d["cutoff_function"],
                    radial_hyps,
                    cutoff_hyps,
                    descriptor_settings,
                    d["cutoff_matrix"],
                )
            else:
                desc_calc = B2(
                    d["radial_basis"],
                    d["cutoff_function"],
                    radial_hyps,
                    cutoff_hyps,
                    descriptor_settings,
                )

        elif d["name"] == "B3":
            radial_hyps = [0.0, cutoff]
            cutoff_hyps = []
            descriptor_settings = [n_species, d["nmax"], d["lmax"]]
            desc_calc = B3(
                d["radial_basis"],
                d["cutoff_function"],
                radial_hyps,
                cutoff_hyps,
                descriptor_settings,
            )

        elif d["name"] == "TwoBody":
            desc_calc = TwoBody(cutoff, n_species, d["cutoff_function"], cutoff_hyps)

        elif d["name"] == "ThreeBody":
            desc_calc = ThreeBody(cutoff, n_species, d["cutoff_function"], cutoff_hyps)

        elif d["name"] == "FourBody":
            desc_calc = FourBody(cutoff, n_species, d["cutoff_function"], cutoff_hyps)

        else:
            raise NotImplementedError(f"{d['name']} descriptor is not supported")

        descriptors.append(desc_calc)

    # Define remaining parameters for the SGP wrapper.
    species_map = {flare_config.get("species")[i]: i for i in range(n_species)}
    sae_dct = flare_config.get("single_atom_energies", None)
    if sae_dct is not None:
        assert n_species == len(
            sae_dct
        ), "'single_atom_energies' should be the same length as 'species'"
        single_atom_energies = {i: sae_dct[i] for i in range(n_species)}
    else:
        single_atom_energies = {i: 0 for i in range(n_species)}

    sgp = SGP_Wrapper(
        kernels=kernels,
        descriptor_calculators=descriptors,
        cutoff=cutoff,
        sigma_e=flare_config.get("energy_noise"),
        sigma_f=flare_config.get("forces_noise"),
        sigma_s=flare_config.get("stress_noise"),
        species_map=species_map,
        variance_type=flare_config.get("variance_type", "local"),
        single_atom_energies=single_atom_energies,
        energy_training=flare_config.get("energy_training", True),
        force_training=flare_config.get("force_training", True),
        stress_training=flare_config.get("stress_training", True),
        max_iterations=max_iterations,
        opt_method=opt_algorithm,
        bounds=bounds,
    )

    flare_calc = SGP_Calculator(sgp, use_mapping)
    return flare_calc, kernels


def fresh_start_otf(config):
    """
    Set up MD and OTF training engine
    """

    super_cell = get_super_cell(config["supercell"])
    dft_calc = get_dft_calc(config["dft_calc"])
    flare_calc, kernels = get_flare_calc(config["flare_calc"])
    otf_config = config.get("otf")

    # intialize velocity
    # The "file" option uses the velocities read from the supercell file.
    initial_velocity = otf_config.get("initial_velocity", "file")
    if initial_velocity != "file":
        # Otherwise, the initial_velocity is a number specifying the temperature
        # to initialize the velocity with Boltzmann distribution
        init_temp = float(initial_velocity)
        MaxwellBoltzmannDistribution(super_cell, init_temp * units.kB)
        Stationary(super_cell)
        ZeroRotation(super_cell)

    otf = OTF(
        super_cell,
        flare_calc=flare_calc,
        dft_calc=dft_calc,
        **otf_config,
    )

    otf.run()


def restart_otf(config):
    """
    Set up MD and OTF training engine. Restart with checkpoint files
    """

    otf_config = config.get("otf")
    checkpoint = otf_config.get("checkpoint")
    otf = OTF.from_checkpoint(checkpoint)

    # allow modification of some parameters
    for attr in [
        "number_of_steps",
        "rescale_steps",
        "rescale_temps",
        "write_model",
        "freeze_hyps",
        "store_dft_output",
    ]:
        if attr in otf_config:
            setattr(otf, attr, otf_config.get(attr))

    otf.run()


def main():
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    mode = config.get("otf").get("mode", "fresh")
    if mode == "fresh":
        fresh_start_otf(config)
    elif mode == "restart":
        restart_otf(config)
