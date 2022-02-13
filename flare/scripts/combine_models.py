import logging, sys, time
import numpy as np
from flare.scripts.otf_train import get_gp_calc, get_sgp_calc

def append_gp(gp1, gp2, update_matrices=True, logger=None):
    pass

def append_sgp(gp1, gp2, update_matrices=True, logger=None):
    """
    Add gp2 to gp1
    """

    for s in range(len(gp2)):
        tic = time.time()
        logger.info(f"- Frame: {s}")

        custom_range = gp2.sparse_gp.sparse_indices[0][s]
        struc_cpp = gp2.training_data[s]
        logger.info(f"Number of atoms: {struc_cpp.nat}")
        logger.info(f"Adding atom {custom_range}")

        if len(struc_cpp.energy) > 0:
            energy = struc_cpp.energy[0]
        else:
            energy = None

        gp1.update_db(
            struc_cpp,
            struc_cpp.forces,
            custom_range=custom_range,
            energy=energy,
            stress=struc_cpp.stresses,
            mode="specific",
            sgp=None,
            update_qr=False,
        )

        toc = time.time()
        logger.info(f"Time of adding frame: {toc - tic}")

    if update_matrices:
        tic = time.time()
        gp1.update_matrices_QR()
        toc = time.time()
        logger.info(f"Time of updating matrices: {toc - tic}")

    logger.info("\n")


def main():
    t0 = time.time()

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    # initialize GP
    gp_name = flare_config.get("gp")
    if gp_name == "GaussianProcess":
        from flare.bffs.gp.calculator import FLARE_Calculator
        flare_calc, kernels = get_gp_calc(config["flare_calc"])
        gp = flare_calc.gp_model
        append_model = append_gp

    elif gp_name == "SGP_Wrapper":
        from flare_pp.sparse_gp_calculator import SGP_Calculator
        flare_calc, kernels = get_sgp_calc(config["flare_calc"])
        gp = flare_calc.gp_model
        append_model = append_sgp

    else:
        raise NotImplementedError(f"{gp_name} is not implemented")

    # append GP
    output_name = config["combine"].get("output_name", "combine")
    output = Output(output_name, always_flush=True, print_as_xyz=True)
    output.write_header(
        str(gp),
        dt = None,
        Nsteps = 0,
        structure = None,
        std_tolerance = None,
    )
    logger = logging.getLogger(output.basename + "log")

    json_files = config["combine"]["models"]
    for f in json_files:
        curr_model = flare_calc.__class__.from_file(f)
        if len(curr_calc) == 2: 
            curr_calc, curr_kern = curr_model
        else:
            curr_calc = curr_model

        if json_files.index(f) == len(json_files) - 1:
            update_matrices = True
        else:
            update_matrices = False

        logger.info("--------------------------------------------------------")
        logger.info(f"GP model: {f}".center(56))
        logger.info("--------------------------------------------------------")

        append_model(gp, curr_calc.gp_model, update_matrices, logger)

        logger.info(f"Number of training structures: {len(gp)}")
        output.write_wall_time(t0)
        logger.info(f"\n")
    
    optimize = config["combine"].get("optimize", False)
    if optimize:
        logger.info("Hyperparameter optimization")
        gp.train(logger_name=output.basename + "hyps")
        output.write_hyps(
            gp.hyp_labels, 
            gp.hyps, 
            t0, 
            gp.likelihood, 
            gp.likelihood_gradient, 
            hyps_mask=gp.hyps_mask,
        )

    output.conclude_run()
