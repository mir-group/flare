import os
import numpy as np
from . import config


def preset_atoms(newton="off", read_restart=None):
    """
    Basic settings for LAMMPS
    """
    if read_restart is not None:  # start from dumped trajectory
        read_data_command = f"\nread_restart        {read_restart}\n"
    else:
        read_data_command = f"\nread_data           {config.LMP_DAT}\n"

    preset_command = f"""
atom_style          atomic
units               metal
boundary            p p p
atom_modify         sort 0 0.0
newton              {newton}
box                 tilt large
{read_data_command}
"""
    return preset_command


def preset_mgp(coeff_dir, species, read_restart=None, compute_unc=True):
    """
    Default setting for using MGP pair_style, which requires `newton off`. The
    `pair_style`, `pair_coeff` and the `compute` command for uncertainty are presented
    """
    preset_command = preset_atoms(newton="off", read_restart=read_restart)

    # TODO: for passive learning, the uncertainty is not needed
    mgp_file = os.path.join(coeff_dir, config.COEFF + ".mgp")
    std_file = os.path.join(coeff_dir, config.COEFF + ".std")
    if compute_unc:
        compute_unc_command = f"""
compute             unc all uncertainty/atom {std_file} {species} yes yes
"""
    else:
        compute_unc_command = ""

    ff_command = f""" 
pair_style          mgp
pair_coeff          * * {mgp_file} {species} yes yes
{compute_unc_command}
"""

    return preset_command + ff_command


def preset_flare_pp(coeff_dir, species, read_restart=None, compute_unc=True, uncertainty_style="sgp"):
    """
    Default setting for using FLARE pair_style, which requires `newton on`. The
    `pair_style`, `pair_coeff` and the `compute` command for uncertainty are presented
    """
    preset_command = preset_atoms(newton="on", read_restart=read_restart)

    mean_file = os.path.join(coeff_dir, config.COEFF + ".flare")
    if uncertainty_style == "map":
        var_file = os.path.join(coeff_dir, config.COEFF + ".flare.std")
    elif uncertainty_style == "sgp":
        L_inv_file = os.path.join(coeff_dir, f"L_inv_{config.COEFF}.flare")
        sparse_desc_file = os.path.join(coeff_dir, f"sparse_desc_{config.COEFF}.flare")
        var_file = f"{L_inv_file} {sparse_desc_file}"

    if compute_unc:
        compute_unc_command = f"""
compute             unc all flare/std/atom {var_file}
"""
    else:
        compute_unc_command = ""

    ff_command = f"""
pair_style          flare
pair_coeff          * * {mean_file}
{compute_unc_command}
"""

    return preset_command + ff_command


def vanilla_md(
    init_temp,
    dump_freq,
    n_steps,
    N_iter,
    timestep=0.001,
    fix_commands={"fix 1": "all nve"},
    group_commands=None,
    output_commands=None,
    seed=12345,
    read_restart=None,
    tol=None,
):
    """
    Default setting for MD. Only some basic commands including velocity initialization,
    dumping and thermostat.
    The `c_unc` is crucial since it indicates the LAMMPS uncertainty
    """

    # group atoms
    group_cmd = ""
    if group_commands is not None:
        for key in group_commands:
            group_cmd += key.ljust(20, " ") + group_commands[key] + "\n"

    # if restart from file, then do not initialize velocites
    vel_cmd = ""
    if read_restart is None:
        if (not isinstance(init_temp, str)) or ("velocity" not in init_temp):
            vel_cmd = f"velocity            all create {init_temp} {seed} dist gaussian rot yes mom yes\n"
        else:
            vel_cmd = init_temp + "\n"

    # add customized fix commands
    fix_cmd = ""
    if len(fix_commands) > 0:
        for key in fix_commands:
            fix_cmd += key.ljust(20, " ") + fix_commands[key] + "\n"

    # add customized compute commands
    if tol is not None:
        unc_cmd = "compute MaxUnc      all reduce max c_unc\n"
        c_unc = "c_unc"
        c_MaxUnc = "$(c_MaxUnc)"
    else:
        unc_cmd = ""
        c_unc = ""
        c_MaxUnc = ""

    # Add output commands including dump, thermo.
    # Here we use `fix print` to replace `thermo` because `thermo` sometimes miss lines
    output_cmd = ""
    if output_commands is None:
        output_cmd = f"""
dump                dump_all all custom {dump_freq} {config.LMP_TRJ} id type x y z vx vy vz fx fy fz {c_unc}
fix                 thermoprint all print {dump_freq} "$(step) $(temp) $(ke) $(pe) $(etotal) $(pxx) $(pyy) $(pzz) $(pyz) $(pxz) $(pxy) {c_MaxUnc}" append {config.LMP_THERMO}
"""
    else:
        for key in output_commands:
            output_cmd += key.ljust(20, " ") + output_commands[key] + "\n"

    # check uncertainty if the tolerance is given
    if tol is not None:
        # using an absolute threshold for uncertainty
        run_cmd = f"""
variable            abstol equal {np.abs(tol)}
variable            UncMax equal c_MaxUnc
variable            a loop {N_iter}
label               loopa
    run             {n_steps}
    if "${{UncMax}} > ${{abstol}}" then &
        "print 'Iteration $a has uncertainty above threshold ${{abstol}}'" &
        "jump {config.LMP_IN} break"
    next            a
jump {config.LMP_IN} loopa
label break
"""
    else:
        # not use uncertainty, only run a non-Bayesian plain MD
        run_cmd = f"""run             {n_steps}"""

    md_cmd = f"""
{group_cmd}
{vel_cmd}
{fix_cmd}
{unc_cmd}
timestep            {timestep}     # set timestep
{output_cmd}
{run_cmd}

write_restart       {config.LMP_RESTART}  # write to restart file for the next run
"""
    return md_cmd
