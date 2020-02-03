"""
any DFT interface can be added here, as long as two functions listed below are implemented

* parse_dft_input(dft_input)
* dft_module.run_dft_par(dft_input, structure, dft_loc, no_cpus)
"""

from flare.dft_interface import qe_util, cp2k_util#, vasp_util
dft_software = { "qe": qe_util,
                 #"vasp": vasp_util,
                 "cp2k": cp2k_util}

try:
    import pymatgen
    from flare.dft_interface import vasp_util
    dft_software["vasp"]=vasp_util
except ImportError:
    pass
