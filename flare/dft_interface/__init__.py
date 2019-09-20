# import flare
# print('hello', dir(flare))
from flare.dft_interface import qe_util, cp2k_util
dft_software = { "qe":qe_util,
                 "cp2k":cp2k_util}
