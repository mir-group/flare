"""
a dummy to reproduce the kernels.mc_simple module
such that the flare.mc_simple import can work as it was
"""

import flare.kernels.mc_simple as origin

two_plus_three_body_mc = origin.two_plus_three_body_mc
two_plus_three_body_mc_grad = origin.two_plus_three_body_mc_grad
two_plus_three_mc_force_en = origin.two_plus_three_mc_force_en
two_plus_three_mc_en = origin.two_plus_three_mc_en
three_body_mc = origin.three_body_mc
three_body_mc_grad = origin.three_body_mc_grad
three_body_mc_force_en = origin.three_body_mc_force_en
three_body_mc_en = origin.three_body_mc_en
two_body_mc = origin.two_body_mc
two_body_mc_grad = origin.two_body_mc_grad
two_body_mc_force_en = origin.two_body_mc_force_en
two_body_mc_en = origin.two_body_mc_en
str_to_mc_kernel = origin.str_to_mc_kernel
_str_to_kernel = origin._str_to_kernel
