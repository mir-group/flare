from flare.kernels.kernels import str_to_kernel
from flare.kernels.mc_simple import str_to_mc_kernel
from flare.kernels.mc_sephyps import str_to_mc_kernel as str_to_mc_sephyps_kernel


def str_to_kernels(name: str, multihyps: bool =False,
        include_grad: bool = False):
    """
    return kernels and kernel gradient base on a string

    Args:

    name (str):
    multihyps (bool):
    include_grad (bool) : whether gradient should be include

    """

    if (include_grad):
        if 'mc' in name:
            if (multihyps is False):
                force_kernel, grad = \
                    str_to_mc_kernel(name, include_grad=True)
            else:
                force_kernel, grad = \
                    str_to_mc_sephyps_kernel(name,
                            include_grad=True)
        else:
            force_kernel, grad = str_to_kernel(name,
                                               include_grad=True)
        return force_kernel, grad
    else:
        if 'mc' in name:
            if (multihyps is False):
                kernel = str_to_mc_kernel(name)
            else:
                kernel = str_to_mc_sephyps_kernel(name)
        else:
            kernel = str_to_kernel(name)
        return grad
