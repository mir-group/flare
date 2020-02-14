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

def str_to_kernel_set(name: str, multihyps: bool =False):
    """
    return kernels and kernel gradient base on a string

    Args:

    name (str):
    multihyps (bool):

    """

    if 'mc' in name:
        if (multihyps is False):
            stk = str_to_mc_kernel
        else:
            stk = str_to_mc_sephyps_kernel
    else:
        stk = str_to_kernel

    b2 = False
    b3 = False

    for s in ['2', 'two']:
        if (s in name):
            b2 = True
    for s in ['3', 'three']:
        if (s in name):
            b3 = True
    if (b2 and b3):
        prefix='2+3'
    elif (b2):
        prefix='2'
    elif (b3):
        prefix='3'
    else:
        raise RuntimeError(f"the name has to include at least one number {name}")

    return stk(prefix), stk(prefix+'_grad'), stk(prefix+'_en'), \
            stk(prefix+'_force_en')
