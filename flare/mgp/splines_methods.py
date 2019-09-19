import numpy as np
import numpy
import sys
sys.path.append('../../flare')
import time
from memory_profiler import profile

import flare.mgp.utils as utils
from flare.mgp.cubic_splines_numba import *

class PCASplines:
    '''
    build splines for PCA decomposition, mainly used for the mapping of the variance
    '''
    def __init__(self, y, u_bounds, l_bounds, orders, svd_rank):
        self.svd_rank = svd_rank
        self.model = self.build_pca_cubic(y, u_bounds, l_bounds, orders)

    def build_cubic(self, y, u_bounds, l_bounds, orders):
        dim_0 = 1
        for d in range(len(y.shape)-1):
            dim_0 *= y.shape[d]
        dim_1 = y.shape[-1]

        var_matr = np.reshape(y, (dim_0, dim_1))
        models = []
        for r in range(self.svd_rank):
            spline_u = CubicSpline(l_bounds, u_bounds, orders, var_matr[:, r])
            models.append(spline_u)
        return models  
        
    def build_pca_cubic(self, y, u_bounds, l_bounds, orders):
        dim_0 = 1
        for d in range(len(y.shape)-1):
            dim_0 *= y.shape[d]
        dim_1 = y.shape[-1]
       
        var_matr = np.reshape(y, (dim_0, dim_1))
        U, S, Vh = utils.svd_grid(var_matr, rank=self.svd_rank)
        self.V = Vh.T
        models = []
        for r in range(self.svd_rank):
            spline_u = CubicSpline(l_bounds, u_bounds, orders, S[r]*U[:, r])
            models.append(spline_u)
        return models      
        
    def __call__(self, x):
        y_pred = []
        for r in range(self.svd_rank):
            y_pred.append(self.model[r](x))
        return np.array(y_pred)



class SplinesInterpolation:
    
    def __init__(self, y, u_bounds, l_bounds, orders):
        self.model = self.build_cubic(y, u_bounds, l_bounds, orders)
 
    def build_cubic(self, y, u_bounds, l_bounds, orders):
        spline = CubicSpline(l_bounds, u_bounds, orders, y)
        return spline        

    def __call__(self, x):
        return self.model(x)



"""High-level API for cubic splines"""

class CubicSpline:

    """Class representing a cubic spline interpolator on a regular cartesian grid.."""

    __grid__ = None
    __values__ = None
    __coeffs__ = None

    def __init__(self, a, b, orders, values=None):
        """Creates a cubic spline interpolator on a regular cartesian grid.

        Parameters:
        -----------
        a : array of size d (float)
            Lower bounds of the cartesian grid.
        b : array of size d (float)
            Upper bounds of the cartesian grid.
        orders : array of size d (int)
            Number of nodes along each dimension (=(n1,...,nd) )
        values : (optional, (n1 x ... x nd) array)
            Values on the nodes of the function to interpolate.

        Returns
        -------
        spline : CubicSpline
            Cubic spline interpolator. Can be evaluated at point(s) `y` with
            `spline(y)`
        """


        self.d = len(a)
        assert(len(b) == self.d)
        assert(len(orders) == self.d)
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.orders = np.array(orders, dtype=int)
        self.dtype =  self.a.dtype
        self.__coeffs__ = None

        if values is not None:
            self.set_values(values)


    def set_values(self, values):
        '''Set values on the nodes for the function to interpolate.'''

        values = np.array(values, dtype=float)

        if not np.all( np.isfinite(values)):
            raise Exception('Trying to interpolate non-finite values')

        sh = self.orders.tolist()
        sh2 = [ e+2 for e in self.orders]

        values = values.reshape(sh)

        self.__values__ = values

        # this should be done without temporary memory allocation
        self.__coeffs__ = filter_coeffs(self.a, self.b, self.orders, values)



    def interpolate(self, points, values=None, with_derivatives=False):
        '''Interpolate spline at a list of points.

        Parameters
        ----------
        points : (array-like) list of point where the spline is evaluated.
        values : (optional) container for inplace computation

        Returns
        -------
        values : (array-like) list of point where the spline is evaluated.
        '''

        if not np.all( np.isfinite(points)):
            raise Exception('Spline interpolator evaluated at non-finite points.')

        if not with_derivatives:
            if points.ndim == 1:
                # evaluate only on one point
                return eval_cubic_spline(self.a, self.b, self.orders, self.__coeffs__, points)
            else:

                N, d = points.shape
                assert(d==self.d)
                if values is None:
                    values = np.empty(N, dtype=self.dtype)
                vec_eval_cubic_spline(self.a, self.b, self.orders, self.__coeffs__, points, values)
                return values
        else:
            raise Exception("Not implemented.")


    @property
    def grid(self):
        """Cartesian enumeration of all nodes."""

        if self.__grid__ is None:
            self.__grid__ = mlinspace(self.a, self.b, self.orders)
        return self.__grid__

    def __call__(self, s):
        """Interpolate the spline at one or many points"""

        if s.ndim == 1:
            res = self.__call__( numpy.atleast_2d(s) )
            return res[0]

        return self.interpolate(s)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int( n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def mlinspace(a,b,orders,out=None):
    sl = [numpy.linspace(a[i],b[i],orders[i]) for i in range(len(a))]

    if out is None:
        out = cartesian(sl)
    else:
        cartesian(sl, out)

    return out



## the functions in this file work for any dimension (d<=4)
## they can optionnally allocate memory for the result


def eval_cubic_spline(a, b, orders, coefs, point):
    """Evaluates a cubic spline at one point

    Parameters:
    -----------
    a : array of size d (float)
        Lower bounds of the cartesian grid.
    b : array of size d (float)
        Upper bounds of the cartesian grid.
    orders : array of size d (int)
        Number of nodes along each dimension (=(n1,...,nd) )
    coefs : array of dimension d, and size (n1+2, ..., nd+2)
        Filtered coefficients.
    point : array of size d
        Coordinate of the point where the splines must be interpolated.

    Returns
    -------
    value : float
        Interpolated value.
    """

    a = numpy.array(a, dtype=float)
    b = numpy.array(b, dtype=float)
    orders = numpy.array(orders, dtype=int)

    d = len(a)

    if d == 1:
        value = eval_cubic_spline_1(a, b, orders, coefs, point)

    elif d == 2:
        value = eval_cubic_spline_2(a, b, orders, coefs, point)

    elif d == 3:
        value = eval_cubic_spline_3(a, b, orders, coefs, point)

    elif d == 4:
        value = eval_cubic_spline_4(a, b, orders, coefs, point)

    return value


def vec_eval_cubic_spline(a, b, orders, coefs, points, values=None):
    """Evaluates a cubic spline at many points

    Parameters:
    -----------
    a : array of size d (float)
        Lower bounds of the cartesian grid.
    b : array of size d (float)
        Upper bounds of the cartesian grid.
    orders : array of size d (int)
        Number of nodes along each dimension. (=(n1,...,nd))
    coefs : array of dimension d, and size (n1+2, ..., nd+2)
        Filtered coefficients.
    points : array of size N x d
        List of points where the splines must be interpolated.
    values (optional) :  array of size (N)
        If not None, contains the result.

    Returns
    -------
    values : array of size (N)
        Interpolated values. values[i] contains spline evaluated at point points[i,:].
    """

    a = numpy.array(a, dtype=float)
    b = numpy.array(b, dtype=float)
    orders = numpy.array(orders, dtype=int)

    d = a.shape[0]

    if values is None:
        N = points.shape[0]
        values = numpy.empty(N)

    if d == 1:
        vec_eval_cubic_spline_1(a, b, orders, coefs, points, values)
    elif d == 2:
        vec_eval_cubic_spline_2(a, b, orders, coefs, points, values)
    elif d == 3:
        vec_eval_cubic_spline_3(a, b, orders, coefs, points, values)
    elif d == 4:
        vec_eval_cubic_spline_4(a, b, orders, coefs, points, values)

    return values


