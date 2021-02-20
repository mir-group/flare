# flare++
Documentation can be accessed [here](https://mir-group.github.io/flare_pp/).

## Usage

If compiling on Cannon, run the following:
```
module load cmake/3.17.3-fasrc01
module load python/3.6.3-fasrc01
module load gcc/9.3.0-fasrc01
```

And set the following environment variables (to link Eigen to MKL):
```
export MKL_INCLUDE=/n/sw/intel-cluster-studio-2019/mkl/include
export MKL_ROOT=/n/sw/intel-cluster-studio-2019/mkl/lib/intel64_lin
export MKL_INT=$MKL_ROOT/libmkl_intel_lp64.so
export MKL_THREAD=$MKL_ROOT/libmkl_gnu_thread.so
export MKL_CORE=$MKL_ROOT/libmkl_core.so
export MKL_AVX=$MKL_ROOT/libmkl_avx512.so
export MKL_DEF=$MKL_ROOT/libmkl_def.so
export MKL_VML=$MKL_ROOT/libmkl_vml_avx512.so
export MKL_LIBS="$MKL_CORE;$MKL_INT;$MKL_THREAD;$MKL_AVX;$MKL_DEF;$MKL_VML"
```
