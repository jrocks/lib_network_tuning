import numpy
import glob
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

home = "/data1/home/rocks/"
src_dir = "./"

sources = [src_dir+"cython_src/mech_network_solver.pyx"]
sources.extend(glob.glob(src_dir+"cpp_src/*.cpp"))


setup(
    ext_modules = cythonize(Extension(
        "mech_network_solver", 
        sources=sources,
        include_dirs=[src_dir+"cpp_src", 
        src_dir+"cython_src", 
        numpy.get_include(), 
        home+"/alglib/include"],
        library_dirs=[home+"/alglib/lib"],
        libraries=["m", "rt", "alglib", "suitesparseconfig", "umfpack", "cholmod", "blas", "lapack", "amd","camd", "colamd", "ccolamd", "metis"],
        language="c++",
        extra_compile_args = ["-g0"]
        ))
)

# setup(
#     ext_modules = cythonize(Extension(
#         "mech_network_solver", 
#         sources=sources,
#         include_dirs=[src_dir+"cpp_src", 
#         src_dir+"cython_src", 
#         numpy.get_include(), 
#         home+"/alglib/include", 
#         "/data1/jcode/local/include/eigen3.2.0",
#         home+"SuiteSparse/UFconfig",
#         home+"SuiteSparse/UMFPACK/Include", 
#         home+"SuiteSparse/CHOLMOD/Include",
#         home+"SuiteSparse/AMD/Include", 
#         home+"SuiteSparse/CAMD/Include",
#         home+"SuiteSparse/COLAMD/Include",
#         home+"SuiteSparse/CCOLAMD/Include"],
#         library_dirs=[home+"/alglib/lib",
#         home+"SuiteSparse/UFconfig",
#         home+"SuiteSparse/UMFPACK/Lib",
#         home+"SuiteSparse/CHOLMOD/Lib",
#         home+"SuiteSparse/AMD/Lib",
#         home+"SuiteSparse/CAMD/Lib",
#         home+"SuiteSparse/COLAMD/Lib",
#         home+"SuiteSparse/CCOLAMD/Lib",
#         home+"SuiteSparse/metis-4.0"],
#         libraries=["m", "rt", "alglib", "ufconfig", "umfpack", "cholmod", "blas", "lapack", "amd","camd", "colamd", "ccolamd", "metis"],
#         language="c++",
#         extra_compile_args = ["-g0"]
#         ))
# )