# -*- coding: utf-8 -*-
#
"""Setuptools-based setup script for PyLU."""

from __future__ import absolute_import

#########################################################
# Config
#########################################################

# choose build type here
#
build_type="optimized"
#build_type="debug"


#########################################################
# Init
#########################################################

# check for Python 2.7 or later
# http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
import sys
if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')

import os

from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension modules for PyLU.")


#########################################################
# Definitions
#########################################################

extra_compile_args_math_optimized    = ['-fopenmp', '-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
extra_compile_args_math_debug        = ['-fopenmp', '-march=native', '-O0', '-g']

extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug     = ['-O0', '-g']

extra_link_args_optimized    = ['-fopenmp']
extra_link_args_debug        = ['-fopenmp']


if build_type == 'optimized':
    my_extra_compile_args_math    = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args            = extra_link_args_optimized
    debug = False
    print "build configuration selected: optimized"
else: # build_type == 'debug':
    my_extra_compile_args_math    = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args            = extra_link_args_debug
    debug = True
    print "build configuration selected: debug"


#########################################################
# Long description
#########################################################

DESC="""Solve A x = b.

PyLU uses LU decomposition with partial pivoting (row swaps),
and requires only NumPy and Cython.

The main use case for PyLU (over numpy.linalg.solve) is solving many
small systems inside a nogil block in Cython code, without requiring
SciPy (for its cython_lapack module).

Python and Cython interfaces are provided. The API is designed
to be as simple to use as possible.
"""


#########################################################
# Helpers
#########################################################

my_include_dirs = ["."]  # IMPORTANT, see https://github.com/cython/cython/wiki/PackageHierarchy

def ext(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension( extName,
                      [extPath],
                      extra_compile_args=my_extra_compile_args_nonmath
                    )
def ext_math(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension( extName,
                      [extPath],
                      extra_compile_args=my_extra_compile_args_math,
                      extra_link_args=my_extra_link_args,
                      libraries=["m"]  # "m" links libm, the math library on unix-likes; see http://docs.cython.org/src/tutorial/external.html
                    )

# http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
datadir = "test"
datafiles = [(root, [os.path.join(root, f) for f in files if f.endswith(".py")])
    for root, dirs, files in os.walk(datadir)]

datafiles.append( ('.', ["README.md", "LICENSE.md"]) )


#########################################################
# Modules
#########################################################

ext_module_dgesv = ext_math( "pylu.dgesv" )

#########################################################

# Extract __version__ from the package __init__.py
# (since it's not a good idea to actually run __init__.py during the build process).
#
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
#
import ast
with file('pylu/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            version = ast.parse(line).body[0].value.s
            break
    else:
        version = '0.0.unknown'
        print "WARNING: Version information not found, using placeholder '%s'" % (version)


setup(
    name = "pylu",
    version = version,
    author = "Juha Jeronen",
    author_email = "juha.jeronen@jyu.fi",
    url = "https://github.com/Technologicat/PyLU",

    description = "small nogil-compatible linear equation system solver",
    long_description = DESC,

    license = "BSD",
    platforms = ["Linux"],  # free-form text field; http://stackoverflow.com/questions/34994130/what-platforms-argument-to-setup-in-setup-py-does

    classifiers = [ "Development Status :: 4 - Beta",
                    "Environment :: Console",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: BSD License",
                    "Operating System :: POSIX :: Linux",
                    "Programming Language :: Cython",
                    "Programming Language :: Python",
                    "Programming Language :: Python :: 2",
                    "Programming Language :: Python :: 2.7",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Mathematics",
                    "Topic :: Software Development :: Libraries",
                    "Topic :: Software Development :: Libraries :: Python Modules"
                  ],

    setup_requires = ["cython", "numpy"],
    install_requires = ["numpy"],
    provides = ["pylu"],

    # same keywords as used as topics on GitHub
    keywords = ["numerical linear-equations cython"],

    ext_modules = cythonize( [ ext_module_dgesv ],
                             include_path = my_include_dirs,
                             gdb_debug = debug ),

    # Declare packages so that  python -m setup build  will copy .py files (especially __init__.py).
    packages = ["pylu"],

    # Install also Cython headers so that other Cython modules can cimport ours
    # FIXME: force sdist, but sdist only, to keep the .pyx files (this puts them also in the bdist)
    package_data={'pylu': ['*.pxd', '*.pyx']},  # note: paths relative to each package

    # Disable zip_safe, because:
    #   - Cython won't find .pxd files inside installed .egg, hard to compile libs depending on this one
    #   - dynamic loader may need to have the library unzipped to a temporary folder anyway (at import time)
    zip_safe = False,

    # Usage examples; not in a package
    data_files = datafiles
)

