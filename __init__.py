'''
    AutoParallelizePy
    
    Exposing the core functions of the package
    
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
'''

# Import from the subpackage
from .libs.domainDecomposeND import domainDecomposeND
from .libs import funcs, mpi

# Optionally still expose libs as a namespace
from . import libs

# Define what the top-level package exposes
__all__ = [
    "domainDecomposeND",
    "funcs",
    "mpi",
    "libs",  # Optional: remove if you don't want APP.libs to be available
]
