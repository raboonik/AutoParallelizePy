"""
    Summary: A simple MPI program to demonstrate 
    the AutoParallelizePy utilities:
        function bcast
    
    Aims: Create a scalar, a 1D array, and a 3D
    array on the main rank and use bcast to copy 
    them onto other procs. 
    
    Tip: Broadcasting is the copying of the same
    scalar or array onto all the other procs.
"""

import time
start = time.time()

import numpy as np
from mpi4py import * 
import AutoParallelizePy as APP

# Initialize the MPI environment
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#      Init Parallel        #◈
comm     = MPI.COMM_WORLD   #◈
size     = comm.Get_size()  #◈
rank     = comm.Get_rank()  #◈
mainrank = 0                #◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈

# Create a scalar integer on the main rank and bcast to other procs
if rank == mainrank:
    intScalar = 5
else:
    intScalar = None

bcastIntScalar = APP.bcast(comm, rank, mainrank, intScalar, dtype='int')

# Create a 1D array of length 127 of random real numbers on the main
# rank and bcast to other procs
if rank == mainrank:
    arr1D = np.random.uniform(low=-20, high=20, size=([127]))
else:
    arr1D = None

bcastArr1D = APP.bcast(comm, rank, mainrank, arr1D, dtype='float')

# Create a 3D array of shape [23,34,67] of random real numbers on  
# the mainrank and bcast to other procs
if rank == mainrank:
    arr3D = np.random.uniform(low=-20, high=20, size=([23,34,67]))
else:
    arr3D = None

bcastArr3D = APP.bcast(comm, rank, mainrank, arr3D, dtype='float')

if rank == mainrank:
    print("")
    if        (intScalar == bcastIntScalar and
        np.all(arr1D     == bcastArr1D)    and   
        np.all(arr3D     == bcastArr3D)):
        print("Broadcasting successful!")
    else:
        print("Failed!")     
    
    print("Running example04 took ",time.time() - start, " seconds!")