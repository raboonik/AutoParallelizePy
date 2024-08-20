"""
    Summary: A simple MPI program to demonstrate 
    the AutoParallelizePy utilities:
        function scatter_array_ND
        class    domainDecomposeND
    
    Aims: Create a 2D array ON THE MAIN RANK to work
    as our test input data. Use scatter_array_ND to 
    SCATTER chunks of this data according to some
    domain decomposition scheme to all the other cores.
    Finally use gather_array_ND to gather all the sub-
    arrays back into another array on the main rank 
    which recovers the original array.
    
    Tip1: Note that as opposed to example02 where we 
    assumed the input data was available on all the 
    procs, here the input data only exists on the main 
    rank, which makes it more memory efficient. However,
    since scattering the data potentially takes more time 
    than slicing it in situ, which makes it slightly less 
    time efficient.
    
    Tip2: Scattering only works for sending subarrays of
    an array on a source proc to other procs. Use broadcasting
    to send a scalar to other procs.
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

# Create the same 2D input array as example02_gather_array_ND
# this time only on the main rank, and set to None on other procs
arrShape = [57,89]
if rank == mainrank: 
    Arr = np.arange(np.prod(arrShape)).reshape(arrShape) * np.pi
else:
    Arr = None

# Configure the domain decomposition scheme such that
# both axes of the 2D data is parallelized
parallel_axes = [0,1]
domDecompND   = APP.domainDecomposeND(size,arrShape,parallel_axes)

# Use scatter_array_ND to scatter chunks of the input data to
# other procs
myArr = APP.scatter_array_ND(comm,rank,mainrank,domDecompND,Arr,dtype='float')

# Gather the subarrays back on the mainrank and compare 
# with the original data
gatheredArrOnMainRank = APP.gather_array_ND(comm, rank, mainrank, domDecompND, myArr, 'float')

if rank == mainrank:
    print("")
    if np.all(gatheredArrOnMainRank == Arr):
        print("The original data was successfully reconstructed!")
    else:
        print("Failed!")
    
    print("Running example03 took ",time.time() - start, " seconds!")