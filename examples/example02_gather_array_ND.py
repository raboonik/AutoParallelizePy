"""
    Summary: A simple MPI program to demonstrate 
    the AutoParallelizePy utilities:
        function get_subarray_ND
        function gather_array_ND
        class    domainDecomposeND
    
    Aims: Create a 2D array to work as our test 
    input data. Use get_subarray_ND for each proc 
    to take a chunk of the data according to some
    domain decomposition scheme and use gather_array_ND
    to gather all the subarrays back into another 
    array on the main rank which recovers the original
    array.
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

# Create the 2D input array
arrShape = [57,89]
Arr = np.arange(np.prod(arrShape)).reshape(arrShape) * np.pi

# Configure the domain decomposition scheme such that
# both axes of the 2D data is parallelized
parallel_axes = [0,1]
domDecompND   = APP.domainDecomposeND(size,arrShape,parallel_axes)

# Have each proc to take a chunk of the input data
myArr = APP.get_subarray_ND(rank,domDecompND,Arr)

# Gather the subarrays back on the mainrank and compare 
# with the original data
gatheredArrOnMainRank = APP.gather_array_ND(comm, rank, mainrank, domDecompND, myArr, 'float')

if rank == mainrank:
    print("")
    if np.all(gatheredArrOnMainRank == Arr):
        print("The original data was successfully reconstructed!")
    else:
        print("Failed!")
    
    print("Running example02 took ",time.time() - start, " seconds!")