
"""
    Summary: A simple MPI program to demonstrate 
    the AutoParallelizePy utilities:
        function reshape_array_ND
        function get_subarray_ND
        function gather_array_ND
        class    domainDecomposeND
    
    Aims: Create a 3D array as the input data
    and use get_subarray_ND to have each proc 
    take a slice of it according to some domain 
    decomposition rule. Then use reshape_array_ND
    to reshape the subarrays according to a new
    domain decomposition rule.
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

# Create the 3D input array
arrShape = [57,89,62]
Arr = np.arange(np.prod(arrShape)).reshape(arrShape) * np.pi

# Configure the domain decomposition scheme such that
# both axes of the 3D data is parallelized
parallel_axes = [0,1,2]
domDecompND   = APP.domainDecomposeND(size,arrShape,parallel_axes)

# Have each proc to take a chunk of the input data
myArr = APP.get_subarray_ND(rank,domDecompND,Arr)

# Reconfigure the domain decomposition scheme, this time only
# parallelize the first and third axes
new_parallel_axes = [0,2]
new_DomDecompND   = APP.domainDecomposeND(size,arrShape,new_parallel_axes)

# Reshape the subarrays according to the new domain decomposition rule
new_myArr = APP.reshape_array_ND(comm, rank, mainrank, domDecompND, new_DomDecompND, myArr,  dtype='float')

# Gather both
gatheredArrOnMainRank     = APP.gather_array_ND(comm, rank, mainrank, domDecompND, myArr, 'float')
new_gatheredArrOnMainRank = APP.gather_array_ND(comm, rank, mainrank, new_DomDecompND, new_myArr, 'float')

if rank == mainrank:
    print("")
    if (np.all(Arr == gatheredArrOnMainRank) and
        np.all(Arr == new_gatheredArrOnMainRank)):
        print("Reshaping successful!")
    else:
        print("Failed!")
    
    print("Running example05 took ",time.time() - start, " seconds!")