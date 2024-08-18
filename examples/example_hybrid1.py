"""
    Summary: A hybrid example program to test and demonstrate 
    the use of the following AutoParallelizePy methods:
        class    domainDecomposeND
        function get_subarray
        function scatter_vector_ND
        function gather_vector_ND
    
    Aims: Create a 4D array of random real numbers, domain 
    decompose and parallelize it, and then de-parallelize
    to recover the same array.
"""

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

# Using create_randoms_acorss_cores, create a 4D array of random 
# floats to be parallelized in the first, second, and fourth 
# dimensions and broadcast to all procs
dataShape     = [27,56,34,86]
parallel_axes = [0,1,3]
origArr = APP.create_randoms_acorss_cores(comm, rank, mainrank, dataShape)

# Using domainDecomposeND Configure the automatic domain decomposition  
# scheme based on the prescribed data shape and parallelized axes
domDecompND = APP.domainDecomposeND(size,dataShape,parallel_axes)

# Use get_subarray to have each proc take a slice of the original array
myarrV1 = APP.get_subarray(rank,domDecompND,origArr) 

# Now let's slice the original array this time using scatter_vector_ND
# To do so, frist store the original array on the main rank
if rank == mainrank:
    copyArr = origArr.copy()
else:
    copyArr = None

# Use scatter_vector_ND to scatter chunks/slices of the original array
# across all procs as prescribed in domDecompND
myarrV2 = APP.scatter_vector_ND(comm,rank,mainrank,domDecompND,copyArr,dtype='float')

# Check that the two versions do indeed are the same
if np.all(myarrV1 == myarrV2):
    print("rank = {} -- Success! The two versions of local sub-arrays sliced using get_subarray and scatter_vector_ND yielded the same results!".format(rank))
else:
    print("rank = {} -- Failed!".format(rank))

# Use gather_vector_ND to gather all the local data chunks of both versions on the main rank  
# and retrieve the original data
gathered_myArrV1 = APP.gather_vector_ND(comm, rank, mainrank, domDecompND, myarrV1, 'float')
gathered_myArrV2 = APP.gather_vector_ND(comm, rank, mainrank, domDecompND, myarrV2, 'float')

# Check if the gathered data recovers the original array
if rank == mainrank:
    print("")
    if np.all(gathered_myArrV1 == origArr):
        print("Success! gathered_myArrV1 = origArr")
    else:
        print("Failed! gathered_myArrV1 != origArr")
    print("")
    if np.all(gathered_myArrV2 == origArr):
        print("Success! gathered_myArrV2 = origArr")
    else:
        print("Failed! gathered_myArrV2 != origArr")