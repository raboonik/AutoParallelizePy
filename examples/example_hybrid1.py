"""
    Summary: A hybrid example program to test and demonstrate 
    the use of the following AutoParallelizePy methods:
        function create_randoms_acorss_cores
        class    domainDecomposeND
        function get_subarray_ND
        function scatter_array_ND
        function gather_array_ND
    
    Aims: Create the same 4D array of random real numbers across
    all procs using create_randoms_acorss_cores, domain decompose 
    and parallelize it, and then gather the subarrays on the main 
    rank to recover the original array.
    
    Tip: create_randoms_acorss_cores creates an array of random
    real numbers and broadcasts it to all the procs.
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

# Using create_randoms_acorss_cores, create a 4D array of random 
# floats to be parallelized only in the first, second, and fourth 
# dimensions and broadcast to all procs
dataShape     = [27,56,34,86]
parallel_axes = [0,1,3]
Arr           = APP.create_randoms_acorss_cores(comm, rank, mainrank, dataShape)

# Using domainDecomposeND Configure the automatic domain decomposition  
# scheme based on the prescribed data shape and parallelized axes
domDecompND = APP.domainDecomposeND(size,dataShape,parallel_axes)

# Use get_subarray_ND to have each proc take a slice of the original array
myarr_getSubarray = APP.get_subarray_ND(rank,domDecompND,Arr) 

# Now let's slice the original array this time using scatter_array_ND
# To do so, frist store the original array on the main rank
if rank == mainrank:
    copyArr = Arr.copy()
else:
    copyArr = None

# Use scatter_array_ND to scatter chunks/slices of the original array
# across all procs as prescribed in domDecompND
myarr_scatterND = APP.scatter_array_ND(comm,rank,mainrank,domDecompND,copyArr,dtype='float')

# Check that the two versions do indeed are the same
if np.all(myarr_getSubarray == myarr_scatterND):
    print("rank = {} -- Success! The two versions of local sub-arrays sliced using get_subarray_ND and scatter_array_ND yielded the same results!".format(rank))
else:
    print("rank = {} -- Failed!".format(rank))

# Use gather_array_ND to gather all the local data chunks of myarr_getSubarray  
# and retrieve the original data
gathered_myArr = APP.gather_array_ND(comm, rank, mainrank, domDecompND, myarr_getSubarray, 'float')

# Check if the gathered data recovers the original array
if rank == mainrank:
    print("")
    if np.all(gathered_myArr == Arr):
        print("Success! The original data was correctly retrieved!")
    else:
        print("Failed!")
    
    print("Running this exampel program took ",time.time() - start, " seconds!")