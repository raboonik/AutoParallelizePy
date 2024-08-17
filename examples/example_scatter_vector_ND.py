import numpy as np
from mpi4py import * 
import AutoParallelizePy

#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#      Init Parallel        #◈
comm     = MPI.COMM_WORLD   #◈
size     = comm.Get_size()  #◈
rank     = comm.Get_rank()  #◈
mainrank = 0                #◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈

# Create a 4D array of random floats
axes_limits       = [27,56,34,86]
parallel_axes     = [0,1,3]
new_parallel_axes = [1,2]

testArr = AutoParallelizePy.create_randoms_acorss_cores(comm, rank, mainrank, axes_limits)

if rank == mainrank:
    lowHigh=[-13.54,13.3]
    mainArr = testArr.copy()
else:
    mainArr = None

# Domain decompose
domDecompND = AutoParallelizePy.domainDecomposeND(size,axes_limits,parallel_axes)

myarr0 = AutoParallelizePy.get_subarray(rank,domDecompND,testArr) 
myarr  = AutoParallelizePy.scatter_vector_ND(comm,rank,mainrank,domDecompND,mainArr,dtype='float')

if np.all(myarr0 == myarr):
    print("rank = {} -- Success!".format(rank))
else:
    print("rank = {} -- Failed!".format(rank))

gathered_new_myTestArr = AutoParallelizePy.gather_vector_ND(comm, rank, mainrank, domDecompND, myarr0, 'float')
gathered_new_myTestArr1 = AutoParallelizePy.gather_vector_ND(comm, rank, mainrank, domDecompND, myarr, 'float')

if rank == mainrank:
    print("")
    if np.all(gathered_new_myTestArr == testArr):
        print("Success! gathered_new_myTestArr = testArr")
    else:
        print("Failed! gathered_new_myTestArr != testArr")
    print("")
    if np.all(gathered_new_myTestArr1 == testArr):
        print("Success! gathered_new_myTestArr1 = testArr")
    else:
        print("Failed! gathered_new_myTestArr1 != testArr")