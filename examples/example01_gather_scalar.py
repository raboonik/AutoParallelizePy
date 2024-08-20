"""
    Summary: A simple MPI program to demonstrate 
    the AutoParallelizePy utilities:
        function gather_scalar
    
    Aims: Create scalar numbers on each proc and
    gather them on the main rank and print the 
    results.
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

# Define a traceable scalar on each rank
scalarNum = rank

# Gather on the main rank and print
out = APP.gather_scalar(comm, size, rank, mainrank, scalarNum, dtype='float')
if rank == mainrank:
    print("Gathered data on the main rank = ", out)