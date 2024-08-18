"""
    Summary: A simple single-core example program 
    emulating a hypothetical MPI environment consisting
    of 32 processes to test and demonstrate the 
    AutoParallelizePy class:
        class    domainDecomposeND
    
    Aims: Create a simple 4D array of arbitrary
    shape and auto-domain decompose using the
    built-in domainDecomposeND class and recover
    the original array after decomposition. 
"""

import numpy as np
import AutoParallelizePy as APP

# Even though this example isn't MPI-parallelized, we may emulate a
# hypothetical MPI environment assuming there are a total of 32 procs
size = 32

# Create a simple 4D array of random numbers
arrShape = [47,37,19,98]
origArr   = np.random.uniform(low=-20, high=20, size=(arrShape))

# Configure the domain decomposition scheme such that only the first,
# second, and fourth dimensions of this array are parallelized
parallel_axes = [0,1,3]
domDecompND   = APP.domainDecomposeND(size,arrShape,parallel_axes)

# Now let's have our hypothetical procs (ranks) each take a slice of 
# the data and dump it in a new array to retrieve the original data 
testArr = np.zeros(arrShape)
for rank in range(size):
    slq0 = domDecompND.slq[0][rank]
    elq0 = domDecompND.elq[0][rank]
    slq1 = domDecompND.slq[1][rank]
    elq1 = domDecompND.elq[1][rank]
    slq2 = domDecompND.slq[2][rank]
    elq2 = domDecompND.elq[2][rank]
    slq3 = domDecompND.slq[3][rank]
    elq3 = domDecompND.elq[3][rank]
    mySubArr = origArr[slq0:elq0,slq1:elq1,slq2:elq2,slq3:elq3]
    # Dump the subarray into the test array in the correct block
    testArr[slq0:elq0,slq1:elq1,slq2:elq2,slq3:elq3] = mySubArr

# If the domain decomposition scheme is correctly done, testArr 
# recovers the original array. Let's check that 
print("*********************************")
if np.all(origArr == testArr):
    print("The original data was successfully recovered!")
else:
    print("Failed!")
print("*********************************")
print("")

# Now let's print the different attributes of the object domDecompND
print("Assuming the data is N-dimensional and MPI-parallelized across a total number of np cores:")
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.nblock"+'\033[0m'+" gives an N-D array of the number of MPI-blocks in each dimension.")
print("The product of all the blocks must recover the total number of cores in an MPI scheme.")
print("In this example we have domDecompND.nblock = ", domDecompND.nblock)
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.coordinates"+'\033[0m'+" gives an Nxnp array of all the Cartesian coordinates of the MPI-blocks.")
print("In this example we have domDecompND.coordinates = ", domDecompND.coordinates)
print("and hence the coordinates of the MPI-block handled by 27th (rank = 27) processor is:",domDecompND.coordinates[:,27])
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.slq"+'\033[0m'+" gives an Nxnp array of all the datapoint indices that mark the begining of each MPI-block dimension.")
print("In this example we have domDecompND.slq = ", domDecompND.slq)
print("and hence the MPI-block belonging to the 27th processor starts at:",domDecompND.slq[:,27])
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.elq"+'\033[0m'+" gives an Nxnp array of all the datapoint indices that mark the end of each MPI-block dimension.")
print("In this example we have domDecompND.elq = ", domDecompND.elq)
print("and hence the MPI-block belonging to the 27th processor ends at:",domDecompND.elq[:,27])
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.mynq"+'\033[0m'+" gives an Nxnp array of the total number of gridpoints of each MPI-block dimension.")
print("Note that we have domDecompND.mynq = domDecompND.elq - domDecompND.slq")
print("In this example we have domDecompND.mynq = ", domDecompND.mynq)
print("and hence the MPI-block belonging to the 27th processor has the following number of gridpoints in each dimension:",domDecompND.mynq[:,27])
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.split_sizes"+'\033[0m'+" gives an array of length np of the total number of gridpoints handled by each processor.")
print("Note that we have domDecompND.split_sizes == np.prod(domDecompND.mynq, axis=0)")
print("In this example we have domDecompND.split_sizes = ", domDecompND.split_sizes)
print("and hence the 27th processor has the following total number of gridpoints:",domDecompND.split_sizes[27])
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.arrShape"+'\033[0m'+" gives the shape of the array to be MPI-parallelized.")
print("In this example we have domDecompND.arrShape = ", domDecompND.arrShape)
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.parallel_axes"+'\033[0m'+" gives the user's input list of axes/dimensions to be parallelized.")
print("In this example we have domDecompND.parallel_axes = ", domDecompND.parallel_axes)
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.n_dim"+'\033[0m'+" gives the number of dimensions of the raw array.")
print("In this example we have domDecompND.n_dim = ", domDecompND.n_dim)
print("-------------------------------------------------------------------")
print('\033[1m'+"domDecompND.n_par_dim"+'\033[0m'+" gives the number of parallelized dimensions.")
print("In this example we have domDecompND.n_par_dim = ", domDecompND.n_par_dim)
print("-------------------------------------------------------------------")