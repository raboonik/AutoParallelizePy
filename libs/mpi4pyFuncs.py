"""
    Here are the core functions of AutoParallelizePy.
    
    Dependencies:
        domainDecomposeND
        mpi4py
        funcs
    
    Variable naming: 
        comm:        MPI-communicator
        rank:        A processor (proc) in the parallelized scheme
        size:        Total number of procs available
        Arr :        An N-dimensional array
        subArr :     A chunk of Arr handled by a proc
        mainrank:    The integer ID of the single processor in
                     charge of gathering and distributing sub-data
                     from other procs
        domDecompND: Domain decomposition scheme domDecompND (output of domainDecomposeND)
    
    Tip: All the functions suffixed with "_array_ND" depend upon
    the specific domain decomposition domDecompND as an input.
"""

from mpi4py import *
from funcs import *
from domainDecomposeND import *
alphabet = 'abcdefghijklmnopqrstuvwxyz'


#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈ MPI Gather ◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
def gather_scalar(comm, size, rank, mainrank, scalarNum, dtype='int') -> np.array:
    """
        Gather scalars stored locally on each proc (rank) into an array of length size
        stored on the main rank.
        
        Required in: None
        
        Dependencies: None
        
        Inputs: 
            comm, size, rank, mainrank
            Scalar number (scalarNum)
            datatype of the scalars (dtype)
        
        Output: 1D array of length size existing on the main proc with all the scalars
        sorted according to the proc IDs (ranks).
    """
    
    if dtype == 'float':
        dtype    = float
    elif dtype == 'int':
        dtype    = int
    
    if rank == mainrank:
        out = np.zeros(size, dtype)
    else:
        out = None
    
    comm.Gather(sendbuf=np.array(scalarNum,dtype=dtype),recvbuf=out, root=mainrank)
    
    return out


def gather_array_ND(comm, rank, mainrank, domDecompND, subArr, dtype='float') -> np.array:
    """
        Gather ND (sub)-arrays stored locally on each proc into a (main) ND array
        stored on the main rank.
        
        Required in: reshape_array_ND
        
        Dependencies:
            Direct:   None
            Indirect: domDecompND
        
        Inputs: 
            comm, rank, mainrank, domDecompND, subArr
            datatype of the input ND array (dtype)
        
        Output: ND array containing all the subarrays located at block coordinates
        according to the domain decomposition scheme domDecompND.
    """
    
    size          = domDecompND.n_processors
    arrShape      = domDecompND.arrShape
    parallel_axes = domDecompND.parallel_axes
    dim           = len(arrShape)
    
    if len(parallel_axes) == 0: return subArr
    
    if dtype == 'float':
        dtype    = float
        mpidtype = MPI.DOUBLE
    elif dtype == 'int':
        dtype    = int
        mpidtype = MPI.INT
    
    if len(parallel_axes) == 1:
        # subArr is 1D
        
        parallel_axis = parallel_axes[0]
        
        permutationCode_i   = alphabet[:dim]
        if dim > 1:
            # If dim > 1, then we need to make sure that it its the first axis being parallelized, so the data needs permutation of axes
            inxs                = range(dim)
            nonParallelizedInxs = [i for i in inxs if i != parallel_axis]
            permutationCodeInxs = list(np.append(parallel_axis,nonParallelizedInxs))
            
            permutationCode_f   = "".join([permutationCode_i[i] for i in permutationCodeInxs])
            rolled_arrShape  = [arrShape[i] for i in permutationCodeInxs]
            
            subArr = np.einsum(permutationCode_i+'->'+permutationCode_f, subArr)
        else:
            permutationCode_f = permutationCode_i
            rolled_arrShape   = arrShape
        
        split_sizes = domDecompND.split_sizes
        
        if rank == mainrank:
            tempFlat = np.zeros([np.prod(arrShape)],dtype)
        else:
            tempFlat = None
        
        subArrFlat = np.ravel(subArr)
        comm.Gatherv(sendbuf=[subArrFlat , mpidtype], recvbuf=[tempFlat , split_sizes], root=mainrank)
        
        if rank == mainrank:
            out = tempFlat.reshape(rolled_arrShape)
            out = np.einsum(permutationCode_f+'->'+permutationCode_i, out)
        else:
            out = None
    else:
        # subArr is ND with N < 1
        if np.any(np.array(parallel_axes) != np.sort(parallel_axes)): 
            parallel_axes.sort()
        
        myarrShape = np.array(list(subArr.shape),dtype=int)
        
        if rank == mainrank:
            allmyarrShape = np.zeros(size * dim,int)
        else:
            allmyarrShape = None
        
        split_sizes = domDecompND.split_sizes
        
        comm.barrier()
        
        split_sizes_mylLst = dim*np.ones(size,int)
        
        comm.Gatherv(sendbuf=[myarrShape, MPI.INT], recvbuf=[allmyarrShape, split_sizes_mylLst], root=mainrank)
        
        if rank == mainrank: 
            blocks       = np.insert(np.cumsum(split_sizes),0,0)
        
        if rank == mainrank:
            outFlat  = np.zeros(np.prod(arrShape),dtype)
        else:
            outFlat  = None
        
        myArrFlat = np.ravel(subArr)
        comm.Gatherv(sendbuf=[myArrFlat , mpidtype], recvbuf=[outFlat , split_sizes], root=mainrank)
        del(myArrFlat)
        
        comm.barrier()
        if rank == mainrank:
            out  = np.zeros(arrShape, dtype)
            slq  = domDecompND.slq
            elq  = domDecompND.elq
            for ranki in range(size):
                temp = outFlat[blocks[ranki]:blocks[ranki+1]].reshape(allmyarrShape[ranki*dim:(ranki+1) * dim])
                slices = tuple(slice(slq[i,ranki], elq[i,ranki], 1) for i in range(dim))
                out[slices] = temp
                del(temp)
        else:
            out = None
        
    return out
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈


def get_subarray_ND(rank,domDecompND,Arr,ignoreAxes=None) -> np.array:
    """
        Slice the N-dimensional array 'Arr' into an N-dimensional sub-array for
        proc rank according to the prescribed domain decomposition domDecompND.
        
        Required in: 
            scatter_array_ND
            reshape_array_ND
        
        Dependencies:
            Direct: None
            Indirect: domainDecomposeND
        
        Inputs: 
            Mandatory:
                rank, domDecompND
                ND array (Arr)
            
            Optional:
                1D list of axes to be ignored in the slicing (ignoreAxes). E.g.
                if domDecompND.parallel_axes = [0,1,2,3] and ignoreAxes = [0, 2]
                then get_subarray_ND will ignore axes 1 and 3 in parallel_axes
                when perfoming the slicing. While when ignoreAxes = None,
                the slicing takes place in perfect compliance with domDecompND.
        
        Output: ND sub-array belonging to the MPI-block handled by proc 'rank'.
    """
    
    arrShape = Arr.shape
    n_dim    = len(arrShape)
    if ignoreAxes == None:
        slices = tuple(slice(domDecompND.slq[i,rank], domDecompND.elq[i,rank], 1) for i in range(n_dim))
    else:
        if n_dim != domDecompND.n_dim: raise ValueError("The dimensions of the input array must be consistent with the input decomposition scheme!")
        ignoreAxes    = np.ravel([ignoreAxes])
        ignoreAxesLen = len(ignoreAxes)
        if n_dim < ignoreAxesLen:
            raise ValueError("The number of prescribed axes exceeds the dimensions of the input array!")
        slices = tuple(slice(0,arrShape[i],1) if i not in ignoreAxes else slice(domDecompND.slq[i,rank], domDecompND.elq[i,rank], 1) for i in range(n_dim))
    
    return Arr[slices]


#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈ MPI Scatter ◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
def scatter_array_ND(comm,rank,mainrank,domDecompND,mainArr,dtype='float')  -> np.array:
    """
        An alternative version of MPI-scatter using the mpi4py native send 
        and receive functions, to scatter chuncks of a main ND array stored
        on  the  mainrank  across  all  the  cores  according to the domain 
        decomposition scheme domDecompND
        
        Tip1: MPI-scattering is the opposite of MPI-gathering. While in MPI-
        gathering we gather locally stored numbers or arrays of numbers from
        each proc into a single main array stored on the main rank according
        to a specific domain decomposition prescribed by domDecompND, in MPI- 
        scattering we send chunks of the single main array stored on the main
        rank to each proc according to the domain decomposition scheme prescribed
        by domDecompND.
        
        Tip2: Given the above, since a scalar number cannot be broken into
        smaller parts, use the bcast function instead of scatter to broadcast
        any scalar number stored on the main rank to all the other ranks.
    
        Required in: None
        
        Dependencies: 
            Direct:   get_subarray_ND
            Indirect: domainDecomposeND
        
        Inputs: 
            comm,rank,mainrank,domDecompND
            ND array on the main rank to be scattered according to domDecompND (mainArr)
            datatype of the input ND array (dtype)
        
        Output: ND sub-array belonging to the MPI-block handled by proc 'rank'.
    """
    
    size = domDecompND.n_processors
    if dtype == 'float':
        dtype    = float
    elif dtype == 'int':
        dtype    = int
    
    split_sizes  = domDecompND.split_sizes
    
    if rank == mainrank:
        myOutArr = np.ravel(get_subarray_ND(mainrank, domDecompND, mainArr))
        for ranki in range(1,size):
            temp = np.ravel(get_subarray_ND(ranki, domDecompND, mainArr))
            comm.Send(temp, dest=ranki, tag=ranki)
    else:
        myOutArr = np.zeros(split_sizes[rank],dtype)
        comm.Recv(myOutArr, source=mainrank, tag=rank)
    
    return myOutArr.reshape(domDecompND.mynq[:,rank])
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈


from typing import overload, Union
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈ MPI Broadcast ◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
@overload
def bcast(Arr: int) -> int: ...

@overload
def bcast(Arr: float) -> float: ...

@overload
def bcast(Arr: np.array) -> np.array: ...

def bcast(comm, rank, mainrank, Arr: Union[int, float, np.array], dtype='float') -> Union[int, float, np.array]:
    """
        Broadcast any number or array of numbers stored
        on the mainrank to all the other procs within the
        MPI environment. 
        
        Tip: MPI-broadcasting is simply the task of copying
        the same number or array of numbers stored on the main
        rank to all the other procs.
        
        Required in: create_randoms_acorss_cores
        
        Dependencies: None
        
        Inputs: 
            comm,rank,mainrank
            ND array on the main rank to be broadcast (copied on) to all other procs
            datatype of the input ND array (dtype)
                
        Output: 
            Int or float scalars if the input is a scalar of the same datatype
            Array of integers of floats if the input is an array of the same datatype
    """
    
    if dtype == 'float':
        dtype    = float
        mpidtype = MPI.DOUBLE
    elif dtype == 'int':
        dtype    = int
        mpidtype = MPI.INT
    
    switch = 0
    if rank == mainrank:
        if np.array([Arr]).shape[0] == 1:
            switch = 1
    
    switch = comm.bcast(switch,root=mainrank)
    
    if switch == 1:
        outputArr = comm.bcast(Arr,root=mainrank)
    else:
        if rank == mainrank:
            arrShape = np.array(Arr.shape,int)
        else:
            arrShape = None
        
        arrShape = comm.bcast(arrShape,root=mainrank)
        if rank == mainrank:
            outputArr = np.ravel(Arr)
        else:
            outputArr = np.zeros(np.prod(arrShape),dtype)
        
        comm.Bcast([outputArr,mpidtype], root=mainrank)
        
        outputArr = outputArr.reshape(arrShape)
        
    return outputArr
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈


def reshape_array_ND(comm, rank, mainrank, old_DomDecompND, newDomDecompND, subArrArr,  dtype='float') -> np.array:
    """
        Reshape subarrays of original shapes described by the domain
        decomposition scheme old_DomDecompND into subarrays of new
        shapes described by an alternative domain decomposition scheme
        newDomDecompND.
        
        Required in: None
        
        Dependencies:
            Direct:   gather_array_ND
            Indirect: domainDecomposeND
        
        Inputs: 
            comm, rank, mainrank, old_DomDecompND, newDomDecompND
            ND subarrays to be reshaped 
        
        Output: reshaped ND subarrays
    """
    
    size = old_DomDecompND.n_processors
    # Gather everything on mainrank
    mainRankArr = gather_array_ND(comm, rank, mainrank, old_DomDecompND, subArrArr, dtype)
    
    if dtype == 'float':
        dtype    = float
    elif dtype == 'int':
        dtype    = int
    
    split_sizes  = newDomDecompND.split_sizes
    
    if rank == mainrank:
        myOutArr = np.ravel(get_subarray_ND(mainrank, newDomDecompND, mainRankArr))
        for ranki in range(1,size):
            temp = np.ravel(get_subarray_ND(ranki, newDomDecompND, mainRankArr))
            comm.Send(temp, dest=ranki, tag=ranki)
    else:
        myOutArr = np.zeros(split_sizes[rank],dtype)
        comm.Recv(myOutArr, source=mainrank, tag=rank)
    
    return myOutArr.reshape(newDomDecompND.mynq[:,rank])


def create_randoms_acorss_cores(comm, rank, mainrank, arrShape, lowHigh=[-13.54,13.3]):
    """
        Create the same array of shape arrShape of random real
        numbers on all cores.
        
        Required in: None
        
        Dependencies: bcast
        
        Inputs: 
            Mandatory:
                comm, rank, mainrank
                Target shape of the random array (arrShape)
            Optional:
                Set a min and max for the random numbers (lowHigh)
        
        Output: ND array of shape arrShape of random reals
    """
    
    if rank == mainrank:
        origArr = np.random.uniform(low=lowHigh[0], high=lowHigh[1], size=(arrShape))
        copyArr = origArr.copy()
    else:
        origArr = None
    
    origArr = bcast(comm, rank, mainrank, origArr)
    
    if rank == mainrank:
        if np.all(origArr == copyArr):
            print("")
            print("Array of randoms of the prescribed shape has been successfully broadcast to all ranks!")
            print("")
        else:
            print("Something went wrong with broadcasting the array of randoms!")
            print("np.max(np.abs(testArr-testArr0)) = ", np.max(np.abs(origArr-copyArr)))
    
    return origArr