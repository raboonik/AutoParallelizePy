"""
    Here are the core functions of AutoParallelizePy.
    
    Dependencies:
        domainDecomposeND
        mpi4py
    
    Variable naming: 
        comm:     MPI-communicator
        rank:     A processor in the parallelized scheme
        size:     Total number of processors available
        Arr :     An N-dimensional array
        mainrank: The number associated with a single processor
                  in charge of gathering and distributing sub-data
                  from other procs
"""

from mpi4py import *
from funcs import *
from domainDecomposeND import *



# In the following, axis is useful when you need a subset of the input arr, call it subArr
# where each rank adheres to the domain decomposition scheme of domDecompND only along the
# axes prescribed by axis, while taking the entirety of the remaining axes!
def get_subarray(rank,domDecompND,Arr,axis=None) -> np.array:
    """
        Slice the N-dimensional array 'Arr' into an N-dimensional sub-array for
        proc rank according to the prescribed domain decomposition domDecompND.
        
        Required in: 
            scatter_array_ND
            reshape_array_ND
        
        Dependencies:
            Direct:
                None
            Indirect:
                domainDecomposeND
        
        Inputs: 
            Mandatory:
                Proc number (rank)
                Domain decomposition scheme domDecompND (output of domainDecomposeND)
                N-D array (Arr)
            
            Optional:
                List of axes 
        
        Output: N-D sub-array belonging to the block handled by 'rank'
    """
    
    arrShape = Arr.shape
    n_dim    = len(arrShape)
    if axis == None:
        slices = tuple(slice(domDecompND.slq[i,rank], domDecompND.elq[i,rank], 1) for i in range(n_dim))
    else:
        if n_dim != domDecompND.n_dim: raise ValueError("The dimensions of the input array must be consistent with the input decomposition scheme!")
        axis    = np.ravel([axis])
        axisLen = len(axis)
        if n_dim < axisLen:
            raise ValueError("The number of prescribed axes exceeds the dimensions of the input array!")
        slices = tuple(slice(0,arrShape[i],1) if i not in axis else slice(domDecompND.slq[i,rank], domDecompND.elq[i,rank], 1) for i in range(n_dim))
    
    return Arr[slices]


def gather_scalar(comm, size, rank, mainrank, scalarNum, dtype='int') -> np.array:
    """
        Gather scalars from 
        
        Required in: 
            
        
        Dependencies:
            
        
        Inputs: 
            Mandatory:
                
            
            Optional:
                
        
        Output: 
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
    out = comm.bcast(out,root=mainrank)
    
    return out


# Axis is the position of the parallelized dimension, counting from 0
def gather_array_1D(comm, rank, mainrank, domDecompND, myinput, dtype='float'):
    """
        
        
        Required in: 
            
        
        Dependencies:
            
        
        Inputs: 
            Mandatory:
                
            
            Optional:
                
        
        Output: 
    """
    
    axes_limits   = domDecompND.axes_limits
    parallel_axes = domDecompND.parallel_axes
    
    if len(parallel_axes) > 1: raise ValueError("The number of axes to be parallelized exceed the 1D scope of gather_array_1D!")
    else: parallel_axis = parallel_axes[0]
    
    if dtype == 'float':
        dtype    = float
        mpidtype = MPI.DOUBLE
    elif dtype == 'int':
        dtype    = int
        mpidtype = MPI.INT
    
    dim = len(axes_limits)
    permutationCode_i   = alphabet[:dim]
    if dim > 1:
        # If dim > 1, then we need to make sure that it its the first axis being parallelized, so the data needs permutation of axes
        inxs                = range(dim)
        nonParallelizedInxs = [i for i in inxs if i != parallel_axis]
        permutationCodeInxs = list(np.append(parallel_axis,nonParallelizedInxs))
        
        permutationCode_f   = "".join([permutationCode_i[i] for i in permutationCodeInxs])
        rolled_axes_limits  = [axes_limits[i] for i in permutationCodeInxs]
        
        myinput = np.einsum(permutationCode_i+'->'+permutationCode_f, myinput)
    else:
        permutationCode_f = permutationCode_i
        rolled_axes_limits   = axes_limits
    
    split_sizes = domDecompND.split_sizes
    
    if rank == mainrank:
        tempFlat = np.zeros([np.prod(axes_limits)],dtype)
    else:
        tempFlat = None
    
    myinputFlat = np.ravel(myinput)
    comm.Gatherv(sendbuf=[myinputFlat , mpidtype], recvbuf=[tempFlat , split_sizes], root=mainrank)
    
    if rank == mainrank:
        out = tempFlat.reshape(rolled_axes_limits)
        out = np.einsum(permutationCode_f+'->'+permutationCode_i, out)
    else:
        out = None
    
    return out


# In the following, parallel_axes is the number of parallelized axes, and so the N in domDecompND must equal len(parallel_axes)
# The role of axis in the following is to gather the input array myinput where ndim(myinput) <= domDecompND.n_dim, but its 
# domain decomposition is the same as those axes in domDecompND picked prescribed by axis. Therefore, we require that 
# ndim(myinput) = len(axis), and where len(axis) = domDecompND.n_dim, we basically ignore axis.
def gather_array_ND(comm, rank, mainrank, domDecompND, myinput, dtype='float'):
    """
        
        
        Required in: 
            
        
        Dependencies:
            
        
        Inputs: 
            Mandatory:
                
            
            Optional:
                
        
        Output: 
    """
    
    size = domDecompND.n_processors
    axes_limits   = domDecompND.axes_limits
    parallel_axes = domDecompND.parallel_axes
    
    if len(parallel_axes) == 0: return myinput
    
    # gather_array_1D is slightly better than how gather_array_ND in 1D, so
    if len(parallel_axes) == 1:
        return gather_array_1D(comm, rank, mainrank, domDecompND, myinput, dtype=dtype)
    
    # Firstly, sort parallel_axes0. To understand this, consider the following case: axes_limits = [nq1, npar1, nq2, npar2, npar3] and hence
    # parallel_axes = [1,3,4]. However, if, for whatever reason, domDecompND was given [npar1, npar3, npar2] (instead of [npar1, npar2, npar3]),
    # which has parallel_axes = [1,4,3] associated with it, then the associated slq and elq would have the following order [slq[0],slq[2],slq[1]].
    # However, to avoid having to consider all the possible permutations here, we assume that [slq[0],slq[1],slq[2]] <--> [npar1, npar2, npar3]
    # which is ensured if parallel_axes is sorted!
    if np.any(np.array(parallel_axes) != np.sort(parallel_axes)): 
        parallel_axes.sort()
    
    dim = len(axes_limits)
    
    if dtype == 'float':
        dtype    = float
        mpidtype = MPI.DOUBLE
    elif dtype == 'int':
        dtype    = int
        mpidtype = MPI.INT
    
    myaxes_limits = np.array(list(myinput.shape),dtype=int)
    
    if rank == mainrank:
        allmyaxes_limits = np.zeros(size * dim,int)
    else:
        allmyaxes_limits = None
    
    split_sizes = domDecompND.split_sizes
    
    comm.barrier()
    
    split_sizes_mylLst = dim*np.ones(size,int)
    
    comm.Gatherv(sendbuf=[myaxes_limits, MPI.INT], recvbuf=[allmyaxes_limits, split_sizes_mylLst], root=mainrank)
    
    if rank == mainrank: 
        blocks       = np.insert(np.cumsum(split_sizes),0,0)
    
    if rank == mainrank:
        outFlat  = np.zeros(np.prod(axes_limits),dtype)
    else:
        outFlat  = None
    
    myArrFlat = np.ravel(myinput)
    comm.Gatherv(sendbuf=[myArrFlat , mpidtype], recvbuf=[outFlat , split_sizes], root=mainrank)
    del(myArrFlat)
    
    comm.barrier()
    if rank == mainrank:
        out  = np.zeros(axes_limits, dtype)
        slq  = domDecompND.slq
        elq  = domDecompND.elq
        for ranki in range(size):
            temp = outFlat[blocks[ranki]:blocks[ranki+1]].reshape(allmyaxes_limits[ranki*dim:(ranki+1) * dim])
            slices = tuple(slice(slq[i,ranki], elq[i,ranki], 1) for i in range(dim))
            out[slices] = temp
            del(temp)
    else:
        out = None
    return out


# The following is good for when we have an array on mainrank and we wish to scatter
# chunks of it to various procs as instructed by domDecompND. However, we cannot simply
# use Scatterv since we're dealing with a complicated decomposition of multi-dimensional
# arrays, so instead we have to use Send and Recv
# if rank == mainrank:
#     mainArr = ...
# else:
#     mainArr = None

def scatter_array_ND(comm,rank,mainrank,domDecompND,mainArr,dtype='float'):
    """
        
        
        Required in: 
            
        
        Dependencies:
            
        
        Inputs: 
            Mandatory:
                
            
            Optional:
                
        
        Output: 
    """
    
    size = domDecompND.n_processors
    if dtype == 'float':
        dtype    = float
        mpidtype = MPI.DOUBLE
    elif dtype == 'int':
        dtype    = int
        mpidtype = MPI.INT
    
    split_sizes  = domDecompND.split_sizes
    
    if rank == mainrank:
        myOutArr = np.ravel(get_subarray(mainrank, domDecompND, mainArr))
        for ranki in range(1,size):
            temp = np.ravel(get_subarray(ranki, domDecompND, mainArr))
            comm.Send(temp, dest=ranki, tag=ranki)
    else:
        myOutArr = np.zeros(split_sizes[rank],dtype)
        comm.Recv(myOutArr, source=mainrank, tag=rank)
    
    return myOutArr.reshape(domDecompND.mynq[:,rank])


def reshape_array_ND(comm, rank, mainrank, old_DomDecompND, newDomDecompND, myInputArr,  dtype='float'):
    """
        
        
        Required in: 
            
        
        Dependencies:
            Direct:   gather_array_ND
            Indirect: domainDecomposeND
            
        
        Inputs: 
            
                
        
        Output: 
    """
    
    size = old_DomDecompND.n_processors
    # Gather everything on mainrank
    mainRankArr = gather_array_ND(comm, rank, mainrank, old_DomDecompND, myInputArr, dtype)
    
    if dtype == 'float':
        dtype    = float
        mpidtype = MPI.DOUBLE
    elif dtype == 'int':
        dtype    = int
        mpidtype = MPI.INT
    
    split_sizes  = domDecompND.split_sizes
    
    if rank == mainrank:
        myOutArr = np.ravel(get_subarray(mainrank, newDomDecompND, mainRankArr))
        for ranki in range(1,size):
            temp = np.ravel(get_subarray(ranki, newDomDecompND, mainRankArr))
            comm.Send(temp, dest=ranki, tag=ranki)
    else:
        myOutArr = np.zeros(split_sizes[rank],dtype)
        comm.Recv(myOutArr, source=mainrank, tag=rank)
    
    return myOutArr.reshape(newDomDecompND.mynq[:,rank])


def bcast_array_ND(comm, rank, mainrank, inputArr, dtype='float'):
    """
        
        
        Required in: 
            
        
        Dependencies:
            
        
        Inputs: 
            Mandatory:
                
            
            Optional:
                
        
        Output: 
    """
    
    if dtype == 'float':
        dtype    = float
        mpidtype = MPI.DOUBLE
    elif dtype == 'int':
        dtype    = int
        mpidtype = MPI.INT
    
    if rank == mainrank:
        axes_limits = np.array(inputArr.shape,int)
    else:
        axes_limits = None
    
    axes_limits = comm.bcast(axes_limits,root=mainrank)
    if rank == mainrank:
        outputArr = np.ravel(inputArr)
    else:
        outputArr = np.zeros(np.prod(axes_limits),dtype)
    
    # To copy the same exact array from one rank to all the other simply form the main array on mainrank
    # and form zero arrays of the same size on all the other ranks and then use Bcast. Note that Bcast
    # broadcasts arrays while bcast baradcasts scalars! 
    # Also mind that Scatterv is meant for scattering size-inequal chunks (sub-arrays) of a large array 
    # existing only in one rank to all the other ranks! So for that we need split_sizes and blocks! 
    comm.Bcast([outputArr,mpidtype], root=mainrank)
    
    outputArr = outputArr.reshape(axes_limits)
    
    return outputArr


def create_update_bcast_local_subarray(comm,rank,mainrank,domDecompND,axis,newComm=None,subArr=None,inx=[],value=0,mode='create',dtype='float'):    
    """
        
        
        Required in: 
            
        
        Dependencies:
            
        
        Inputs: 
            Mandatory:
                
            
            Optional:
                
        
        Output: 
    """
    
    if mode   == 'create':
        if dtype == 'float':
            dtype0    = float
        elif dtype == 'int':
            dtype0    = int
        # create the new subarray and save all the deets in newDomDecomp and return for later use in 'update'
        # or 'bcast' modes
        newSize          = np.prod(domDecompND.nblock[axis])
        rankCond         = rank < newSize
        if rankCond:
            lenAxis          = len(axis)
            newAxes_limits   = domDecompND.axes_limits[axis]
            newParallel_axes = [ax-(domDecompND.n_dim-lenAxis) for ax in axis if ax in domDecompND.parallel_axes]
            newDomDecomp     = domainDecomposeND(newSize, newAxes_limits, newParallel_axes)
            subArr           = np.zeros(newDomDecomp.mynq[:,rank],dtype0)
            newDomDecomp.__dict__["lenAxis"]  = lenAxis
        else:
            def newDomDecomp(): pass
            subArr           = None
        
        size = domDecompND.n_processors
        
        newDomDecomp.__dict__['rankCond'] = rankCond
        newDomDecomp.__dict__['size']     = size
        newDomDecomp.__dict__['newSize']  = newSize
        
        # We need to create another mpi communicator since here we work with a different size
        # Produces a group by reordering an existing group and taking only unlisted members
        newGroup = comm.group.Excl(range(newSize,size))
        # print(newGroup.size)
        newComm = comm.Create_group(newGroup)
        # print(newComm, newComm.Get_size())
        
        return newComm, subArr, newDomDecomp
    elif mode == 'update':
        # Assume we have newDomDecomp
        if domDecompND.__dict__['rankCond']:
            subArr[tuple(inx)] = value
        else:
            pass
        return subArr
    elif mode == 'bcast':
        if domDecompND.__dict__['rankCond']:
            arr = gather_array_ND(newComm, rank, mainrank, domDecompND, subArr, dtype)
        else:
            arr = None
        arr = bcast_array_ND(comm, rank, mainrank, arr, dtype=dtype)
        return arr


def create_randoms_acorss_cores(comm, rank, mainrank, axes_limits, lowHigh=[-13.54,13.3]):
    """
        
        
        Required in: 
            
        
        Dependencies:
            
        
        Inputs: 
            Mandatory:
                
            
            Optional:
                
        
        Output: 
    """
    
    if rank == mainrank:
        origArr = np.random.uniform(low=lowHigh[0], high=lowHigh[1], size=(axes_limits))
        copyArr = origArr.copy()
    else:
        origArr = None
    
    origArr = bcast_array_ND(comm, rank, mainrank, origArr)
    
    if rank == mainrank:
        if np.all(origArr == copyArr):
            print("")
            print("Array of randoms of the prescribed shape has been successfully broadcast to all ranks!")
            print("")
        else:
            print("Something went wrong with broadcasting the array of randoms!")
            print("np.max(np.abs(testArr-testArr0)) = ", np.max(np.abs(origArr-copyArr)))
    
    return origArr