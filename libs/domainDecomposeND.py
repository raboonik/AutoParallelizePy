"""
    Here are some core functions to be called in
    other parts of AutoParallelizePy.
    
    Dependencies:
        funcs
"""

from funcs import *


class domainDecomposeND:
    def __init__(self, n_processors, axes_limits, parallel_axes, user_nblocks=None, suggest_alternative=False, set_last_to1=False, decompose=True):
        """
            Domain decomposition class designed to optimally break 
            down an array of shape axes_limits into n_processors 
            blocks for easy implementation in parallelization schemes.
            
            Required in:
                AutoParallelizePy
            
            Dependencies:
                funcs
        
            Inputs:
                Mandatory:
                    n_processors:  Total number of processors available
                    axes_limits:   Shape of the N-dimensional array to
                                   be domain decomposed
                    parallel_axes: List of m <= N integers prescribing the axes 
                                   of the N-Dimensional array to be parallelized
                Optional:
                    user_nblocks:        User's prescribed domain decomposition consisting
                                         of N integers assigning the number of blocks in 
                                         each dimension/axis.
                    suggest_alternative: Suggest a more optimal domain decomposition if the
                                         user's input user_nblocks is sub-optimal.
                    set_last_to1:        If true and user_nblocks == None, prevent parallelization 
                                         of the last dimension/axis by overriding the prescribed
                                         user's parallel_axes and setting the last element of
                                         user_nblocks to 1.
                    decompose:           If False, override all the mandatory inputs and do not 
                                         parallelize.
            
            Object attributes:
                nblock:               (N)D list of integers prescribing the final block 
                                      partitioning.
                coordinates:          (N x n_processors)D list of all the block coordinates belonging 
                                      to each processor. 
                slq:                  (N x n_processors)D list of all the indices marking the start of 
                                      sub-intervals in each direction/axis. E.g. slq[m][rank] gives the 
                                      first index handled by processor 'rank' in the m-th axis.
                elq:                  (N x n_processors)D list of all the indices marking the end of 
                                      sub-intervals in each direction/axis. E.g. elq[m][rank] gives the 
                                      last index handled by processor 'rank' in the m-th axis.
                mynq:                 (N x n_processors)D list of the number of points covered in each
                n_dim:                Total number of dimensions of the data 
                n_par_dim:            Total number of axes parallelized
                                      direction by each processor.
                axes_limits:          The input axes_limits
                parallel_axes:        Final parallel_axes
                user_nblocks:         The input user_nblocks
                n_processors:         The input n_processors
                suggest_alternative:  The input suggest_alternative
                set_last_to1:         The input set_last_to1
                decompose:            The input decompose
        """
        n_dim    , axes_limits   = len(axes_limits)  ,np.array(axes_limits)
        n_par_dim, parallel_axes = len(parallel_axes),np.array(parallel_axes)
        
        slq     = np.zeros([n_dim,n_processors], int)
        elq     = np.zeros([n_dim,n_processors], int)
        blcq    = np.zeros([n_dim,n_processors], int)
        nblocks = np.ones(n_dim,int)
        if n_par_dim > 0 and decompose:
            if np.any(parallel_axes < 0): parallel_axes = [ax if ax>=0 else n_dim+ax for ax in parallel_axes]
            
            if np.max(parallel_axes) > n_dim - 1: raise ValueError("Error in the list of the axes to be parallelized! Input a valid list containing \
                                                                   the index of the axes to be parallelized!")
            
            if n_dim < n_par_dim                : raise ValueError("The number of axes to be parallelized cannot be larger than the total number of \
                                                                   axes!")
            
            parallel_axes = np.sort(parallel_axes)
            
            parallel_axes_limits = np.array([axes_limits[i] if i in parallel_axes else 1 for i in range(n_dim)],int)
            
            # Parallel scheme
            if user_nblocks == None:
                nblocks_par            = np.array(get_factors_list(n_processors, n_par_dim , set_last_to1))
                nblocks[parallel_axes] = nblocks_par
            else:
                # Test the input nblocks
                __ones             = count_ones(user_nblocks)
                temp_parallel_axes = np.array([i for i in range(n_dim) if i not in __ones[1]],int)
                
                if len(user_nblocks) != n_dim:                  raise ValueError("The input block list must have the same number of elements as axes \
                                                                                 limits! There are only {} provided while there are {} elements required!\
                                                                                 ".format(len(user_nblocks),n_dim))
                
                if n_processors != np.prod(user_nblocks):       raise ValueError("The prescribed block list does not match the number of procs!")
                
                if n_dim - n_par_dim != __ones[0]:              raise ValueError("There is a mismatch between the axes to be parallelized {} and the prescribed \
                                                                                  block list {}! The position of the `1's in the prescribed block list must not \
                                                                                  exist in list of the axes to be parallelized!".format(axes_limits, user_nblocks))
                
                if np.any(temp_parallel_axes != parallel_axes): raise ValueError("The prescribed block list and the list of axes to be parallelized do not match!")
                
                nblocks     = np.array(user_nblocks)
                nblocks_par = nblocks[temp_parallel_axes]
            
            if suggest_alternative: suggested_get_factors_list(nblocks_par, n_processors, n_par_dim , set_last_to1)
            
            if n_processors != np.prod(nblocks):
                raise ValueError("The number of blocks don't match the number of procs!")
            
            indices = get_nested_for_loops_indices(nblocks)
            
            for idim in range(n_dim):
                nblockq = nblocks[idim]            
                nlq     = np.array([ii for ii in range(0,int(nblockq)+1)])
                if nblockq == 1:
                    # No parallelization here
                    elq[idim,:] = axes_limits[idim]
                else:
                    myslq,myelq = get_slq_elq(parallel_axes_limits[idim],nblockq)
                    for rank in range(n_processors):
                        slq[idim,rank] = myslq[indices[idim][rank]]
                        elq[idim,rank] = myelq[indices[idim][rank]]
                    blcq[idim,rank] = nlq[indices[idim][rank]]  
        else:
            elq = np.array([[axes_limits[i] for rank in range(n_processors)] for i in range(n_dim)],dtype=int)
        
        # Object attributes
        self.nblock              = nblocks
        self.coordinates         = blcq
        self.slq                 = slq
        self.elq                 = elq
        self.mynq                = elq - slq
        self.axes_limits         = axes_limits
        self.parallel_axes       = parallel_axes
        self.n_dim               = n_dim
        self.n_par_dim           = n_par_dim
        self.user_nblocks        = user_nblocks
        self.n_processors        = n_processors
        self.suggest_alternative = suggest_alternative
        self.set_last_to1        = set_last_to1 
        self.decompose           = decompose    