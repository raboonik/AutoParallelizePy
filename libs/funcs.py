import numpy as np


"""
    Here are some core functions to be used in
    domainDecomposeND.
    
    Dependencies:
        None
"""


def get_nested_for_loops_indices(blksArr) -> np.array:
    """
        Generates all the indices in a nested loop of
        depth N to be used in a single loop.
        
        Required in: 
            class domainDecomposeND
        
        Dependencies:
            None
        
        Input: An array of length N with each element 
               representing the range of an inner for loop.
        
        Output: An array of dimensions N x prod(blksArr)
                containing all the indices needed to
                reconstruct the nested for loop in a single
                loop. 
    """
    
    nestLvl  = len(blksArr)
    totalLen = np.prod(blksArr)
    indices  = np.zeros([nestLvl, totalLen], int)
    
    for i in range(nestLvl-1):
        for j in range(blksArr[i]):
            myLen                          = np.prod(blksArr[i+1:])
            indices[i,j*myLen:(j+1)*myLen] = j
    
    # At this stage the outermost loop's indices are fully taken care of
    
    for j in range(blksArr[-1]):
        indices[-1,j] = j
    
    for i in range(1,nestLvl):
        inx_i_f = np.prod(blksArr[i:])
        repeat  = np.prod(blksArr[:i])
        for j in range(1, repeat):
            indices[i, inx_i_f * j : inx_i_f * (j + 1)] = indices[i,0:inx_i_f]
    
    return indices


def is_divisibale(divisor) -> tuple[int, int, bool]:
    """
        Breaks down any given composite integer in terms
        of its two smaller multiples.
        
        Dependencies:
            None
        
        Required in: 
            def get_factors_list
        
        Input: Integer number (of,say, cores in a given
               parallelization scheme)
        
        Output: 
            [0, 0, false] if input is prime.
            [multiple1, multiple2, true] if input is composite.
    """
    
    divisCond = False
    for quotient in range(2, divisor):
        dividend, remainder = divmod(divisor, quotient)
        if remainder == 0:
            divisCond = True
            break
    
    if not divisCond:
        quotient = 0
        dividend = 0
    return quotient, dividend, divisCond


def get_factors_list(divisor, ltarget=None, set_last_to1=False) -> list:
    """
        Breaks down any given composite integer 'divisor'
        into its 'ltarget' number of largest multiples.
        Think of the divisor as the total number of cores
        in a parallelization scheme and how it can be 
        distributed over 'ltarget' number of dimensions.
        
        Required in: 
            class domainDecomposeND
        
        Dependencies:
            is_divisibale
        
        Input: Integer number (of,say, cores in a given
               parallelization scheme)
        
        Output: list of all the multiples that make up
                the input
    """
    
    if ltarget == 0: ltarget = None
    if ltarget == 1: return [divisor]
    
    if ltarget == None:
        out = []
        subn1, subn2, divisCond = is_divisibale(divisor)
        # Check if divisor is a prime number
        if not divisCond: return [divisor]
        else: out.append(subn1)
        while True:
            temp = subn2
            subn1, subn2, divisCond = is_divisibale(subn2)
            if not divisCond: 
                out.append(temp)
                break
            else: out.append(subn1)
    else:
        if divisor == 0: return [0 for i in range(ltarget)]
        res = get_factors_list(divisor, None)
        k, m = divmod(len(res), ltarget)
        out = [int(np.prod(res[i*k+min(i, m):(i+1)*k+min(i+1, m)])) for i in range(ltarget)]
    
    out.sort()
    out.reverse()
    
    if set_last_to1:
        if out[-1] != 1:
            if len(out) >= 2:
                tempi = out[0] * out[1]
                tempf = out[-1] * out[-2]
                if tempi <= tempf:
                    out[0] = tempi
                    out.remove(out[1])
                else:
                    out[-1] = tempf
                    out.remove(out[-2])
            out.append(1)
    return out


def count_ones(lst) -> tuple[int, list]:
    """
        Count the number of ones and their indices
        in a given list of numbers. 
        
        Required in: 
            def suggested_get_factors_list
            class domainDecomposeND
        
        Dependencies:
            None
        
        Input: list of numbers.
        
        Output: number of ones and list of indices
    """
    
    out = [i for i in range(len(lst)) if lst[i] == 1]
    return len(out), out


def get_slq_elq(num,n) -> tuple[np.array, np.array]:
    """
        Split the interval [0, num] into two maximally equidistant n-dimensional
        arrays of sub-intervals [slq0,slq1,...,slqn] and [elq0,elq1,...,elqn]
        with slq0 = 0 and elqn = n, such that the sub-interval [slqm, elqm]
        belongs to processor of rank m in the domain decomposition scheme.
        
        Required in: 
            class domainDecomposeND
        
        Dependencies:
            None
        
        Input: Integers num and n
        
        Output: Two arrays of all the indices marking the begining and end
                of n sub-intervals.
    """
    if num < n:
        out = np.append(np.zeros(n-num,int),np.arange(num+1))
    else:
        stride = round(num/n)
        out = np.array([i*stride for i in range(n+1)])
        while out[-1] > num:
            out = out - 1
        
        if out[0]<0:
            out[0:-out[0]+1] = np.array([j*(stride-1) for j in range(-out[0]+1)])
        
        if out[-1]<num:
            deficit = num-out[-1]
            for i in range(deficit):
                out[-1-i] = out[-1-i] + deficit - i
    
    slq = np.array(out[0:-1])
    elq = np.array(out[1:])
    if out[-1]<num: raise ValueError("Something went wrong in get_slq_elq!")
    return slq,elq


def suggested_get_factors_list(factorsLst, n, ltarget=None, set_last_to1=False) -> None:
    """
        Print a more optimal parallelization scheme
        based on the number of available cores.
        
        Required in:
            class domainDecomposeND
        
        Dependencies:
            get_factors_list
            count_ones
        
        Input: list of numbers.
        
        Output: number of ones and list of indices
    """
    
    if n == 0: return
    if ltarget != None:
        ones,_   = count_ones(factorsLst)
        if ones == 0:
            # Everything is already optimal
            return
        
        if ones > 0 and set_last_to1: ones -= 1
        
        if ones == 0:
            # Everything is already optimal
            return
        
        if set_last_to1: optimal_ones = 1
        else:            optimal_ones = 0
        if ones > 0:
            i = 0
            np      = n
            np1     = n
            onesp   = ones
            switchp = True
            switchm = True
            nm      = n
            nm1     = n
            onesm   = ones
            while i < 50 and (onesp > optimal_ones or onesm > optimal_ones):
                # Only increase np if it's not yet reached an optimal value!
                if onesp > optimal_ones:
                    np       += 1
                    tempp     = get_factors_list(np, ltarget, set_last_to1)
                    onesp,_   = count_ones(tempp)
                
                 # Only decrease nm if it's still bigger than the least number for n which is 1    
                if nm > 1:
                    nm       -= 1
                    tempm     = get_factors_list(nm, ltarget, set_last_to1)
                    onesm,_   = count_ones(tempm)
                else:
                    onesm = optimal_ones
                
                # Save the penultimate optimal state
                if onesp < optimal_ones + 1 and switchp:
                    np1 = np
                    switchp = False
                if onesm < optimal_ones + 1 and switchm:
                    nm1 = nm
                    switchm = False
                
                i        += 1
        
        # Check if any improvement was made
        if onesp < ones or onesm < ones:
            if nm > 1:
                if   onesp < onesm:
                    if np1 < np:
                        print("The parallelization scheme can be slightly improved by increasing the number of cores from {} to {}".format(n,np1))
                        print("However, it would be most efficient if increased to {}".format(np))
                        print(factorsLst,"-->",get_factors_list(np1,ltarget, set_last_to1)," or, ",get_factors_list(np,ltarget, set_last_to1))
                    else:
                        print("The parallelization scheme can be most efficiently improved by increasing the number of cores from {} to {}".format(n,np))
                        print(factorsLst,"-->",get_factors_list(np,ltarget, set_last_to1))
                else: # onesp >= onesm
                    if nm1 < nm:
                        print("The parallelization scheme can be slightly improved by increasing the number of cores from {} to {}".format(n,nm1))
                        print("However, it would be most efficient if decreased to {}".format(nm))
                        print(factorsLst,"-->",get_factors_list(nm1,ltarget, set_last_to1)," or, ",get_factors_list(nm,ltarget, set_last_to1))
                    else:
                        print("The parallelization scheme can be most efficiently improved by decreasing the number of cores from {} to {}".format(n,nm))
                        print(factorsLst,"-->",get_factors_list(nm,ltarget, set_last_to1))
            else:
                if np1 < np:
                    print("The parallelization scheme can be slightly improved by increasing the number of cores from {} to {}".format(n,np1))
                    print("However, it would be most efficient if increased to {}".format(np))
                    print(factorsLst,"-->",get_factors_list(np1,ltarget, set_last_to1)," or, ",get_factors_list(np,ltarget, set_last_to1))
                else:
                    print("The parallelization scheme can be most efficiently improved by increasing the number of cores from {} to {}".format(n,np))
                    print(factorsLst,"-->",get_factors_list(np1,ltarget, set_last_to1))
            
            return
        else:
            # Nothing was improved
            return            
    else:
        # No suggestions
        return