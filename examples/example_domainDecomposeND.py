import AutoParallelizePy
import numpy as np

size = 48
axes_limits = [47,37,1,98]  # aka how many data points are in each dimension
parallel_axes = [0,1,3]
user_nblocks = [3,4,1,4]
user_nblocks = None
domDecompND = AutoParallelizePy.domainDecomposeND(size,axes_limits,parallel_axes,user_nblocks,False,True,False)
a  = np.arange(np.prod(axes_limits)).reshape(axes_limits)
a1 = np.zeros(axes_limits)
for rank in range(size):
    slq0 = domDecompND.slq[0][rank]
    elq0 = domDecompND.elq[0][rank]
    slq1 = domDecompND.slq[1][rank]
    elq1 = domDecompND.elq[1][rank]
    slq2 = domDecompND.slq[2][rank]
    elq2 = domDecompND.elq[2][rank]
    slq3 = domDecompND.slq[3][rank]
    elq3 = domDecompND.elq[3][rank]
    a1[slq0:elq0,slq1:elq1,slq2:elq2,slq3:elq3] = a[slq0:elq0,slq1:elq1,slq2:elq2,slq3:elq3]


print(np.all(a == a1))