if rank == mainrank:
    out = gather_scalar(comm, size, rank, mainrank, scalarNum, dtype='int')