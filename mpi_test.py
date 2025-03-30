from mpi4py import MPI
print(f'Success! Rank {MPI.COMM_WORLD.Get_rank()}/{MPI.COMM_WORLD.Get_size()}')
