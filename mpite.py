#!/usr/bin/env python

from mpi4py import MPI

def main():
    # Get MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Print a message from each process
    print(f"Hello from rank {rank} out of {size} processes!")

if __name__ == "__main__":
    main()
