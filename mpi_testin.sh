#!/bin/bash

# Script to run MPI analysis with varying core counts
# Usage: ./mpi_testin.sh

# Define the core counts to test
CORE_COUNTS=(1 2 4 6 12)

for cores in "${CORE_COUNTS[@]}"; do
    echo "=========================================="
    echo "Running analysis with $cores core(s)"
    echo "=========================================="
    
    if [ $cores -gt 6 ]; then
        # Use --use-hwthread-cpus for more than 6 cores
        mpirun --use-hwthread-cpus -np $cores python3 $(pwd)/mpite.py
    else
        mpirun -np $cores python3 $(pwd)/mpite.py
    fi
    
    echo "Analysis with $cores core(s) completed"
    echo ""
done

echo "All analyses completed."
