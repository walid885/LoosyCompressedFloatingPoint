#!/bin/bash

# Script to run the MPI compression analysis with varying core counts
# Usage: ./run_multicore_analysis.sh

# Define the core counts to test
CORE_COUNTS=(1 2 4 8)

for cores in "${CORE_COUNTS[@]}"; do
    echo "=========================================="
    echo "Running analysis with $cores core(s)"
    echo "=========================================="
    
    # Run with mpirun and the specified number of processes
    # Adjust path to your script as needed
    mpirun -n $cores python your_script.py $cores
    
    echo "Analysis with $cores core(s) completed"
    echo ""
done

echo "All analyses completed. Check the mpi_compression_analysis_*cores directories for results."
