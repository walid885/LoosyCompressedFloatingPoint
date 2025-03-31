#!/bin/bash

# Script to run the MPI compression analysis with varying core counts
# Usage: ./multicoreAna.sh

# Ensure the script runs from its directory
cd "$(dirname "$0")"

# Activate virtual environment (if applicable)
# source /path/to/your/venv/bin/activate

# Define the core counts to test
CORE_COUNTS=(1 2 4 6)

for cores in "${CORE_COUNTS[@]}"; do
    echo "=========================================="
    echo "Running analysis with $cores core(s)"
    echo "=========================================="
    
    # Create a directory for this core count
    output_dir="mpi_compression_analysis_${cores}cores"
    mkdir -p "$output_dir"
    
    # Run with mpirun and the specified number of processes
    # Pass the number of cores and output directory as arguments
    mpirun -np $cores python3 ./CompressingInsightsParalleld.py "$output_dir"
    
    echo "Analysis with $cores core(s) completed"
    echo ""
done

echo "All analyses completed. Check the mpi_compression_analysis_*cores directories for results."
