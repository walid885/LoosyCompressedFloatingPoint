# Lossy Floating Point Compression Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This project implements a comprehensive floating-point compression analysis framework that evaluates the impact of zeroing out least significant bits (LSBs) on different statistical distributions. The tool performs in-depth entropy analysis, measures information loss, and includes parallel processing capabilities to leverage multi-core systems.

## 📊 Features

- **Multi-distribution Analysis**: Tests compression effects on various statistical distributions (Uniform, Gaussian, Exponential, Skewed, Bimodal)
- **Entropy Measurement**: Calculates Shannon entropy and Kullback-Leibler divergence to quantify information loss
- **Bit-level Analysis**: Examines bit patterns in IEEE 754 floating-point representation
- **Parallel Processing**: Uses Python's `ProcessPoolExecutor` and MPI for scalable multi-core analysis
- **Comprehensive Visualization**: Generates detailed plots showing compression ratio, MSE, entropy changes, and distribution comparisons
- **Performance Analysis**: Evaluates parallel scaling efficiency with Amdahl's Law estimates

## 🔧 System Requirements

- **OS**: Linux (Ubuntu or similar distributions recommended)
- **CPU**: Multi-core processor (testing performed on AMD Ryzen 5 5600H)
- **Memory**: Sufficient RAM to handle parallel processes (8GB+ recommended)
- **Software**:
  - Python 3.x
  - Open MPI (for MPI-based execution)
  - Python packages: numpy, matplotlib, seaborn, pandas, scipy, zlib, json, mpi4py (optional)

## 🚀 Installation

### 1. Set up a Python environment

```bash
# Create and activate a virtual environment
python3 -m venv compression_env
source compression_env/bin/activate
```

### 2. Install required Python packages

```bash
pip install numpy matplotlib seaborn pandas scipy mpi4py
```

### 3. Install MPI (for MPI-based execution)

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev

# Verify installation
mpirun --version
```

## 💻 Usage

### Single-core Analysis

To run the basic analysis using the `EnhancedFloatingPointCompressor` class:

```bash
python3 EnhancedFloatingPointCompressor.py
```

This will:
1. Generate various statistical distributions
2. Analyze compression at different LSB zeroing levels
3. Create visualizations in the `enhanced_compression_analysis` directory
4. Output comprehensive reports on entropy and compression metrics

### Multi-core Analysis with ProcessPoolExecutor

The main script already includes multi-core analysis using Python's built-in `ProcessPoolExecutor`. By default, it will test with 1, 2, 4, and 8 cores, but you can modify this in the script:

```python
# Modify these values in the script
core_counts = [1, 2, 4, 8]  # Adjust based on your hardware
```

### Multi-core Analysis with MPI

For MPI-based execution, use the provided shell script:

```bash
# Make the script executable
chmod +x multicoreAna.sh

# Run the analysis
./multicoreAna.sh
```

This script will run the analysis with 1, 2, 4, and 6 cores by default. You can modify `CORE_COUNTS` in the script to test with different core counts.

## 📋 Output

The analysis generates the following outputs in the `enhanced_compression_analysis` directory:

- **PDF Visualizations**:
  - `comprehensive_compression_analysis.pdf`: Detailed compression metrics for each distribution
  - `*_distribution_comparison.pdf`: Visual comparison of original vs. compressed distributions
  - `bit_pattern_analysis.pdf`: Analysis of bit patterns for different distributions
  - `multicore_performance_analysis.pdf`: Parallel performance metrics

- **Text Reports**:
  - `comprehensive_analysis_report.txt`: Detailed metrics on compression effectiveness
  - `multicore_performance_report.txt`: Analysis of parallel scaling efficiency

- **JSON Data**:
  - `compression_analysis_results.json`: Raw data for all compression metrics
  - `multicore_performance_results.json`: Performance measurements across core counts

## 📊 Understanding the Results

### Compression Metrics

- **Entropy Reduction**: Indicates information loss due to compression
- **KL Divergence**: Measures how much the compressed distribution differs from the original
- **Compression Ratio**: Higher values indicate better compressibility
- **MSE (Mean Squared Error)**: Quantifies numerical accuracy loss

### Parallel Performance

- **Speedup**: Execution time improvement relative to single-core baseline
- **Efficiency**: Speedup divided by core count (ideal is 100%)
- **Theoretical Maximum**: Estimated using Amdahl's Law based on serial fraction

## 🔍 Project Structure

- `EnhancedFloatingPointCompressor.py`: Main analysis class
- `CompressingInsightsParalleld.py`: MPI-based parallel implementation
- `multicoreAna.sh`: Shell script for running MPI analysis with various core counts
- `enhanced_compression_analysis/`: Output directory for results and visualizations

## 🛠️ Troubleshooting

### MPI Issues

- **"Insufficient slots" error**: Use `--use-hwthread-cpus` or `--oversubscribe` flag with mpirun
- **Executable not found**: Ensure mpirun and Python are correctly installed and in your PATH
- **Performance degradation**: Too many processes can cause overhead; monitor system resources

### Python Issues

- **Missing modules**: Ensure all required packages are installed
- **Memory errors**: Reduce sample_size in the compressor initialization if RAM is limited

## 🔮 Future Improvements

- Implement adaptive LSB zeroing based on data characteristics
- Explore domain-specific compression strategies for scientific computing
- Add GPU-accelerated analysis for larger datasets
- Implement additional quality metrics for domain-specific applications
- Integrate with other compression algorithms for comparison


```