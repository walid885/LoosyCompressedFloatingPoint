from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import seaborn as sns
import pandas as pd
from scipy import stats
import zlib
import json

class EnhancedFloatingPointCompressor:
    def __init__(self, seed=42, sample_size=1_000_000):
        """
        Initialize the compressor with specific parameters
        
        :param seed: Random seed for reproducibility
        :param sample_size: Number of floating-point numbers to generate
        """
        np.random.seed(seed)
        self.sample_size = sample_size
        
        # Create output directory
        self.output_dir = 'enhanced_compression_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # LSB (Least Significant Bits) to zero out
        self.lsb_levels = [8, 10, 12, 14, 16, 20, 24, 32]
        
        # Color palette for consistent visualization
        self.color_palette = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
        ]

    def split_data(self,data, num_chunks):
        """
        Split data into chunks for parallel processing
        
        :param data: Input array to split
        :param num_chunks: Number of chunks to create
        :return: List of data chunks
        """
        chunk_size = len(data) // num_chunks
        return [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]


    def generate_distributions(self):
        """
        Generate floating-point numbers from different distributions
        
        :return: Dictionary of distributions
        """
        # Ensure sample size is even for bimodal distribution
        half_size = self.sample_size // 2
        return {
            'Uniform': np.random.uniform(0, 100, self.sample_size),
            'Gaussian': np.random.normal(50, 15, self.sample_size),
            'Exponential': np.random.exponential(10, self.sample_size),
            'Skewed': np.random.lognormal(0, 1, self.sample_size),
            'Bimodal': np.concatenate([
                np.random.normal(20, 5, half_size), 
                np.random.normal(80, 5, half_size)
            ])
        }

    def calculate_entropy(self, data, bins=100):
        """
        Calculate Shannon entropy using histogram binning
        
        :param data: Input data array
        :param bins: Number of bins for discretization
        :return: Entropy value
        """
        # Compute histogram with density normalization
        hist, _ = np.histogram(data, bins=bins, density=True)
        
        # Get bin width for proper entropy calculation
        bin_width = (data.max() - data.min()) / bins
        
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        
        # Calculate entropy with correct scaling for continuous values
        return -np.sum(hist * np.log2(hist)) * bin_width

    def calculate_kl_divergence(self, original_data, compressed_data, bins=100):
        """
        Calculate Kullback-Leibler Divergence between original and compressed data
        
        :param original_data: Original data array
        :param compressed_data: Compressed data array
        :param bins: Number of bins for discretization
        :return: KL Divergence value
        """
        # Calculate shared bin edges to ensure alignment
        min_val = min(original_data.min(), compressed_data.min())
        max_val = max(original_data.max(), compressed_data.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Compute histograms with shared bins
        orig_hist, _ = np.histogram(original_data, bins=bin_edges, density=True)
        comp_hist, _ = np.histogram(compressed_data, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        orig_hist = orig_hist + epsilon
        comp_hist = comp_hist + epsilon
        
        # Normalize to ensure proper probability distributions
        orig_hist = orig_hist / np.sum(orig_hist)
        comp_hist = comp_hist / np.sum(comp_hist)
        
        # Calculate KL Divergence
        return np.sum(orig_hist * np.log2(orig_hist / comp_hist))

    def compress_data_parallel(self, data, num_bits, core_id=None):
        """
        Compress data by zeroing out least significant bits with core tracking
        
        :param data: Input array of floating-point numbers
        :param num_bits: Number of least significant bits to zero out
        :param core_id: ID of the core processing this data
        :return: Compressed data and analysis metrics with execution time
        """
        start_time = time.time()
        
        # Create copy to avoid modifying original data
        data_copy = data.copy()
        
        # Convert to byte representation
        byte_data = data_copy.tobytes()
        original_bytes = len(byte_data)
        
        # Get view of data as unsigned 64-bit integers (IEEE 754 representation)
        int_view = data_copy.view(np.uint64)
        
        # Create a mask to zero out least significant bits
        mask = np.uint64(~((1 << num_bits) - 1))
        
        # Apply the mask to modify the data in place
        int_view[:] = int_view & mask
        
        # Now data_copy contains the compressed values
        compressed_data = data_copy
        
        # Additional analysis
        compressed_bytes = zlib.compress(compressed_data.tobytes())
        
        # Calculate direct bit-level entropy before and after compression
        original_bytes_array = np.frombuffer(data.tobytes(), dtype=np.uint8)
        compressed_bytes_array = np.frombuffer(compressed_data.tobytes(), dtype=np.uint8)
        
        # Bit-level entropy calculation
        orig_bit_counts = np.zeros(8)
        comp_bit_counts = np.zeros(8)
        
        for i in range(8):
            orig_bit_counts[i] = np.sum((original_bytes_array & (1 << i)) > 0)
            comp_bit_counts[i] = np.sum((compressed_bytes_array & (1 << i)) > 0)
        
        orig_bit_entropy = -np.sum((orig_bit_counts/len(original_bytes_array)) * 
                                np.log2(orig_bit_counts/len(original_bytes_array) + 1e-10))
        comp_bit_entropy = -np.sum((comp_bit_counts/len(compressed_bytes_array)) * 
                                np.log2(comp_bit_counts/len(compressed_bytes_array) + 1e-10))
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            'original_data': data,
            'compressed_data': compressed_data,
            'original_entropy_fine': self.calculate_entropy(data, bins=200),
            'original_entropy_coarse': self.calculate_entropy(data, bins=50),
            'compressed_entropy_fine': self.calculate_entropy(compressed_data, bins=200),
            'compressed_entropy_coarse': self.calculate_entropy(compressed_data, bins=50),
            'kl_divergence_fine': self.calculate_kl_divergence(data, compressed_data, bins=200),
            'kl_divergence_coarse': self.calculate_kl_divergence(data, compressed_data, bins=50),
            'original_size': original_bytes,
            'compressed_size': len(compressed_bytes),
            'compression_ratio': original_bytes / len(compressed_bytes),
            'mse': np.mean((data - compressed_data)**2),
            'max_abs_error': np.max(np.abs(data - compressed_data)),
            'orig_bit_entropy': orig_bit_entropy,
            'comp_bit_entropy': comp_bit_entropy,
            'bit_entropy_diff': orig_bit_entropy - comp_bit_entropy,
            'execution_time': execution_time,
            'core_id': core_id
        }


    
    def compress_chunk(self, args):
        """
        Process a chunk of data (for multiprocessing)
        
        :param args: Tuple containing (data_chunk, num_bits, core_id)
        :return: Compression results for this chunk
        """
        data_chunk, num_bits, core_id = args
        return self.compress_data_parallel(data_chunk, num_bits, core_id)



    def _generate_performance_report(self, performance_results):
        """
        Generate visualizations and analysis of multicore performance
        
        :param performance_results: List of performance measurement dictionaries
        """
        # Convert to DataFrame
        perf_df = pd.DataFrame(performance_results)
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multicore Performance Analysis', fontsize=16)
        
        # 1. Execution Time vs Core Count
        for dist_name in perf_df['Distribution'].unique():
            dist_data = perf_df[perf_df['Distribution'] == dist_name]
            for lsb in dist_data['LSB_Zeroed'].unique():
                lsb_data = dist_data[dist_data['LSB_Zeroed'] == lsb]
                axs[0, 0].plot(lsb_data['Num_Cores'], lsb_data['Execution_Time'], 
                            marker='o', label=f'{dist_name}, LSB={lsb}')
        
        axs[0, 0].set_title('Execution Time vs. Core Count')
        axs[0, 0].set_xlabel('Number of Cores')
        axs[0, 0].set_ylabel('Execution Time (s)')
        axs[0, 0].legend(fontsize='small')
        
        # 2. Speedup vs Core Count
        for dist_name in perf_df['Distribution'].unique():
            dist_data = perf_df[perf_df['Distribution'] == dist_name]
            for lsb in dist_data['LSB_Zeroed'].unique():
                lsb_data = dist_data[dist_data['LSB_Zeroed'] == lsb]
                axs[0, 1].plot(lsb_data['Num_Cores'], lsb_data['Speedup'], 
                            marker='o', label=f'{dist_name}, LSB={lsb}')
        
        # Add ideal speedup line
        max_cores = perf_df['Num_Cores'].max()
        axs[0, 1].plot([1, max_cores], [1, max_cores], 'k--', label='Ideal Speedup')
        
        axs[0, 1].set_title('Speedup vs. Core Count')
        axs[0, 1].set_xlabel('Number of Cores')
        axs[0, 1].set_ylabel('Speedup')
        axs[0, 1].legend(fontsize='small')
        
        # 3. Efficiency vs Core Count
        for dist_name in perf_df['Distribution'].unique():
            dist_data = perf_df[perf_df['Distribution'] == dist_name]
            for lsb in dist_data['LSB_Zeroed'].unique():
                lsb_data = dist_data[dist_data['LSB_Zeroed'] == lsb]
                axs[1, 0].plot(lsb_data['Num_Cores'], lsb_data['Efficiency'], 
                            marker='o', label=f'{dist_name}, LSB={lsb}')
        
        axs[1, 0].set_title('Efficiency vs. Core Count')
        axs[1, 0].set_xlabel('Number of Cores')
        axs[1, 0].set_ylabel('Efficiency (Speedup/Cores)')
        axs[1, 0].legend(fontsize='small')
        
        # 4. Actual vs Theoretical Speedup
        # Average across distributions and LSB values for clarity
        core_data = perf_df.groupby('Num_Cores').agg({
            'Speedup': 'mean',
            'Theoretical_Max_Speedup': 'mean'
        }).reset_index()
        
        axs[1, 1].plot(core_data['Num_Cores'], core_data['Speedup'], 
                    marker='o', label='Actual Speedup')
        axs[1, 1].plot(core_data['Num_Cores'], core_data['Theoretical_Max_Speedup'], 
                    'r--', label='Theoretical (Amdahl\'s Law)')
        
        axs[1, 1].set_title('Actual vs. Theoretical Speedup')
        axs[1, 1].set_xlabel('Number of Cores')
        axs[1, 1].set_ylabel('Speedup')
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'multicore_performance_analysis.pdf'), dpi=300)
        plt.close()
        
        # Generate textual performance report
        self._write_performance_report(perf_df)



    def analyze_compression(self):
        """
        Perform comprehensive compression and entropy analysis
        """
        # Generate distributions
        distributions = self.generate_distributions()
        
        # Prepare results storage
        all_results = []
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(len(distributions), 5, figsize=(28, 4*len(distributions)))
        fig.suptitle('Comprehensive Compression and Entropy Analysis', fontsize=16)
        
        # Iterate through distributions
        for row, (dist_name, original_data) in enumerate(distributions.items()):
            # Prepare storage for results
            compression_results = []
            
            # Analyze compression for different LSB levels
            for lsb in self.lsb_levels:
                # Compress data
                result = self.compress_data(original_data, lsb)
                
                # Store detailed results
                compression_result = {
                    'Distribution': dist_name,
                    'LSB_Zeroed': lsb,
                    'Original_Entropy_Fine': result['original_entropy_fine'],
                    'Compressed_Entropy_Fine': result['compressed_entropy_fine'],
                    'Original_Entropy_Coarse': result['original_entropy_coarse'],
                    'Compressed_Entropy_Coarse': result['compressed_entropy_coarse'],
                    'KL_Divergence_Fine': result['kl_divergence_fine'],
                    'KL_Divergence_Coarse': result['kl_divergence_coarse'],
                    'Compression_Ratio': result['compression_ratio'],
                    'MSE': result['mse'],
                    'Max_Absolute_Error': result['max_abs_error'],
                    'Original_Bit_Entropy': result['orig_bit_entropy'],
                    'Compressed_Bit_Entropy': result['comp_bit_entropy'],
                    'Bit_Entropy_Difference': result['bit_entropy_diff']
                }
                
                compression_results.append(compression_result)
                all_results.append(compression_result)
            
            # Convert to DataFrame
            df = pd.DataFrame(compression_results)
            
            # Plotting
            # Fine Entropy
            axs[row, 0].plot(df['LSB_Zeroed'], df['Original_Entropy_Fine'], label='Original', marker='o')
            axs[row, 0].plot(df['LSB_Zeroed'], df['Compressed_Entropy_Fine'], label='Compressed', marker='x')
            axs[row, 0].set_title(f'{dist_name}: Entropy (Fine Bins)')
            axs[row, 0].set_xlabel('LSB Zeroed')
            axs[row, 0].set_ylabel('Entropy')
            axs[row, 0].legend()

            # Coarse Entropy
            axs[row, 1].plot(df['LSB_Zeroed'], df['Original_Entropy_Coarse'], label='Original', marker='o')
            axs[row, 1].plot(df['LSB_Zeroed'], df['Compressed_Entropy_Coarse'], label='Compressed', marker='x')
            axs[row, 1].set_title(f'{dist_name}: Entropy (Coarse Bins)')
            axs[row, 1].set_xlabel('LSB Zeroed')
            axs[row, 1].set_ylabel('Entropy')
            axs[row, 1].legend()

            # KL Divergence
            axs[row, 2].plot(df['LSB_Zeroed'], df['KL_Divergence_Fine'], label='Fine Bins', marker='o')
            axs[row, 2].plot(df['LSB_Zeroed'], df['KL_Divergence_Coarse'], label='Coarse Bins', marker='x')
            axs[row, 2].set_title(f'{dist_name}: KL Divergence')
            axs[row, 2].set_xlabel('LSB Zeroed')
            axs[row, 2].set_ylabel('KL Divergence')
            axs[row, 2].legend()

            # Bit-level Entropy
            axs[row, 3].plot(df['LSB_Zeroed'], df['Original_Bit_Entropy'], label='Original', marker='o')
            axs[row, 3].plot(df['LSB_Zeroed'], df['Compressed_Bit_Entropy'], label='Compressed', marker='x')
            axs[row, 3].set_title(f'{dist_name}: Bit-level Entropy')
            axs[row, 3].set_xlabel('LSB Zeroed')
            axs[row, 3].set_ylabel('Bit Entropy')
            axs[row, 3].legend()

            # Compression Ratio and MSE
            axs[row, 4].plot(df['LSB_Zeroed'], df['Compression_Ratio'], label='Compression Ratio', marker='o')
            axs[row, 4].set_title(f'{dist_name}: Compression Metrics')
            axs[row, 4].set_xlabel('LSB Zeroed')
            axs[row, 4].set_ylabel('Compression Ratio')
            
            # Add a second y-axis for MSE
            ax2 = axs[row, 4].twinx()
            ax2.semilogy(df['LSB_Zeroed'], df['MSE'], label='MSE', color='red', marker='x')
            ax2.set_ylabel('Mean Squared Error (log scale)')
            
            # Combine legends
            lines1, labels1 = axs[row, 4].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axs[row, 4].legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_compression_analysis.pdf'), dpi=300)
        plt.close()
        
        # Generate distribution comparison visualizations
        self._generate_distribution_comparisons(distributions)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Generate comprehensive report
        self._generate_report(results_df)
        
        # Save results to JSON
        results_df.to_json(os.path.join(self.output_dir, 'compression_analysis_results.json'), orient='records')
        
        return results_df
    
    def _generate_distribution_comparisons(self, distributions):
        """
        Generate visualizations comparing original and compressed distributions
        
        :param distributions: Dictionary of distributions
        """
        for dist_name, original_data in distributions.items():
            # Create figure for this distribution
            plt.figure(figsize=(18, 10))
            
            # Select a subset for visualization
            sample_size = min(10000, len(original_data))
            sample_indices = np.random.choice(len(original_data), sample_size, replace=False)
            sample_data = original_data[sample_indices]
            
            # Plot original distribution
            plt.subplot(2, 4, 1)
            sns.histplot(sample_data, kde=True)
            plt.title(f'Original {dist_name} Distribution')
            
            # Plot compressed distributions for different LSB levels
            for i, lsb in enumerate(self.lsb_levels):
                if i >= 7:  # Only show up to 7 compression levels
                    break
                    
                # Compress the data
                result = self.compress_data(sample_data, lsb)
                compressed_data = result['compressed_data']
                
                # Plot histogram
                plt.subplot(2, 4, i+2)
                sns.histplot(compressed_data, kde=True)
                plt.title(f'{dist_name} with {lsb} LSB Zeroed')
                plt.xlabel(f'MSE: {result["mse"]:.2e}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{dist_name}_distribution_comparison.pdf'), dpi=300)
            plt.close()


    def compress_data(self, data, num_bits):
        """
        Compress data by zeroing out least significant bits
        
        :param data: Input array of floating-point numbers
        :param num_bits: Number of least significant bits to zero out
        :return: Compressed data and analysis metrics
        """
        # Create copy to avoid modifying original data
        data_copy = data.copy()
        
        # Convert to byte representation
        byte_data = data_copy.tobytes()
        original_bytes = len(byte_data)
        
        # Get view of data as unsigned 64-bit integers (IEEE 754 representation)
        int_view = data_copy.view(np.uint64)
        
        # Create a mask to zero out least significant bits
        mask = np.uint64(~((1 << num_bits) - 1))
        
        # Apply the mask to modify the data in place
        int_view[:] = int_view & mask
        
        # Now data_copy contains the compressed values
        compressed_data = data_copy
        
        # For a more visible impact, add controlled noise if needed
        # This is optional and depends on your application needs
        # compressed_data += np.random.normal(0, 1e-10, size=len(compressed_data))
        
        # Additional analysis
        compressed_bytes = zlib.compress(compressed_data.tobytes())
        
        # Calculate direct bit-level entropy before and after compression
        # This gives a more sensitive measure of information content
        original_bytes_array = np.frombuffer(data.tobytes(), dtype=np.uint8)
        compressed_bytes_array = np.frombuffer(compressed_data.tobytes(), dtype=np.uint8)
        
        # Bit-level entropy calculation
        orig_bit_counts = np.zeros(8)
        comp_bit_counts = np.zeros(8)
        
        for i in range(8):
            orig_bit_counts[i] = np.sum((original_bytes_array & (1 << i)) > 0)
            comp_bit_counts[i] = np.sum((compressed_bytes_array & (1 << i)) > 0)
        
        orig_bit_entropy = -np.sum((orig_bit_counts/len(original_bytes_array)) * 
                                  np.log2(orig_bit_counts/len(original_bytes_array) + 1e-10))
        comp_bit_entropy = -np.sum((comp_bit_counts/len(compressed_bytes_array)) * 
                                  np.log2(comp_bit_counts/len(compressed_bytes_array) + 1e-10))
        
        return {
            'original_data': data,
            'compressed_data': compressed_data,
            'original_entropy_fine': self.calculate_entropy(data, bins=200),
            'original_entropy_coarse': self.calculate_entropy(data, bins=50),
            'compressed_entropy_fine': self.calculate_entropy(compressed_data, bins=200),
            'compressed_entropy_coarse': self.calculate_entropy(compressed_data, bins=50),
            'kl_divergence_fine': self.calculate_kl_divergence(data, compressed_data, bins=200),
            'kl_divergence_coarse': self.calculate_kl_divergence(data, compressed_data, bins=50),
            'original_size': original_bytes,
            'compressed_size': len(compressed_bytes),
            'compression_ratio': original_bytes / len(compressed_bytes),
            'mse': np.mean((data - compressed_data)**2),
            'max_abs_error': np.max(np.abs(data - compressed_data)),
            'orig_bit_entropy': orig_bit_entropy,
            'comp_bit_entropy': comp_bit_entropy,
            'bit_entropy_diff': orig_bit_entropy - comp_bit_entropy
        }
    
    def _generate_report(self, results_df):
        """
        Generate a comprehensive text report
        
        :param results_df: DataFrame with compression and entropy results
        """
        # Report path
        report_path = os.path.join(self.output_dir, 'comprehensive_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Comprehensive Compression and Entropy Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary for each distribution
            for dist in results_df['Distribution'].unique():
                dist_data = results_df[results_df['Distribution'] == dist]
                
                f.write(f"Analysis for {dist} Distribution:\n")
                f.write("-" * 30 + "\n")
                
                # Compute and write statistical insights
                f.write("Statistical Insights:\n")
                f.write(f"  Average Compression Ratio: {dist_data['Compression_Ratio'].mean():.4f}\n")
                f.write(f"  Average Fine Bin Entropy Loss: {(dist_data['Original_Entropy_Fine'] - dist_data['Compressed_Entropy_Fine']).mean():.4f}\n")
                f.write(f"  Average Coarse Bin Entropy Loss: {(dist_data['Original_Entropy_Coarse'] - dist_data['Compressed_Entropy_Coarse']).mean():.4f}\n")
                f.write(f"  Average KL Divergence (Fine Bins): {dist_data['KL_Divergence_Fine'].mean():.4f}\n")
                f.write(f"  Average Bit Entropy Difference: {dist_data['Bit_Entropy_Difference'].mean():.4f}\n")
                f.write(f"  Average Mean Squared Error: {dist_data['MSE'].mean():.6e}\n")
                
                # Show compression level impact
                f.write("\nImpact of Compression Levels:\n")
                for lsb in dist_data['LSB_Zeroed'].unique():
                    lsb_data = dist_data[dist_data['LSB_Zeroed'] == lsb]
                    f.write(f"  LSB {lsb}:\n")
                    f.write(f"    Compression Ratio: {lsb_data['Compression_Ratio'].values[0]:.4f}\n")
                    f.write(f"    KL Divergence: {lsb_data['KL_Divergence_Fine'].values[0]:.6f}\n")
                    f.write(f"    MSE: {lsb_data['MSE'].values[0]:.6e}\n")
                    f.write(f"    Bit Entropy Diff: {lsb_data['Bit_Entropy_Difference'].values[0]:.6f}\n\n")
        
        print(f"Comprehensive report generated at: {report_path}")

    def generate_bit_pattern_analysis(self, distributions):
        """
        Generate bit pattern analysis for floating point numbers
        
        :param distributions: Dictionary of distributions
        """
        # Create figure for bit pattern analysis
        plt.figure(figsize=(20, 15))
        
        for i, (dist_name, original_data) in enumerate(distributions.items()):
            # Sample data for analysis
            sample_size = min(1000, len(original_data))
            sample_indices = np.random.choice(len(original_data), sample_size, replace=False)
            sample_data = original_data[sample_indices]
            
            # Convert to binary representation
            binary_representations = []
            for val in sample_data[:10]:  # Only show a few examples
                bits = ''.join(bin(b)[2:].zfill(8) for b in struct.pack('!d', val))
                binary_representations.append(bits)
            
            # Plot sample binary representations
            plt.subplot(len(distributions), 2, 2*i+1)
            for j, bits in enumerate(binary_representations):
                y_pos = j
                for k, bit in enumerate(bits):
                    if bit == '1':
                        plt.plot(k, y_pos, 'ko', markersize=3)
            plt.title(f'{dist_name} Binary Patterns (Sample)')
            plt.xlabel('Bit Position')
            plt.ylabel('Sample Index')
            plt.xlim(0, 64)
            plt.ylim(-0.5, len(binary_representations)-0.5)
            
            # Bit position statistics
            plt.subplot(len(distributions), 2, 2*i+2)
            bit_counts = np.zeros(64)
            for val in sample_data:
                bits = ''.join(bin(b)[2:].zfill(8) for b in struct.pack('!d', val))
                for k, bit in enumerate(bits):
                    if bit == '1':
                        bit_counts[k] += 1
            
            # Normalize
            bit_counts = bit_counts / len(sample_data)
            
            # Plot with IEEE 754 regions highlighted
            plt.bar(range(64), bit_counts)
            plt.axvspan(0, 1, alpha=0.2, color='red', label='Sign')
            plt.axvspan(1, 12, alpha=0.2, color='green', label='Exponent')
            plt.axvspan(12, 64, alpha=0.2, color='blue', label='Mantissa')
            
            plt.title(f'{dist_name} Bit Frequency')
            plt.xlabel('Bit Position')
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bit_pattern_analysis.pdf'), dpi=300)
        plt.close()

    def _write_performance_report(self, perf_df):
        """
        Generate a comprehensive text report of multicore performance
        
        :param perf_df: DataFrame with performance measurements
        """
        # Report path
        report_path = os.path.join(self.output_dir, 'multicore_performance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Multicore Performance Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            f.write("Overall Performance Summary:\n")
            f.write("-" * 30 + "\n")
            
            # Average speedup for each core count
            f.write("Average Speedup by Core Count:\n")
            for cores in sorted(perf_df['Num_Cores'].unique()):
                avg_speedup = perf_df[perf_df['Num_Cores'] == cores]['Speedup'].mean()
                ideal_speedup = cores
                efficiency = avg_speedup / ideal_speedup * 100
                f.write(f"  {cores} cores: {avg_speedup:.2f}x speedup ({efficiency:.1f}% efficiency)\n")
            
            f.write("\n")
            
            # Detail by distribution
            for dist_name in sorted(perf_df['Distribution'].unique()):
                dist_data = perf_df[perf_df['Distribution'] == dist_name]
                
                f.write(f"Performance for {dist_name} Distribution:\n")
                f.write("-" * 30 + "\n")
                
                # By LSB level and core count
                for lsb in sorted(dist_data['LSB_Zeroed'].unique()):
                    lsb_data = dist_data[dist_data['LSB_Zeroed'] == lsb]
                    
                    f.write(f"  LSB {lsb}:\n")
                    for cores in sorted(lsb_data['Num_Cores'].unique()):
                        core_data = lsb_data[lsb_data['Num_Cores'] == cores]
                        time = core_data['Execution_Time'].values[0]
                        speedup = core_data['Speedup'].values[0]
                        efficiency = core_data['Efficiency'].values[0] * 100
                        
                        f.write(f"    {cores} cores: {time:.4f}s, {speedup:.2f}x speedup, {efficiency:.1f}% efficiency\n")
                    
                    # Add a newline between LSB levels
                    f.write("\n")
                
                # Add a newline between distributions
                f.write("\n")
            
            # Performance bottlenecks and insights
            f.write("Performance Insights:\n")
            f.write("-" * 30 + "\n")
            
            # Calculate scaling efficiency
            max_cores = max(perf_df['Num_Cores'])
            scaling_efficiency = perf_df[perf_df['Num_Cores'] == max_cores]['Speedup'].mean() / max_cores * 100
            
            f.write(f"Overall scaling efficiency at {max_cores} cores: {scaling_efficiency:.1f}%\n")
            
            # Amdahl's Law analysis
            if scaling_efficiency < 70:
                f.write("The parallel scaling is limited, suggesting significant serial components or overhead.\n")
                f.write("Estimated serial fraction based on Amdahl's Law: ")
                # Estimate serial fraction (s) from: speedup = 1 / (s + (1-s)/p)
                speedup = perf_df[perf_df['Num_Cores'] == max_cores]['Speedup'].mean()
                s = (max_cores - speedup) / (speedup * (max_cores - 1))
                f.write(f"{s:.2%}\n")
            else:
                f.write("The algorithm shows good parallel scaling properties.\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            if scaling_efficiency < 50:
                f.write("- Consider optimizing the serial portions of the algorithm\n")
                f.write("- Reduce inter-process communication\n")
                f.write("- Increase granularity of parallel tasks\n")
            elif scaling_efficiency < 80:
                f.write("- Consider larger dataset sizes to improve parallel efficiency\n")
                f.write("- Optimize data chunking to balance load across cores\n")
            else:
                f.write("- Current implementation shows good scaling, consider testing with larger core counts\n")
                f.write("- Experiment with different work distribution strategies\n")


# Execute the analysis
if __name__ == "__main__":
    # Clear previous output
    import shutil
    output_dir = 'enhanced_compression_analysis'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    compressor = EnhancedFloatingPointCompressor()
    distributions = compressor.generate_distributions()
    
    # Generate bit pattern analysis
    compressor.generate_bit_pattern_analysis(distributions)
    
    # Run main analysis
    print("\nRunning single-core compression analysis...")
    results = compressor.analyze_compression()
    print("Single-core analysis complete!")
    
    # Run multicore analysis
    print("\nRunning multicore compression analysis...")
    # Choose core counts to test (adjust based on available hardware)
    core_counts = [1, 2, 4, 8]
    # If running on a machine with fewer cores, use:
    # core_counts = [1, 2, mp.cpu_count()]
    multicore_results = compressor.analyze_compression_multicore(core_counts)
    print("Multicore analysis complete!")
    
    # Print summary of results
    print("\nCompression and Entropy Analysis Results:")
    print(results.head())
    
    print("\nMulticore Performance Summary:")
    performance_summary = multicore_results.groupby('Num_Cores')['Speedup'].mean().reset_index()
    for _, row in performance_summary.iterrows():
        print(f"  {int(row['Num_Cores'])} cores: {row['Speedup']:.2f}x speedup")
    
    print("\nComplete! Check the output directory for detailed analysis.")
