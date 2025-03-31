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
    results = compressor.analyze_compression()
    print("\nCompression and Entropy Analysis Results:")
    print(results.head())
    
    print("\nComplete! Check the output directory for detailed analysis.")