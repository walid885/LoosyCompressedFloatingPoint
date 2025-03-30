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
        self.lsb_levels = [8, 10, 12, 14, 16]
        
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
        
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        
        # Calculate entropy
        return -np.sum(hist * np.log2(hist))

    def calculate_kl_divergence(self, original_data, compressed_data, bins=100):
        """
        Calculate Kullback-Leibler Divergence between original and compressed data
        
        :param original_data: Original data array
        :param compressed_data: Compressed data array
        :param bins: Number of bins for discretization
        :return: KL Divergence value
        """
        # Compute histograms
        orig_hist, bin_edges = np.histogram(original_data, bins=bins, density=True)
        comp_hist, _ = np.histogram(compressed_data, bins=bin_edges, density=True)
        
        # Remove zero probabilities
        orig_hist = orig_hist[orig_hist > 0]
        comp_hist = comp_hist[comp_hist > 0]
        
        # Ensure same length by truncating to the shorter array
        min_len = min(len(orig_hist), len(comp_hist))
        orig_hist = orig_hist[:min_len]
        comp_hist = comp_hist[:min_len]
        
        # Calculate KL Divergence
        return np.sum(orig_hist * np.log2(orig_hist / comp_hist))

    def compress_data(self, data, num_bits):
        """
        Compress data by zeroing out least significant bits
        
        :param data: Input array of floating-point numbers
        :param num_bits: Number of least significant bits to zero out
        :return: Compressed data and analysis metrics
        """
        # Convert to 64-bit integer representation
        int_data = data.view(np.uint64)
        
        # Create a mask to zero out least significant bits
        mask = np.uint64(~((1 << num_bits) - 1))
        
        # Apply the mask
        compressed_int = int_data & mask
        
        # Convert back to floating-point
        compressed_data = compressed_int.view(np.float64)
        
        # Additional analysis
        byte_data = compressed_data.tobytes()
        compressed_bytes = zlib.compress(byte_data)
        
        return {
            'original_data': data,
            'compressed_data': compressed_data,
            'original_entropy_fine': self.calculate_entropy(data, bins=200),
            'original_entropy_coarse': self.calculate_entropy(data, bins=50),
            'compressed_entropy_fine': self.calculate_entropy(compressed_data, bins=200),
            'compressed_entropy_coarse': self.calculate_entropy(compressed_data, bins=50),
            'kl_divergence_fine': self.calculate_kl_divergence(data, compressed_data, bins=200),
            'kl_divergence_coarse': self.calculate_kl_divergence(data, compressed_data, bins=50),
            'original_size': len(byte_data),
            'compressed_size': len(compressed_bytes),
            'compression_ratio': len(byte_data) / len(compressed_bytes),
            'mse': np.mean((data - compressed_data)**2),
            'max_abs_error': np.max(np.abs(data - compressed_data))
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
        fig, axs = plt.subplots(len(distributions), 4, figsize=(24, 4*len(distributions)))
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
                    'Max_Absolute_Error': result['max_abs_error']
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

            # Compression Ratio and MSE
            axs[row, 3].plot(df['LSB_Zeroed'], df['Compression_Ratio'], label='Compression Ratio', marker='o')
            axs[row, 3].set_title(f'{dist_name}: Compression Metrics')
            axs[row, 3].set_xlabel('LSB Zeroed')
            axs[row, 3].set_ylabel('Compression Ratio')
            
            # Add a second y-axis for MSE
            ax2 = axs[row, 3].twinx()
            ax2.plot(df['LSB_Zeroed'], df['MSE'], label='MSE', color='red', marker='x')
            ax2.set_ylabel('Mean Squared Error')
            
            # Combine legends
            lines1, labels1 = axs[row, 3].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axs[row, 3].legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_compression_analysis.pdf'), dpi=300)
        plt.close()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Generate comprehensive report
        self._generate_report(results_df)
        
        # Save results to JSON
        results_df.to_json(os.path.join(self.output_dir, 'compression_analysis_results.json'), orient='records')
        
        return results_df

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
                f.write(f"  Average Mean Squared Error: {dist_data['MSE'].mean():.6f}\n\n")
        
        print(f"Comprehensive report generated at: {report_path}")

# Execute the analysis
if __name__ == "__main__":
    # Clear previous output
    import shutil
    output_dir = 'enhanced_compression_analysis'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    compressor = EnhancedFloatingPointCompressor()
    results = compressor.analyze_compression()
    print("\nCompression and Entropy Analysis Results:")
    print(results)