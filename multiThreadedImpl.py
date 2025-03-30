# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import seaborn as sns
import pandas as pd
from scipy import stats
import zlib
import json
import time
import shutil
# Ensure mpi4py is installed: pip install mpi4py
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    # Provide fallback values if MPI is not available (for single-process run)
    class FakeComm:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def gather(self, data, root=0): return [data] if root == 0 else None
        def Barrier(self): pass
        def bcast(self, data, root=0): return data
    MPI = None # Set MPI to None if not available
    print("WARNING: mpi4py not found. Running in single-process mode.")


# --- Compressor Class (methods called by each rank) ---
class EnhancedFloatingPointCompressor:
    def __init__(self, seed=42, sample_size=1_000_000, output_dir_base='mpi_compression_analysis'):
        # Initialize random seed - crucial for identical data generation across ranks
        # Use a potentially rank-specific seed if truly independent streams were needed,
        # but for identical input data, use the same seed.
        np.random.seed(seed)
        self.sample_size = sample_size

        # Output directory setup - base name, rank 0 will create specific version
        self.output_dir_base = output_dir_base
        self.output_dir = None # Will be set by rank 0 later

        # Define LSB levels here for access by all ranks
        self.lsb_levels = [8, 10, 12, 14, 16, 20, 24, 32] # Mantissa bits for float64
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        ]

    # --- Methods: generate_distributions, calculate_entropy, calculate_kl_divergence, compress_data ---
    # These methods are largely unchanged from the previous version.
    # They will be called independently by each MPI process on its assigned data/tasks.

    def generate_distributions(self):
        """Generates identical distributions on each calling rank due to fixed seed."""
        half_size = self.sample_size // 2
        other_half_size = self.sample_size - half_size
        distributions = {
            'Uniform': np.random.uniform(0, 100, self.sample_size).astype(np.float64),
            'Gaussian': np.random.normal(50, 15, self.sample_size).astype(np.float64),
            'Exponential': np.random.exponential(10, self.sample_size).astype(np.float64),
            'Skewed': np.random.lognormal(0, 1, self.sample_size).astype(np.float64),
            'Bimodal': np.concatenate([
                np.random.normal(20, 5, half_size),
                np.random.normal(80, 5, other_half_size)
            ]).astype(np.float64)
        }
        # Ensure no NaNs/Infs are generated if possible, or handle them later
        for name, data in distributions.items():
            if np.any(np.isinf(data)) or np.any(np.isnan(data)):
                 print(f"WARNING: NaNs or Infs generated in '{name}' distribution. Results may be affected.")
                 # Optionally clip or replace invalid values
                 # distributions[name] = np.nan_to_num(data, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        return distributions


    def calculate_entropy(self, data, bins=100):
        """Calculates Shannon entropy using histogram binning."""
        data = data[np.isfinite(data)] # Filter out non-finite values
        if data.size == 0: return 0.0 # Handle empty data after filtering
        try:
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_widths = np.diff(bin_edges)
            non_zero_indices = hist > 0
            hist = hist[non_zero_indices]
            # Ensure bin_widths aligns with filtered hist if bins were calculated automatically
            if len(bin_widths) > len(hist):
                # This case can happen if density=True produces bins where hist is zero.
                # We need bin widths corresponding to the non-zero hist bins.
                # A simple approach is using the average width if bins were uniform.
                # Or recalculate widths for the non-zero bins if edges are known.
                # Assuming uniform bins for simplicity here if mismatch:
                 avg_width = (data.max() - data.min()) / bins if bins > 0 else 1.0
                 bin_widths_nz = np.full(hist.shape, avg_width)

            elif len(bin_widths) < len(hist):
                 # This shouldn't happen with standard histogram usage
                 return np.nan # Indicate an issue
            else:
                 bin_widths_nz = bin_widths[non_zero_indices]

            if len(hist) == 0: return 0.0
            # Add epsilon for numerical stability with log2
            return -np.sum(hist * np.log2(hist + 1e-15) * bin_widths_nz)
        except ValueError as e:
            print(f"Error in calculate_entropy (data range possibly zero?): {e}")
            return np.nan # Indicate error


    def calculate_kl_divergence(self, original_data, compressed_data, bins=100):
        """Calculates Kullback-Leibler Divergence."""
        # Filter non-finite values
        original_data = original_data[np.isfinite(original_data)]
        compressed_data = compressed_data[np.isfinite(compressed_data)]
        if original_data.size == 0 or compressed_data.size == 0:
            return np.nan # Cannot compare empty distributions

        min_val = min(np.min(original_data), np.min(compressed_data))
        max_val = max(np.max(original_data), np.max(compressed_data))

        if np.isclose(min_val, max_val): # Handle cases where data range is near zero
             if np.allclose(original_data, compressed_data): return 0.0
             else: max_val = min_val + 1e-9 # Add tiny offset

        bin_edges = np.linspace(min_val, max_val, bins + 1)
        try:
            orig_hist, _ = np.histogram(original_data, bins=bin_edges, density=False)
            comp_hist, _ = np.histogram(compressed_data, bins=bin_edges, density=False)

            # Add small epsilon BEFORE normalization to handle zero counts
            epsilon = 1e-12
            orig_prob = (orig_hist + epsilon) / np.sum(orig_hist + epsilon)
            comp_prob = (comp_hist + epsilon) / np.sum(comp_hist + epsilon)

            # Calculate KL Divergence P || Q = sum(P(i) * log2(P(i) / Q(i)))
            # Use 'where' to avoid division by zero or log(0) where P(i) is effectively zero
            kl_div = np.sum(np.where(orig_prob > epsilon, orig_prob * np.log2(orig_prob / comp_prob), 0))

            # Check for potential numerical issues resulting in negative KL divergence
            if kl_div < -1e-9: # Allow for small floating point errors
                 print(f"Warning: Negative KL divergence encountered ({kl_div}). Check inputs/epsilon.")
                 return np.nan # Or handle as error
            return max(0.0, kl_div) # Ensure non-negativity

        except ValueError as e:
             print(f"Error calculating KL divergence (data range issue?): {e}")
             return np.nan


    def compress_data(self, data, num_bits):
        """Compresses data by zeroing LSBs of the mantissa (float64)."""
        if not (0 <= num_bits <= 52):
            raise ValueError("num_bits must be between 0 and 52 for float64 mantissa.")

        # Work on a copy, ensure input is float64
        data_orig = data.astype(np.float64, copy=True)
        data_copy = data_orig.copy() # We need original later for metrics

        # Filter non-finite values BEFORE compression, as bitmasking NaNs/Infs is undefined
        finite_mask = np.isfinite(data_copy)
        if not np.all(finite_mask):
             # print(f"Warning: Non-finite values found in data for LSB={num_bits}. Compressing only finite values.")
             pass # We'll operate only on the finite view

        int_view = data_copy.view(np.uint64)

        if num_bits > 0:
            # Mask zeros out the last num_bits OF THE MANTISSA
            mask = np.uint64(~((1 << num_bits) - 1))
            # Apply mask only to finite values
            int_view[finite_mask] &= mask

        # data_copy now contains compressed values (or original NaNs/Infs)
        compressed_data = data_copy

        # Calculate metrics using only the originally finite values for fair comparison
        data_orig_finite = data_orig[finite_mask]
        compressed_data_finite = compressed_data[finite_mask]

        # Handle cases where filtering leaves no data
        if data_orig_finite.size == 0:
            return {
                'original_entropy_fine': np.nan, 'original_entropy_coarse': np.nan,
                'compressed_entropy_fine': np.nan, 'compressed_entropy_coarse': np.nan,
                'kl_divergence_fine': np.nan, 'kl_divergence_coarse': np.nan,
                'original_size': data_orig.nbytes, 'compressed_size_zlib': 0,
                'compression_ratio': np.inf, 'mse': np.nan, 'max_abs_error': np.nan,
            }

        # Calculate zlib size on the *full* potentially sparse array after compression
        # as this reflects practical storage benefit.
        try:
             compressed_bytes_zlib = zlib.compress(compressed_data.tobytes())
             zlib_size = len(compressed_bytes_zlib)
             comp_ratio = data_orig.nbytes / zlib_size if zlib_size > 0 else np.inf
        except Exception as e:
             print(f"Error during zlib compression: {e}")
             zlib_size = -1 # Indicate error
             comp_ratio = np.nan


        return {
            # Metrics calculated only on originally finite data points
            'original_entropy_fine': self.calculate_entropy(data_orig_finite, bins=200),
            'original_entropy_coarse': self.calculate_entropy(data_orig_finite, bins=50),
            'compressed_entropy_fine': self.calculate_entropy(compressed_data_finite, bins=200),
            'compressed_entropy_coarse': self.calculate_entropy(compressed_data_finite, bins=50),
            'kl_divergence_fine': self.calculate_kl_divergence(data_orig_finite, compressed_data_finite, bins=200),
            'kl_divergence_coarse': self.calculate_kl_divergence(data_orig_finite, compressed_data_finite, bins=50),
            'original_size': data_orig.nbytes, # Size of the original full array
            'compressed_size_zlib': zlib_size,
            'compression_ratio': comp_ratio,
            'mse': np.mean((data_orig_finite - compressed_data_finite)**2),
            'max_abs_error': np.max(np.abs(data_orig_finite - compressed_data_finite)) if data_orig_finite.size > 0 else 0.0,
        }

    # --- Plotting, Reporting, Bit Analysis (Run by Rank 0 Only) ---
    def _create_output_dir(self, num_procs):
        """Creates the output directory string and the directory itself."""
        self.output_dir = f"{self.output_dir_base}_{num_procs}procs"
        # Check existence and remove ONLY if it's the base name structure
        # To prevent accidentally removing other directories
        if os.path.exists(self.output_dir) and self.output_dir.startswith(self.output_dir_base):
            print(f"Rank 0: Removing existing output directory: {self.output_dir}")
            try:
                shutil.rmtree(self.output_dir)
            except OSError as e:
                 print(f"Rank 0 Warning: Could not remove directory {self.output_dir}: {e}")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Rank 0: Ensured output directory exists: {self.output_dir}")
        except OSError as e:
             print(f"Rank 0 Error: Could not create directory {self.output_dir}: {e}")
             # Fallback or raise error? For now, print error and continue if possible.


    def _generate_plots(self, results_df, num_procs):
        """Generates the main analysis plot. Assumes called by Rank 0."""
        if self.output_dir is None:
             print("Rank 0 Error: Output directory not set for plotting.")
             return
        if results_df.empty:
            print("Rank 0: No results to plot.")
            return

        print("Rank 0: Generating analysis plot...")
        unique_distributions = results_df['Distribution'].unique() # Keep Uppercase 'D'
        num_dist = len(unique_distributions)
        fig, axs = plt.subplots(num_dist, 4, figsize=(24, 4 * num_dist), squeeze=False)
        fig.suptitle(f'Compression Analysis (using {num_procs} MPI processes)', fontsize=16, y=1.02)

        for row, dist_name in enumerate(unique_distributions):
            # Keep Uppercase 'D' and 'LSB' for filtering/sorting
            df = results_df[results_df['Distribution'] == dist_name].sort_values('LSB_Zeroed')
            if df.empty: continue

            # --- FIX: Use lowercase keys for metrics from compress_data ---
            # Fine/Coarse Entropy
            # Use .iloc[[0]*len(df)] to repeat the first original value for plotting baseline
            axs[row, 0].plot(df['LSB_Zeroed'], df['original_entropy_fine'].iloc[[0]*len(df)], label='Orig Fine', marker='s', linestyle='--', color='blue')
            axs[row, 0].plot(df['LSB_Zeroed'], df['compressed_entropy_fine'], label='Comp Fine', marker='x', color='blue')
            axs[row, 0].plot(df['LSB_Zeroed'], df['original_entropy_coarse'].iloc[[0]*len(df)], label='Orig Coarse', marker='o', linestyle='--', color='orange')
            axs[row, 0].plot(df['LSB_Zeroed'], df['compressed_entropy_coarse'], label='Comp Coarse', marker='.', color='orange')
            axs[row, 0].set_title(f'{dist_name}: Entropy (Value)')
            axs[row, 0].set_xlabel('LSB Mantissa Bits Zeroed')
            axs[row, 0].set_ylabel('Entropy')
            axs[row, 0].legend(fontsize='small'); axs[row, 0].grid(True, linestyle=':')

            # KL Divergence
            axs[row, 1].plot(df['LSB_Zeroed'], df['kl_divergence_fine'], label='KL Fine', marker='o', color='green')
            axs[row, 1].plot(df['LSB_Zeroed'], df['kl_divergence_coarse'], label='KL Coarse', marker='x', color='red')
            axs[row, 1].set_title(f'{dist_name}: KL Divergence')
            axs[row, 1].set_xlabel('LSB Mantissa Bits Zeroed'); axs[row, 1].set_ylabel('KL Divergence (bits)')
            axs[row, 1].legend(fontsize='small'); axs[row, 1].grid(True, linestyle=':')

            # Max Absolute Error
            axs[row, 2].semilogy(df['LSB_Zeroed'], df['max_abs_error'], label='Max Abs Err', marker='^', color='purple')
            axs[row, 2].set_title(f'{dist_name}: Max Absolute Error')
            axs[row, 2].set_xlabel('LSB Mantissa Bits Zeroed'); axs[row, 2].set_ylabel('Max Abs Error (log)')
            axs[row, 2].legend(fontsize='small'); axs[row, 2].grid(True, linestyle=':')

            # Compression Ratio & MSE
            ax_comp = axs[row, 3]
            ax_comp.plot(df['LSB_Zeroed'], df['compression_ratio'], label='zlib Ratio', marker='o', color='tab:blue')
            ax_comp.set_title(f'{dist_name}: Comp & MSE'); ax_comp.set_xlabel('LSB Mantissa Bits Zeroed')
            ax_comp.set_ylabel('zlib Comp Ratio', color='tab:blue')
            ax_comp.tick_params(axis='y', labelcolor='tab:blue'); ax_comp.grid(True, linestyle=':')
            ax_mse = ax_comp.twinx()
            ax_mse.semilogy(df['LSB_Zeroed'], df['mse'], label='MSE', color='tab:red', marker='x')
            ax_mse.set_ylabel('MSE (log)', color='tab:red')
            ax_mse.tick_params(axis='y', labelcolor='tab:red')
            lines1, labels1 = ax_comp.get_legend_handles_labels(); lines2, labels2 = ax_mse.get_legend_handles_labels()
            ax_comp.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize='small')
            # --- End Fix ---

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plot_filename = os.path.join(self.output_dir, f'compression_analysis_{num_procs}_procs.pdf')
        try:
            plt.savefig(plot_filename, dpi=300)
            print(f"Rank 0: Saved analysis plot: {plot_filename}")
        except Exception as e:
            print(f"Rank 0 Error: Failed to save plot {plot_filename}: {e}")
        finally:
            plt.close()




    def _generate_report(self, results_df, num_procs):
        """Generates the text report. Assumes called by Rank 0."""
        if self.output_dir is None:
             print("Rank 0 Error: Output directory not set for reporting.")
             return
        if results_df.empty:
            print("Rank 0: No results to write report.")
            return

        print("Rank 0: Generating text report...")
        report_filename = os.path.join(self.output_dir, f'analysis_report_{num_procs}_procs.txt')
        try:
            with open(report_filename, 'w') as f:
                f.write(f"MPI Compression and Entropy Analysis Report ({num_procs} Processes)\n")
                f.write("=" * 60 + "\n\n")

                # Keep Uppercase 'D' and 'LSB'
                for dist in results_df['Distribution'].unique():
                    dist_data = results_df[results_df['Distribution'] == dist].sort_values('LSB_Zeroed')
                    if dist_data.empty: continue

                    f.write(f"Analysis for {dist} Distribution:\n")
                    f.write("-" * 40 + "\n")

                    # --- FIX: Use lowercase keys for metrics from compress_data ---
                    f.write("Average Metrics Across LSB Levels (NaNs ignored):\n")
                    f.write(f"  Avg zlib Compression Ratio: {np.nanmean(dist_data['compression_ratio']):.4f}\n") # lowercase 'c'
                    orig_ent_fine = dist_data['original_entropy_fine'].iloc[0] # lowercase 'o'
                    orig_ent_coarse = dist_data['original_entropy_coarse'].iloc[0] # lowercase 'o'

                    if not np.isnan(orig_ent_fine):
                         fine_loss = orig_ent_fine - dist_data['compressed_entropy_fine'] # lowercase 'c'
                         f.write(f"  Avg Fine Bin Entropy Loss: {np.nanmean(fine_loss):.4f}\n")
                    else: f.write("  Avg Fine Bin Entropy Loss: NaN (Original NaN)\n")
                    if not np.isnan(orig_ent_coarse):
                         coarse_loss = orig_ent_coarse - dist_data['compressed_entropy_coarse'] # lowercase 'c'
                         f.write(f"  Avg Coarse Bin Entropy Loss: {np.nanmean(coarse_loss):.4f}\n")
                    else: f.write("  Avg Coarse Bin Entropy Loss: NaN (Original NaN)\n")

                    f.write(f"  Avg KL Divergence (Fine Bins): {np.nanmean(dist_data['kl_divergence_fine']):.4f}\n") # lowercase 'k'
                    f.write(f"  Avg Mean Squared Error: {np.nanmean(dist_data['mse']):.6e}\n") # lowercase 'm'
                    f.write(f"  Avg Max Absolute Error: {np.nanmean(dist_data['max_abs_error']):.6e}\n\n") # lowercase 'm'
                    f.write("Impact per LSB Zeroed Level:\n")
                    f.write(" LSB | zlib Ratio | KL Div Fine |      MSE      | Max Abs Error \n")
                    f.write("-----|------------|-------------|---------------|---------------\n")
                    for _, row in dist_data.iterrows():
                         # Keep Uppercase 'LSB'
                         # Use lowercase for metrics
                        f.write(f" {row['LSB_Zeroed']:<3} |   {row['compression_ratio']:<8.3f} |   {row['kl_divergence_fine']:<9.4f} | {row['mse']:<13.4e} | {row['max_abs_error']:<13.4e}\n")
                    # --- End Fix ---
                    f.write("\n" + "=" * 40 + "\n\n")

            print(f"Rank 0: Comprehensive report generated: {report_filename}")
        except Exception as e:
            print(f"Rank 0 Error: Failed to write report {report_filename}: {e}")


    def generate_bit_pattern_analysis(self, distributions):
        """Generates bit pattern analysis plot. Assumes called by Rank 0."""
        if self.output_dir is None:
             print("Rank 0 Error: Output directory not set for bit pattern analysis.")
             return

        print("Rank 0: Generating bit pattern analysis plot...")
        # (Bit pattern plotting code is identical to the previous MPI version)
        # ... (bit pattern plotting logic using axs[i, 0] and axs[i, 1]) ...
        num_dist = len(distributions)
        fig, axs = plt.subplots(num_dist, 2, figsize=(15, 5 * num_dist), squeeze=False)
        fig.suptitle('Bit Pattern Analysis (IEEE 754 Double Precision)', fontsize=16, y=1.0)

        for i, (dist_name, original_data) in enumerate(distributions.items()):
            sample_size_bits = min(2000, len(original_data))
            sample_indices = np.random.choice(len(original_data), sample_size_bits, replace=False)
            sample_data = original_data[sample_indices]

            # Plot sample patterns
            ax_pattern = axs[i, 0]; num_examples = min(15, sample_size_bits)
            binary_representations = []
            for val in sample_data[:num_examples]:
                 try: packed = struct.pack('!d', val); bits = ''.join(format(byte, '08b') for byte in packed); binary_representations.append(bits)
                 except (struct.error, OverflowError): binary_representations.append(" " * 64)
            for j, bits in enumerate(binary_representations):
                y_pos = j
                for k, bit in enumerate(bits):
                    if bit == '1': ax_pattern.plot(k, y_pos, 'ko', markersize=2)
                    elif bit == '0': ax_pattern.plot(k, y_pos, 'o', mfc='white', mec='grey', markersize=2)
            ax_pattern.set_title(f'{dist_name} Patterns (Sample)'); ax_pattern.set_xlabel('Bit Pos (0=S, 1-11=E, 12-63=M)')
            ax_pattern.set_ylabel('Sample Idx'); ax_pattern.set_xlim(-1, 64); ax_pattern.set_ylim(-0.5, num_examples - 0.5)
            ax_pattern.set_yticks(range(num_examples)); ax_pattern.grid(True, axis='x', ls=':', lw=0.5)

            # Plot frequency
            ax_freq = axs[i, 1]
            bit_counts = np.zeros(64, dtype=float); valid_samples = 0
            for val in sample_data:
                 try: packed = struct.pack('!d', val); bits = ''.join(format(byte, '08b') for byte in packed)
                 except (struct.error, OverflowError): continue # Skip invalid floats for frequency
                 for k, bit in enumerate(bits):
                     if bit == '1': bit_counts[k] += 1
                 valid_samples += 1
            bit_freq = bit_counts / valid_samples if valid_samples > 0 else np.zeros(64)
            ax_freq.bar(range(64), bit_freq, width=0.8)
            ax_freq.axvspan(-0.5, 0.5, alpha=0.15, color='red', label='Sign (1)')
            ax_freq.axvspan(0.5, 11.5, alpha=0.15, color='green', label='Exp (11)')
            ax_freq.axvspan(11.5, 63.5, alpha=0.15, color='blue', label='Mant (52)')
            ax_freq.set_title(f'{dist_name} Bit Freq'); ax_freq.set_xlabel('Bit Pos'); ax_freq.set_ylabel('Freq of "1"s')
            ax_freq.set_ylim(0, 1); ax_freq.legend(fontsize='small'); ax_freq.grid(True, axis='y', ls=':')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        bit_pattern_filename = os.path.join(self.output_dir, 'bit_pattern_analysis.pdf')
        try:
            plt.savefig(bit_pattern_filename, dpi=300)
            print(f"Rank 0: Saved bit pattern analysis: {bit_pattern_filename}")
        except Exception as e:
            print(f"Rank 0 Error: Failed to save bit pattern plot {bit_pattern_filename}: {e}")
        finally:
            plt.close()

# --- Main Execution Block ---
# (No changes needed in the main block regarding key names, as it correctly
#  uses Uppercase 'Distribution' and 'LSB_Zeroed' when adding them, and
#  relies on the DataFrame column names created from the lowercase keys
#  returned by compress_data for the actual metrics).
if __name__ == "__main__":

    # --- MPI Initialization ---
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
    else:
        comm = FakeComm() # Use fallback for single process run

    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Configuration ---
    sample_size_main = 1_000_000 # Can be reduced for faster testing
    random_seed = 42
    output_dir_base_name = 'mpi_compression_analysis'

    # --- Setup ---
    compressor = EnhancedFloatingPointCompressor(seed=random_seed,
                                                 sample_size=sample_size_main,
                                                 output_dir_base=output_dir_base_name)
    start_time = 0.0
    if rank == 0:
        start_time = time.time()
        print(f"--- Starting MPI Analysis using {size} process(es) ---")
        compressor._create_output_dir(size)

    # --- Data Generation ---
    if rank == 0: print("All Ranks: Generating initial data distributions...")
    comm.Barrier()
    distributions = compressor.generate_distributions()
    if rank == 0: print("All Ranks: Distributions generated.")

    # --- Bit Pattern Analysis (Run Once by Rank 0) ---
    if rank == 0:
        if compressor.output_dir is None: compressor._create_output_dir(size)
        compressor.generate_bit_pattern_analysis(distributions)
    comm.Barrier()

    # --- Task Definition & Distribution ---
    all_tasks = []
    dist_names = list(distributions.keys())
    for dist_name in dist_names:
        for lsb in compressor.lsb_levels:
            all_tasks.append({'dist_name': dist_name, 'lsb': lsb})

    num_tasks = len(all_tasks)
    tasks_per_rank = num_tasks // size
    extra_tasks = num_tasks % size
    start_index = rank * tasks_per_rank + min(rank, extra_tasks)
    end_index = start_index + tasks_per_rank + (1 if rank < extra_tasks else 0)
    my_tasks = all_tasks[start_index:end_index]

    if rank == 0:
        print(f"Rank 0: Total tasks = {num_tasks}. Distributing among {size} processes.")

    # --- Execute Assigned Tasks ---
    local_results = []
    if rank == 0: print("All Ranks: Starting parallel computation...")
    for i, task in enumerate(my_tasks):
        dist_name = task['dist_name']
        lsb = task['lsb']

        original_data = distributions[dist_name]
        try:
            # compress_data returns lowercase keys
            result_dict = compressor.compress_data(original_data, lsb)
            # Add identifiers (using Uppercase as intended later)
            result_dict['Distribution'] = dist_name
            result_dict['LSB_Zeroed'] = lsb
            local_results.append(result_dict)
        except Exception as e:
            print(f"Rank {rank} Error: Failed processing task ({dist_name}, LSB {lsb}): {e}")
            # Append placeholder with lowercase keys matching compress_data failure
            local_results.append({
                'Distribution': dist_name, 'LSB_Zeroed': lsb, 'mse': np.nan,
                'original_entropy_fine': np.nan, 'original_entropy_coarse': np.nan,
                'compressed_entropy_fine': np.nan, 'compressed_entropy_coarse': np.nan,
                'kl_divergence_fine': np.nan, 'kl_divergence_coarse': np.nan,
                'original_size': -1, 'compressed_size_zlib': -1,
                'compression_ratio': np.nan, 'max_abs_error': np.nan,
            })


    # --- Gather Results ---
    if rank == 0: print(f"Rank 0: Computation finished. Gathering results ({len(local_results)} from self)...")
    try:
        all_results_nested = comm.gather(local_results, root=0)
    except Exception as e:
        print(f"Rank {rank}: Error during gather operation: {e}")
        all_results_nested = None

    comm.Barrier()

    # --- Post-Processing (Rank 0 Only) ---
    if rank == 0:
        end_time = time.time()
        duration = end_time - start_time
        print(f"--- MPI Analysis with {size} process(es) finished in {duration:.2f} seconds ---")

        all_results_flat = []
        if all_results_nested:
            for rank_results in all_results_nested:
                if rank_results is not None:
                     all_results_flat.extend(rank_results)
            print(f"Rank 0: Gathered {len(all_results_flat)} results total.")
        else:
            print("Rank 0 Warning: Gather operation failed or returned no data.")

        if not all_results_flat:
            print("Rank 0 Error: No results collected. Cannot proceed with analysis.")
        else:
            results_df = pd.DataFrame(all_results_flat)

            if results_df.empty:
                 print("Rank 0 Error: DataFrame is empty after potential NaN handling.")
            else:
                 # Sort results (using Uppercase keys as added)
                results_df = results_df.sort_values(by=['Distribution', 'LSB_Zeroed']).reset_index(drop=True)
                print("Rank 0: Sample of final results head:")
                # Displaying head will show the actual (lowercase) column names for metrics
                print(results_df.head())

                # Generate plots, report, and save JSON (these functions now expect lowercase metric keys)
                compressor._generate_plots(results_df, size)
                compressor._generate_report(results_df, size)

                json_filename = os.path.join(compressor.output_dir, f'compression_analysis_results_{size}_procs.json')
                try:
                    # Convert NaN to null for JSON compatibility if needed, orient='records' handles it well
                    results_df.to_json(json_filename, orient='records', indent=4, default_handler=str) # Added default_handler for safety
                    print(f"Rank 0: Saved analysis results: {json_filename}")
                except Exception as e:
                    print(f"Rank 0 Error: Failed to save results to JSON: {e}")


        print(f"\nRank 0: Complete! Check the '{compressor.output_dir}' directory.")
