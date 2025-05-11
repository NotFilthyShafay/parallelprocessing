import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Set the style for plots
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Read data from CSV files
results_dir = "../results"

# Read all CSV files
st_df = pd.read_csv(f"{results_dir}/single_threaded_performance.csv")
omp_df = pd.read_csv(f"{results_dir}/omp_performance.csv")
cuda_df = pd.read_csv(f"{results_dir}/cuda_performance.csv")
opencl_all_df = pd.read_csv(f"{results_dir}/opencl_performance_all.csv")

# Create output directory if it doesn't exist
os.makedirs("../plots", exist_ok=True)

# 1. Execution time comparison across implementations
plt.figure(figsize=(14, 8))

# Plot single-threaded CPU
plt.plot(st_df['MatrixSize'], st_df['ExecutionTime_ms'], 'o-', linewidth=2, 
         markersize=8, label='CPU Single-threaded')

# Plot OpenMP with 8 threads
omp_8_df = omp_df[omp_df['NumThreads'] == 8]
plt.plot(omp_8_df['MatrixSize'], omp_8_df['ExecutionTime_ms'], 's-', linewidth=2, 
         markersize=8, label='CPU OpenMP (8 threads)')

# Plot CUDA implementations
for kernel_type in cuda_df['KernelType'].unique():
    cuda_kernel_df = cuda_df[cuda_df['KernelType'] == kernel_type]
    plt.plot(cuda_kernel_df['MatrixSize'], cuda_kernel_df['ExecutionTime_ms'], '^-', linewidth=2, 
             markersize=8, label=f'CUDA {kernel_type}')

# Plot OpenCL - use best workgroup size (16)
for wg_size in [8, 16, 32]:
    opencl_wg_df = opencl_all_df[opencl_all_df['WorkgroupSize'] == wg_size]
    if not opencl_wg_df.empty:
        plt.plot(opencl_wg_df['MatrixSize'], opencl_wg_df['ExecutionTime_ms'], 'D-', linewidth=2, 
                markersize=8, label=f'OpenCL naive ({wg_size}×{wg_size})')

plt.title('Matrix Multiplication Execution Time Comparison', fontsize=16)
plt.xlabel('Matrix Size (N×N)', fontsize=14)
plt.ylabel('Execution Time (ms, log scale)', fontsize=14)
plt.yscale('log')
plt.grid(False)
plt.legend(fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.savefig('../plots/execution_time_comparison.png', dpi=300)

# 2. Speedup relative to single-threaded implementation
plt.figure(figsize=(14, 8))

# Create baseline times dictionary
baseline_times = {}
for _, row in st_df.iterrows():
    baseline_times[row['MatrixSize']] = row['ExecutionTime_ms']

# Display available matrix sizes for debugging
print("Available baseline matrix sizes:", sorted(baseline_times.keys()))

# Plot OpenMP speedups for each thread count
for thread_count in sorted(omp_df['NumThreads'].unique()):
    omp_thread_df = omp_df[omp_df['NumThreads'] == thread_count]
    # Make sure to only use matrix sizes that exist in both datasets
    valid_sizes = [size for size in omp_thread_df['MatrixSize'] if size in baseline_times]
    valid_times = [time for size, time in zip(omp_thread_df['MatrixSize'], omp_thread_df['ExecutionTime_ms']) 
                  if size in baseline_times]
    
    speedups = [baseline_times[size] / time for size, time in zip(valid_sizes, valid_times)]
    plt.plot(valid_sizes, speedups, 's-', linewidth=2, 
             markersize=8, label=f'OpenMP ({thread_count} threads)')

# Plot CUDA speedups
for kernel_type in cuda_df['KernelType'].unique():
    cuda_kernel_df = cuda_df[cuda_df['KernelType'] == kernel_type]
    # Make sure to only use matrix sizes that exist in both datasets
    valid_sizes = [size for size in cuda_kernel_df['MatrixSize'] if size in baseline_times]
    valid_times = [time for size, time in zip(cuda_kernel_df['MatrixSize'], cuda_kernel_df['ExecutionTime_ms']) 
                  if size in baseline_times]
    
    speedups = [baseline_times[size] / time for size, time in zip(valid_sizes, valid_times)]
    plt.plot(valid_sizes, speedups, '^-', linewidth=2, 
             markersize=8, label=f'CUDA {kernel_type}')

# Plot OpenCL speedups for each workgroup size
for wg_size in [8, 16, 32]:
    opencl_wg_df = opencl_all_df[opencl_all_df['WorkgroupSize'] == wg_size]
    if not opencl_wg_df.empty:
        # Make sure to only use matrix sizes that exist in both datasets
        valid_sizes = [size for size in opencl_wg_df['MatrixSize'] if size in baseline_times]
        valid_times = [time for size, time in zip(opencl_wg_df['MatrixSize'], opencl_wg_df['ExecutionTime_ms']) 
                      if size in baseline_times]
        
        speedups = [baseline_times[size] / time for size, time in zip(valid_sizes, valid_times)]
        plt.plot(valid_sizes, speedups, 'D-', linewidth=2, 
                markersize=8, label=f'OpenCL ({wg_size}×{wg_size})')

# Add reference line for no speedup (1x)
plt.axhline(y=1, color='r', linestyle='--', label='Baseline (Single-threaded CPU)')

plt.title('Speedup Relative to Single-threaded CPU Implementation', fontsize=16)
plt.xlabel('Matrix Size (N×N)', fontsize=14)
plt.ylabel('Speedup Factor (higher is better)', fontsize=14)
plt.grid(False)
plt.legend(fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.savefig('../plots/speedup_comparison.png', dpi=300)

# 3. OpenMP scaling with thread count
plt.figure(figsize=(14, 8))

for size in sorted(omp_df['MatrixSize'].unique()):
    omp_size_df = omp_df[omp_df['MatrixSize'] == size]
    plt.plot(omp_size_df['NumThreads'], omp_size_df['ExecutionTime_ms'], 'o-', linewidth=2, 
             markersize=8, label=f'Matrix Size {size}×{size}')

plt.title('OpenMP Performance Scaling with Thread Count', fontsize=16)
plt.xlabel('Number of Threads', fontsize=14)
plt.ylabel('Execution Time (ms, log scale)', fontsize=14)
plt.yscale('log')
plt.grid(False)
plt.legend(fontsize=12)
plt.xticks(omp_df['NumThreads'].unique())
plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.savefig('../plots/omp_thread_scaling.png', dpi=300)

# 4. GPU implementation comparison (CUDA vs OpenCL)
plt.figure(figsize=(14, 8))

# Prepare data for grouped bar chart
matrix_sizes = cuda_df['MatrixSize'].unique()
x = np.arange(len(matrix_sizes))
width = 0.2

# CUDA implementations
cuda_naive_times = cuda_df[cuda_df['KernelType'] == 'naive']['ExecutionTime_ms'].values
cuda_shared_times = cuda_df[cuda_df['KernelType'] == 'shared_memory']['ExecutionTime_ms'].values

# OpenCL with different workgroup sizes - get best time for each matrix size
opencl_best_times = []
for size in matrix_sizes:
    size_df = opencl_all_df[opencl_all_df['MatrixSize'] == size]
    if not size_df.empty:
        best_time = size_df['ExecutionTime_ms'].min()
        opencl_best_times.append(best_time)
    else:
        opencl_best_times.append(0)  # Placeholder if no data

# Plot the bars
plt.bar(x - width, cuda_naive_times, width, label='CUDA naive')
plt.bar(x, cuda_shared_times, width, label='CUDA shared memory')
plt.bar(x + width, opencl_best_times, width, label='OpenCL (best config)')

plt.title('GPU Implementation Comparison', fontsize=16)
plt.xlabel('Matrix Size (N×N)', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.xticks(x, matrix_sizes)
plt.legend(fontsize=12)
plt.grid(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.savefig('../plots/gpu_implementation_comparison.png', dpi=300)

# 5. Workgroup size impact (using actual data)
plt.figure(figsize=(14, 8))

# For each matrix size, show how workgroup size affects performance
for size in sorted(opencl_all_df['MatrixSize'].unique()):
    size_df = opencl_all_df[opencl_all_df['MatrixSize'] == size]
    plt.plot(size_df['WorkgroupSize'], size_df['ExecutionTime_ms'], 'o-', linewidth=2, 
             markersize=8, label=f'Matrix Size {size}×{size}')

plt.title('Impact of OpenCL Workgroup Size on Performance', fontsize=16)
plt.xlabel('Workgroup Size (N×N)', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.grid(False)
plt.legend(fontsize=12)
plt.xticks([8, 16, 32])
plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.savefig('../plots/workgroup_size_impact.png', dpi=300)

# 6. Correctness verification through checksums
plt.figure(figsize=(10, 6))

print("All plots have been generated and saved to the 'plots' directory.")