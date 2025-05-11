import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def main():
    # Set up figure style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'font.size': 12})
    
    # Create results directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Load performance data
    cpu_data = load_data('laplace_cpu_performance.csv')
    omp_data = load_data('laplace_omp_performance.csv')
    cuda_data = load_data('laplace_performance.csv')
    
    # Also try to load OpenCL data if available
    try:
        opencl_data = load_data('laplace_opencl_performance.csv')
        has_opencl = True
    except:
        has_opencl = False
        print("OpenCL performance data not found. Proceeding without it.")
    
    # Group and aggregate data
    cpu_grouped = cpu_data.groupby('GridSize')['ExecutionTime_ms'].mean().reset_index()
    
    omp_grouped = omp_data.groupby(['GridSize', 'NumThreads'])['ExecutionTime_ms'].mean().reset_index()
    omp_threads = sorted(omp_data['NumThreads'].unique())
    
    cuda_grouped = cuda_data.groupby(['GridSize'])['ExecutionTime_ms'].mean().reset_index()
    
    if has_opencl:
        opencl_grouped = opencl_data.groupby(['GridSize', 'WorkgroupSize'])['ExecutionTime_ms'].mean().reset_index()
        opencl_workgroups = sorted(opencl_data['WorkgroupSize'].unique())
    
    # Plot 1: Execution time comparison
    plot_execution_times(cpu_grouped, omp_grouped, cuda_grouped, opencl_grouped if has_opencl else None, 
                        omp_threads, opencl_workgroups if has_opencl else None)
    
    # Plot 2: Speedup relative to CPU
    plot_speedup(cpu_grouped, omp_grouped, cuda_grouped, opencl_grouped if has_opencl else None, 
                omp_threads, opencl_workgroups if has_opencl else None)
    
    # Plot 3: OpenMP scaling with thread count
    plot_openmp_scaling(omp_grouped, omp_threads)
    
    # Plot 4: CUDA block size comparison (if multiple block sizes)
    if 'BlockSize' in cuda_data.columns:
        plot_cuda_block_sizes(cuda_data)
    
    # Plot 5: OpenCL workgroup comparison (if available)
    if has_opencl and len(opencl_workgroups) > 1:
        plot_opencl_workgroups(opencl_grouped, opencl_workgroups)
    
    print(f"Plots saved to the 'plots' directory.")

def load_data(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filename}. Please run the implementations first.")

def plot_execution_times(cpu_data, omp_data, cuda_data, opencl_data, omp_threads, opencl_workgroups):
    plt.figure(figsize=(12, 8))
    
    # Plot CPU performance
    plt.plot(cpu_data['GridSize'], cpu_data['ExecutionTime_ms'] / 1000, 
             'o-', color='blue', linewidth=2, markersize=8, label='CPU (Single-Threaded)')
    
    # Plot best OpenMP performance for each grid size
    best_omp = omp_data.loc[omp_data.groupby('GridSize')['ExecutionTime_ms'].idxmin()]
    plt.plot(best_omp['GridSize'], best_omp['ExecutionTime_ms'] / 1000, 
             's-', color='green', linewidth=2, markersize=8, 
             label=f'OpenMP (Best Thread Count)')
    
    # Plot CUDA performance
    plt.plot(cuda_data['GridSize'], cuda_data['ExecutionTime_ms'] / 1000, 
             '^-', color='red', linewidth=2, markersize=8, label='CUDA')
    
    # Plot OpenCL performance if available
    if opencl_data is not None:
        best_opencl = opencl_data.loc[opencl_data.groupby('GridSize')['ExecutionTime_ms'].idxmin()]
        plt.plot(best_opencl['GridSize'], best_opencl['ExecutionTime_ms'] / 1000, 
                 'd-', color='purple', linewidth=2, markersize=8, 
                 label='OpenCL (Best Workgroup Size)')
    
    plt.xlabel('Grid Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Laplace Solver Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/execution_times.png', dpi=300)

def plot_speedup(cpu_data, omp_data, cuda_data, opencl_data, omp_threads, opencl_workgroups):
    plt.figure(figsize=(12, 8))
    
    # Merge with CPU data to calculate speedup
    cpu_times = {row['GridSize']: row['ExecutionTime_ms'] for _, row in cpu_data.iterrows()}
    
    # Prepare data for plotting
    grid_sizes = sorted(cpu_data['GridSize'].unique())
    
    # OpenMP speedup - take best thread count for each grid size
    omp_speedups = []
    for size in grid_sizes:
        size_data = omp_data[omp_data['GridSize'] == size]
        if not size_data.empty:
            best_time = size_data['ExecutionTime_ms'].min()
            omp_speedups.append(cpu_times.get(size, 0) / best_time if best_time > 0 else 0)
        else:
            omp_speedups.append(0)
    
    # CUDA speedup
    cuda_speedups = []
    for size in grid_sizes:
        size_data = cuda_data[cuda_data['GridSize'] == size]
        if not size_data.empty:
            time = size_data['ExecutionTime_ms'].values[0]
            cuda_speedups.append(cpu_times.get(size, 0) / time if time > 0 else 0)
        else:
            cuda_speedups.append(0)
    
    # Plot speedups
    plt.plot(grid_sizes, [1] * len(grid_sizes), 'o-', color='blue', linewidth=2, 
             markersize=8, label='CPU (Single-Threaded - Baseline)')
    plt.plot(grid_sizes, omp_speedups, 's-', color='green', linewidth=2, 
             markersize=8, label='OpenMP (Best Thread Count)')
    plt.plot(grid_sizes, cuda_speedups, '^-', color='red', linewidth=2, 
             markersize=8, label='CUDA')
    
    # OpenCL speedup if available
    if opencl_data is not None:
        opencl_speedups = []
        for size in grid_sizes:
            size_data = opencl_data[opencl_data['GridSize'] == size]
            if not size_data.empty:
                best_time = size_data['ExecutionTime_ms'].min()
                opencl_speedups.append(cpu_times.get(size, 0) / best_time if best_time > 0 else 0)
            else:
                opencl_speedups.append(0)
        
        plt.plot(grid_sizes, opencl_speedups, 'd-', color='purple', linewidth=2, 
                 markersize=8, label='OpenCL (Best Workgroup Size)')
    
    plt.xlabel('Grid Size')
    plt.ylabel('Speedup Relative to CPU')
    plt.title('Laplace Solver Performance Speedup')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('plots/speedup.png', dpi=300)

def plot_openmp_scaling(omp_data, thread_counts):
    plt.figure(figsize=(12, 8))
    
    grid_sizes = sorted(omp_data['GridSize'].unique())
    
    for size in grid_sizes:
        size_data = omp_data[omp_data['GridSize'] == size]
        times = []
        threads = []
        
        for thread in thread_counts:
            thread_data = size_data[size_data['NumThreads'] == thread]
            if not thread_data.empty:
                times.append(thread_data['ExecutionTime_ms'].values[0])
                threads.append(thread)
        
        if times:
            # Normalize to single-thread performance
            normalized_times = [times[0] / t for t in times]
            plt.plot(threads, normalized_times, 'o-', linewidth=2, markersize=8, label=f'Grid Size {size}x{size}')
    
    plt.xlabel('Thread Count')
    plt.ylabel('Speedup')
    plt.title('OpenMP Scaling with Thread Count')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('plots/openmp_scaling.png', dpi=300)

def plot_cuda_block_sizes(cuda_data):
    if 'BlockSize' not in cuda_data.columns:
        return
    
    plt.figure(figsize=(12, 8))
    
    grid_sizes = sorted(cuda_data['GridSize'].unique())
    block_sizes = sorted(cuda_data['BlockSize'].unique())
    
    for size in grid_sizes:
        size_data = cuda_data[cuda_data['GridSize'] == size]
        times = []
        blocks = []
        
        for block in block_sizes:
            block_data = size_data[size_data['BlockSize'] == block]
            if not block_data.empty:
                times.append(block_data['ExecutionTime_ms'].values[0])
                blocks.append(block)
        
        if times:
            plt.plot(blocks, times, 'o-', linewidth=2, markersize=8, label=f'Grid Size {size}x{size}')
    
    plt.xlabel('CUDA Block Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('CUDA Performance with Different Block Sizes')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('plots/cuda_block_comparison.png', dpi=300)

def plot_opencl_workgroups(opencl_data, workgroup_sizes):
    plt.figure(figsize=(12, 8))
    
    grid_sizes = sorted(opencl_data['GridSize'].unique())
    
    for size in grid_sizes:
        size_data = opencl_data[opencl_data['GridSize'] == size]
        times = []
        wg_sizes = []
        
        for wg in workgroup_sizes:
            wg_data = size_data[size_data['WorkgroupSize'] == wg]
            if not wg_data.empty:
                times.append(wg_data['ExecutionTime_ms'].values[0])
                wg_sizes.append(wg)
        
        if times:
            plt.plot(wg_sizes, times, 'o-', linewidth=2, markersize=8, label=f'Grid Size {size}x{size}')
    
    plt.xlabel('OpenCL Workgroup Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('OpenCL Performance with Different Workgroup Sizes')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('plots/opencl_workgroup_comparison.png', dpi=300)

if __name__ == "__main__":
    main()
